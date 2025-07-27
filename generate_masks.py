#!/usr/bin/env python3
"""
Refactored EgoExo4D FastSAM mask generator
- Unified single/multi-process code path
- Removed duplicate functions (worker/local variants)
- Centralized logging setup
- Argparse defaults cleaned (no interactive prompt). Set num_processes default=1
- If num_processes == 1 -> run in current process (no mp overhead)
- Fixed ego/exo detection logic to use camera names instead of positional indices
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Multiprocessing only when requested (>1 processes)
from multiprocessing import Process, Queue

# ----------------------------
# FastSAM imports are injected via --fastsam_path
# ----------------------------
from fastsam import FastSAM, FastSAMPrompt

# ============================
# Logging
# ============================
def setup_logger(log_dir: Path, process_id: Optional[int] = None) -> logging.Logger:
    """Create a logger writing to file + stdout. One logger per process.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"Process-{process_id}" if process_id is not None else "main"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers when reusing in forks
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - ' + name + ' - %(message)s')

    fh = logging.FileHandler(log_dir / f"mask_generation_{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ============================
# Core util functions
# ============================

def get_fallback_masks_boxes(image_path: Path, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Create fallback zero masks and boxes when FastSAM fails."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not load image for fallback: {image_path}")
            # Use default dimensions if image loading fails
            h, w = 1024, 1024
        else:
            h, w = img.shape[:2]
        
        # Return zero mask (1, H, W) and zero box (1, 4)
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        fallback_box = np.zeros((1, 4), dtype=np.float32)
        return fallback_mask, fallback_box
    except Exception as e:
        logger.warning(f"Error creating fallback for {image_path}: {e}")
        # Use default dimensions as last resort
        fallback_mask = np.zeros((1024, 1024), dtype=np.uint8)
        fallback_box = np.zeros((1, 4), dtype=np.float32)
        return fallback_mask, fallback_box

def run_fastsam_on_image(image_path: Path,
                         model: FastSAM,
                         device: str,
                         conf: float,
                         iou: float,
                         retina_masks: bool,
                         imgsz: int,
                         logger: logging.Logger) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run FastSAM and return (masks[N,H,W], boxes[N,4]) in numpy, or (None,None)."""
    try:
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None, None
        
        everything_results = model(
            str(image_path),
            device=device,
            retina_masks=retina_masks,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )
        
        if not everything_results or len(everything_results) == 0:
            # mask의 경우 shape (H, W) 인 zeros를 반환
            # box의 경우 shape (1, 4) 인 zeros를 반환
            logger.warning(f"No results from FastSAM for {image_path}")
            return get_fallback_masks_boxes(image_path, logger)
            
        prompt = FastSAMPrompt(str(image_path), everything_results, device=device)

        mask_ann = prompt.results[0].masks.data
        box_ann = prompt.results[0].boxes.xywh

        # Convert torch tensors to numpy
        masks = mask_ann.cpu().numpy().astype(np.uint8)  # Shape: [N, H, W]
        boxes = box_ann.cpu().numpy().astype(np.float32)  # Shape: [N, 4], format: [x, y, w, h]

        return masks, boxes
    
    except Exception as e:
        logger.error(f"FastSAM error on {image_path}: {e}")
        return get_fallback_masks_boxes(image_path, logger)


# Global cache for created directories to avoid redundant mkdir calls
_created_dirs: Set[Path] = set()

def save_outputs(masks: np.ndarray, boxes: np.ndarray,
                 out_root: Path, take_id: str, camera: str,
                 frame_idx: str, logger: logging.Logger) -> None:
    try:
        cam_dir = out_root / take_id / camera
        # Only create directory if not already created
        if cam_dir not in _created_dirs:
            cam_dir.mkdir(parents=True, exist_ok=True)
            _created_dirs.add(cam_dir)
        np.savez_compressed(cam_dir / f"{frame_idx}_masks.npz", arr_0=masks)
        np.save(cam_dir / f"{frame_idx}_boxes.npy", boxes)
    except Exception as e:
        logger.error(f"Save error {take_id}/{camera}/{frame_idx}: {e}")


def parse_processed_path(image_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (take_id, camera, frame_idx) from .../processed/<take>/<cam>/<frame>.jpg"""
    parts = image_path.parts
    try:
        idx = parts.index('processed')
        return parts[idx + 1], parts[idx + 2], Path(parts[idx + 3]).stem
    except (ValueError, IndexError):
        return None, None, None


def json_key_to_local(path_str: str, dataset_root: Path, logger: logging.Logger) -> Optional[Path]:
    """Convert JSON path format to local filesystem path."""
    UUID_LENGTH = 36
    UUID_DASHES = 4
    
    parts = [p for p in path_str.strip('/').split('//') if p]
    for i, p in enumerate(parts):
        if len(p) == UUID_LENGTH and p.count('-') == UUID_DASHES:  # crude UUID check
            take_id = p
            camera = parts[i + 1] if i + 1 < len(parts) else None
            frame = parts[-1]
            if take_id and camera and frame:
                return dataset_root / 'processed' / take_id / camera / f"{frame}.jpg"
            break
    
    logger.warning(f"Could not parse JSON path: {path_str}")
    return None

def is_exo_camera(camera_name: str) -> bool:
    """Determine if camera is exocentric based on camera name.
    Exocentric cameras contain 'cam' in their name, egocentric don't.
    """
    return 'cam' in camera_name.lower()


def load_pairs(json_file: Path, logger: logging.Logger) -> List:
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get('pairs', list(data.values()))
    logger.warning(f"Unknown JSON structure in {json_file}")
    return []

# ============================
# Class wrapper
# ============================
class EgoExo4DMaskGenerator:
    def __init__(self, dataset_root: str, fastsam_model_path: str,
                 logger: logging.Logger):
        self.dataset_root = Path(dataset_root)
        self.fastsam_model_path = fastsam_model_path
        self.logger = logger

        # Output dirs
        self.mask_dirs = {
            'train_egoexo': self.dataset_root / 'Masks_TRAIN_EGO2EXO',
            'train_exoego': self.dataset_root / 'Masks_TRAIN_EXO2EGO',
            'val_egoexo': self.dataset_root / 'Masks_VAL_EGO2EXO',
            'val_exoego': self.dataset_root / 'Masks_VAL_EXO2EGO',
            'test_egoexo': self.dataset_root / 'Masks_TEST_EGO2EXO',
            'test_exoego': self.dataset_root / 'Masks_TEST_EXO2EGO',
        }
        for d in self.mask_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # -------- collection helpers ---------
    def collect_missing_file_jobs(self, missing_file_path: str) -> List[Tuple[str, str, str, str, str]]:
        """Collect jobs from missing_mask_files.json."""
        jobs = []
        
        try:
            with open(missing_file_path, 'r') as f:
                missing_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load missing file JSON: {e}")
            return jobs
        
        # Process each split/direction combination
        for json_file in ['train_egoexo_pairs.json', 'train_exoego_pairs.json', 'val_egoexo_pairs.json', 'val_exoego_pairs.json', 'test_egoexo_pairs.json', 'test_exoego_pairs.json']:
            if json_file not in missing_data:
                continue
                
            # Get the output directory for this split/direction
            out_dir = self.mask_dirs.get('_'.join(json_file.split('_')[:2]))
            if not out_dir:
                self.logger.warning(f"Unknown mask directory : {out_dir}")
                continue
            
            # Process each missing file entry
            for entry in missing_data[json_file]:
                take_id = entry['take_id']
                camera = entry['camera']
                frame = entry['frame']
                
                # Construct image path
                img_path = self.dataset_root / 'processed' / take_id / camera / f"{frame}.jpg"
                
                # Only add if image exists
                if img_path.exists():
                    jobs.append((str(img_path), str(out_dir), take_id, camera, str(frame)))
                else:
                    self.logger.warning(f"Image not found for missing mask: {img_path}")
        
        self.logger.info(f"Collected {len(jobs)} missing file jobs to process")
        return jobs
    
    def collect_image_jobs(self, split: str, direction: str, resume: bool) -> List[Tuple[str, str, str, str, str]]:
        mapping = {
            ('train', 'ego2exo'): ('train_egoexo_pairs.json', 'train_egoexo'),
            ('train', 'exo2ego'): ('train_exoego_pairs.json', 'train_exoego'),
            ('val', 'ego2exo'): ('val_egoexo_pairs.json', 'val_egoexo'),
            ('val', 'exo2ego'): ('val_exoego_pairs.json', 'val_exoego'),
            ('test', 'ego2exo'): ('test_egoexo_pairs.json', 'test_egoexo'),
            ('test', 'exo2ego'): ('test_exoego_pairs.json', 'test_exoego'),
        }
        dataset_jsons_dir = self.dataset_root / 'dataset_jsons'
        jobs = []

        key = (split, direction)
        if key not in mapping:
            self.logger.warning(f"Invalid split-direction combination: {key}")
            return jobs
        
        json_name, out_key = mapping[key]
        json_file = dataset_jsons_dir / json_name
        if not json_file.exists():
            self.logger.warning(f"Missing JSON {json_file}")
            return jobs
        
        pairs = load_pairs(json_file, self.logger)
        out_dir = self.mask_dirs[out_key]

        for pair in pairs:

            assert isinstance(pair, list), "Unknown JSON structure"

            for img_str in pair:
                img_path = json_key_to_local(img_str, self.dataset_root, self.logger)
                if img_path is None or not img_path.exists():
                    continue
                take, cam, frame = parse_processed_path(img_path)
                if not (take and cam and frame):
                    continue
                
                # Validate camera type matches direction
                is_exo = is_exo_camera(cam)
                if direction == 'ego2exo' and not is_exo:
                    continue  # Skip exo cameras for ego2exo direction
                if direction == 'exo2ego' and is_exo:
                    continue  # Skip ego cameras for exo2ego direction
                    
                mask_file = out_dir / take / cam / f"{frame}_masks.npz"
                if resume and mask_file.exists():
                    continue
                jobs.append((str(img_path), str(out_dir), take, cam, frame))
        self.logger.info(f"Collected {len(jobs)} images to process")
        return jobs

    # -------- processing paths ---------
    @staticmethod
    def _process_job(job, model, device, conf, iou, retina_masks, imgsz, logger):
        """Process a single job (image) - unified function for single and multi-process modes."""
        img_path, out_dir, take, cam, frame = job
        
        try:
            masks, boxes = run_fastsam_on_image(Path(img_path), model, device, conf, iou, retina_masks, imgsz, logger)
            if masks is not None and boxes is not None:
                save_outputs(masks, boxes, Path(out_dir), take, cam, frame, logger)
                # Clear GPU memory after processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
        except Exception as e:
            logger.error(f"Job processing error {take}/{cam}/{frame}: {e}")
        
        return False

    def run_single_process(self, jobs: List[Tuple[str, str, str, str, str]],
                           device: str, conf: float, iou: float,
                           retina_masks: bool, imgsz: int) -> None:
        self.logger.info("Running in single-process mode")
        model = FastSAM(self.fastsam_model_path)
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i, job in enumerate(tqdm(jobs, desc="Images")):
            ok = EgoExo4DMaskGenerator._process_job(job, model, device, conf, iou, retina_masks, imgsz, self.logger)
            if ok:
                processed += 1
            else:
                failed += 1
            
            # Batch progress reporting every 100 jobs
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                self.logger.info(f"Progress: {i+1}/{len(jobs)} | Rate: {rate:.1f} img/s | Processed: {processed} | Failed: {failed}")
        
        self.logger.info(f"Done. processed={processed}, failed={failed}")

    # -------- multiprocessing ---------
    @staticmethod
    def worker(proc_id: int, jobs: List[Tuple[str, str, str, str, str]],
               device: str, fastsam_model_path: str,
               conf: float, iou: float, retina_masks: bool, imgsz: int,
               progress_q: Queue, log_dir: Path, job_timeout: Optional[int] = None):
        logger = setup_logger(log_dir, proc_id)
        try:
            logger.info(f"Worker {proc_id} | {len(jobs)} images | device {device}")
            model = FastSAM(fastsam_model_path)
            processed = 0
            failed = 0
            
            # Track job start time if timeout is specified
            if job_timeout is not None:
                job_start = time.time()
            
            for i, job in enumerate(jobs):
                ok = EgoExo4DMaskGenerator._process_job(job, model, device, conf, iou, retina_masks, imgsz, logger)
                
                if ok:
                    processed += 1
                else:
                    failed += 1
                
                # Report progress in batches
                if (i + 1) % 10 == 0 or (i + 1) == len(jobs):
                    progress_q.put({'proc_id': proc_id, 'processed': processed, 'failed': failed, 'total': i + 1})
                
                # Check job timeout if specified
                if job_timeout is not None:
                    job_duration = time.time() - job_start
                    if job_duration > job_timeout:
                        logger.error(f"Worker {proc_id} timeout: exceeded {job_timeout}s limit after {job_duration:.1f}s")
                        break  # Exit the loop and terminate this worker process
            
            logger.info(f"Worker {proc_id} finished. processed={processed}, failed={failed}")
        except Exception as e:
            logger.error(f"Worker {proc_id} crashed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def run_multi_process(self, jobs: List[Tuple[str, str, str, str, str]],
                          num_processes: int, device: str,
                          conf: float, iou: float, retina_masks: bool, imgsz: int,
                          log_dir: Path, debug_timeout: Optional[int] = None):
        self.logger.info(f"Running in multi-process mode ({num_processes})")
        # Split jobs
        chunk = len(jobs) // num_processes
        rem = len(jobs) % num_processes
        job_slices = []
        start = 0
        for i in range(num_processes):
            extra = 1 if i < rem else 0
            end = start + chunk + extra
            job_slices.append(jobs[start:end])
            start = end
            self.logger.info(f"Process {i} -> {len(job_slices[i])} jobs")

        progress_q = Queue()
        procs: List[Process] = []
        for i, js in enumerate(job_slices):
            p = Process(target=EgoExo4DMaskGenerator.worker,
                        args=(i, js, device, self.fastsam_model_path, conf, iou, retina_masks, imgsz,
                              progress_q, log_dir, debug_timeout))
            p.start()
            procs.append(p)
            self.logger.info(f"Spawned PID {p.pid}")

        total = len(jobs)
        processed = 0
        start_time = time.time()
        worker_progress = {}  # Track progress per worker
        worker_stats = {}  # Track detailed stats per worker
        
        while any(p.is_alive() for p in procs):
            try:
                current_time = time.time()
                
                # Process progress updates
                progress_updates = []
                while not progress_q.empty():
                    try:
                        update = progress_q.get_nowait()
                        progress_updates.append(update)
                    except:
                        break
                
                # Aggregate progress updates
                if progress_updates:
                    # Update worker progress tracking
                    for update in progress_updates:
                        proc_id = update['proc_id']
                        worker_progress[proc_id] = update['total']
                        worker_stats[proc_id] = {
                            'processed': update['processed'],
                            'failed': update['failed']
                        }
                    
                    # Calculate totals across all workers
                    processed = sum(worker_progress.values())
                    total_success = sum(stats['processed'] for stats in worker_stats.values())
                    total_failed = sum(stats['failed'] for stats in worker_stats.values())
                    
                    elapsed = current_time - start_time
                    if processed > 0:
                        rate = processed / elapsed
                        eta = (total - processed) / rate if rate > 0 else 0
                        self.logger.info(f"Progress {processed}/{total} ({processed/total*100:.1f}%) | Success: {total_success} | Failed: {total_failed} | Rate: {rate:.1f} img/s | ETA {eta/60:.1f}m")
                
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Interrupt detected. Terminating children...")
                for p in procs:
                    p.terminate()
                break

        # Wait for all processes to finish naturally
        for p in procs:
            p.join()
                
        self.logger.info("All workers finished")

# ============================
# CLI
# ============================

def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate masks for Ego-Exo4D using FastSAM (single/multi process)')

    parser.add_argument('--dataset_root', type=str, default='Ego-Exo4d',
                        help='Ego-Exo4d dataset root')
    parser.add_argument('--fastsam_path', type=str, default='/home/compu/JinProjects/jinprojects/FastSAM',
                        help='FastSAM repo directory (added to sys.path)')
    parser.add_argument('--fastsam_model', type=str, default='FastSAM-x.pt',
                        help='FastSAM weight filename')

    parser.add_argument('--gpu_device', type=int, default=0,
                        help='CUDA device index to use')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of worker processes. 1 = no multiprocessing')
    parser.add_argument('--debug_timeout', type=int, default=None,
                        help='Timeout in seconds for worker processes (default: None, set for debugging)')

    parser.add_argument('--conf_threshold', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.9,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Input resolution for FastSAM')
    parser.add_argument('--retina_masks', type=bool, default=True,
                        help='Use retina masks (FastSAM option, default: True)')

    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                        help='Dataset split to process')
    parser.add_argument('--direction', choices=['ego2exo', 'exo2ego'], default='ego2exo',
                        help='Direction to process')
    parser.add_argument('--missing_file', type=str, default=None,
                        help='Path to missing_mask_files.json file to process only missing masks')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Skip already processed frames when not using missing_file mode')

    parser.add_argument('--log_dir', type=str, default='logs_generating_masks',
                        help='Where to store log files')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Inject FastSAM path
    sys.path.append(args.fastsam_path)

    logger = setup_logger(Path(args.log_dir))

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Initialize generator
    gen = EgoExo4DMaskGenerator(dataset_root=args.dataset_root,
                                fastsam_model_path=os.path.join(args.fastsam_path, args.fastsam_model),
                                logger=logger)

    # Collect jobs based on mode
    if args.missing_file:
        # Missing file mode - use JSON file
        logger.info(f"Using missing file mode with: {args.missing_file}")
        jobs = gen.collect_missing_file_jobs(args.missing_file)
    else:
        # Normal mode - use split/direction
        jobs = gen.collect_image_jobs(args.split, args.direction, args.resume)
    
    if not jobs:
        logger.info("Nothing to do. Exiting.")
        return
    
    device = f"cuda:{args.gpu_device}"
    if args.num_processes <= 1:
        gen.run_single_process(jobs, device, args.conf_threshold, args.iou_threshold,
                               args.retina_masks, args.imgsz)
    else:
        gen.run_multi_process(jobs, args.num_processes, device, args.conf_threshold,
                              args.iou_threshold, args.retina_masks, args.imgsz,
                              Path(args.log_dir), args.debug_timeout)

    logger.info("Mask generation completed!")


if __name__ == '__main__':
    main()
