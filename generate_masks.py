#!/usr/bin/env python3
"""
Refactored EgoExo4D FastSAM mask generator
- Unified single/multi-process code path
- Removed duplicate functions (worker/local variants)
- Centralized logging setup
- Argparse defaults cleaned (no interactive prompt). Set num_processes default=1
- If num_processes == 1 -> run in current process (no mp overhead)
# Train split - ego2exo direction
echo "Processing train ego2exo..."
python generate_masks.py --num_processes 10 --splits train --directions ego2exo --gpu_device 0

# Train split - exo2ego direction  
echo "Processing train exo2ego..."
python generate_masks.py --num_processes 10 --splits train --directions exo2ego --gpu_device 1

# Val split - ego2exo direction
echo "Processing val ego2exo..."
python generate_masks.py --num_processes 10 --splits val --directions ego2exo --gpu_device 2

# Val split - exo2ego direction
echo "Processing val exo2ego..."
python generate_masks.py --num_processes 10 --splits val --directions exo2ego --gpu_device 3
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Multiprocessing only when requested (>1 processes)
import multiprocessing as mp
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
        prompt = FastSAMPrompt(str(image_path), everything_results, device=device)


        mask_ann = prompt.results[0].masks.data # ex : torch.Size([N, H, W])
        box_ann = prompt.results[0].boxes.xywh # ex : torch.Size([N, 4]) # 0:x1, 1:y1, 2:w, 3:h


        if mask_ann is None or len(mask_ann) == 0:
            logger.warning(f"No masks generated for {image_path}")
            return None, None

        # Convert torch tensors to numpy
        masks = mask_ann.cpu().numpy().astype(np.uint8)  # Shape: [N, H, W]
        boxes = box_ann.cpu().numpy().astype(np.float32)  # Shape: [N, 4], format: [x, y, w, h]

        return masks, boxes
    except Exception as e:
        logger.error(f"FastSAM error on {image_path}: {e}")
        return None, None


def save_outputs(masks: np.ndarray, boxes: np.ndarray,
                 out_root: Path, take_id: str, camera: str,
                 frame_idx: str, logger: logging.Logger) -> None:
    try:
        cam_dir = out_root / take_id / camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cam_dir / f"{frame_idx}_masks.npz", arr_0=masks)
        np.save(cam_dir / f"{frame_idx}_boxes.npy", boxes)
    except Exception as e:
        logger.error(f"Save error {take_id}/{camera}/{frame_idx}: {e}")


def parse_processed_path(image_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (take_id, camera, frame_idx) from .../processed/<take>/<cam>/<frame>.jpg"""
    parts = image_path.parts
    try:
        idx = parts.index('processed')
        take_id = parts[idx + 1]
        camera = parts[idx + 2]
        frame = Path(parts[idx + 3]).stem
        return take_id, camera, frame
    except (ValueError, IndexError):
        return None, None, None


def json_key_to_local(path_str: str, dataset_root: Path, logger: logging.Logger) -> Optional[Path]:
    """Convert JSON path format to local filesystem path."""
    parts = [p for p in path_str.strip('/').split('//') if p]
    take_id = camera = frame = None
    for i, p in enumerate(parts):
        if len(p) == 36 and p.count('-') == 4:  # crude UUID check
            take_id = p
            camera = parts[i + 1] if i + 1 < len(parts) else None
            frame = parts[-1]
            break
    if not (take_id and camera and frame):
        logger.warning(f"Could not parse JSON path: {path_str}")
        return None
    return dataset_root / 'processed' / take_id / camera / f"{frame}.jpg"


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
    def __init__(self, dataset_root: str, fastsam_model_path: str, fastsam_path: str,
                 logger: logging.Logger):
        self.dataset_root = Path(dataset_root)
        self.fastsam_model_path = fastsam_model_path
        self.fastsam_path = fastsam_path
        self.logger = logger

        # Output dirs
        self.mask_dirs = {
            'train_ego2exo': self.dataset_root / 'Masks_TRAIN_EGO2EXO',
            'train_exo2ego': self.dataset_root / 'Masks_TRAIN_EXO2EGO',
            'val_ego2exo': self.dataset_root / 'Masks_VAL_EGO2EXO',
            'val_exo2ego': self.dataset_root / 'Masks_VAL_EXO2EGO',
            'test_ego2exo': self.dataset_root / 'Masks_TEST_EGO2EXO',
            'test_exo2ego': self.dataset_root / 'Masks_TEST_EXO2EGO',
        }
        for d in self.mask_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # -------- collection helpers ---------
    def collect_image_jobs(self, splits: List[str], directions: List[str], resume: bool) -> List[Tuple[str, str, str, str, str]]:
        mapping = {
            ('train', 'ego2exo'): ('train_egoexo_pairs.json', 'train_ego2exo'),
            ('train', 'exo2ego'): ('train_exoego_pairs.json', 'train_exo2ego'),
            ('val', 'ego2exo'): ('val_egoexo_pairs.json', 'val_ego2exo'),
            ('val', 'exo2ego'): ('val_exoego_pairs.json', 'val_exo2ego'),
            ('test', 'ego2exo'): ('test_egoexo_pairs.json', 'test_ego2exo'),
            ('test', 'exo2ego'): ('test_exoego_pairs.json', 'test_exo2ego'),
        }
        dataset_jsons_dir = self.dataset_root / 'dataset_jsons'
        jobs = []

        for split in splits:
            for direction in directions:
                key = (split, direction)
                if key not in mapping:
                    continue
                json_name, out_key = mapping[key]
                json_file = dataset_jsons_dir / json_name
                if not json_file.exists():
                    self.logger.warning(f"Missing JSON {json_file}")
                    continue
                pairs = load_pairs(json_file, self.logger)
                out_dir = self.mask_dirs[out_key]

                for pair in pairs:
                    if isinstance(pair, list) and len(pair) >= 2:
                        ego_path = pair[0]
                        exo_path = pair[2] if len(pair) > 2 else pair[1]
                    elif isinstance(pair, dict):
                        ego_path = pair.get('ego_rgb', pair.get('ego', ''))
                        exo_path = pair.get('exo_rgb', pair.get('exo', ''))
                    else:
                        continue

                    for img_str in (ego_path, exo_path):
                        if not img_str:
                            continue
                        img_path = json_key_to_local(img_str, self.dataset_root, self.logger)
                        if img_path is None or not img_path.exists():
                            continue
                        take, cam, frame = parse_processed_path(img_path)
                        if not (take and cam and frame):
                            continue
                        mask_file = out_dir / take / cam / f"{frame}_masks.npz"
                        if resume and mask_file.exists():
                            continue
                        jobs.append((str(img_path), str(out_dir), take, cam, frame))
        self.logger.info(f"Collected {len(jobs)} images to process")
        return jobs

    # -------- processing paths ---------
    def _process_job(self, job, model, device, conf, iou, retina_masks, imgsz, logger):
        img_path, out_dir, take, cam, frame = job
        masks, boxes = run_fastsam_on_image(Path(img_path), model, device, conf, iou, retina_masks, imgsz, logger)
        if masks is not None and boxes is not None:
            save_outputs(masks, boxes, Path(out_dir), take, cam, frame, logger)
            return True
        return False

    def run_single_process(self, jobs: List[Tuple[str, str, str, str, str]],
                           device: str, conf: float, iou: float,
                           retina_masks: bool, imgsz: int) -> None:
        self.logger.info("Running in single-process mode")
        model = FastSAM(self.fastsam_model_path)
        processed, errors = 0, 0
        for job in tqdm(jobs, desc="Images"):
            ok = self._process_job(job, model, device, conf, iou, retina_masks, imgsz, self.logger)
            processed += int(ok)
            errors += int(not ok)
        self.logger.info(f"Done. processed={processed}, errors={errors}")

    # -------- multiprocessing ---------
    @staticmethod
    def worker(proc_id: int, jobs: List[Tuple[str, str, str, str, str]],
               device: str, fastsam_model_path: str,
               conf: float, iou: float, retina_masks: bool, imgsz: int,
               progress_q: Queue, log_dir: Path, test_mode: bool = False):
        logger = setup_logger(log_dir, proc_id)
        try:
            logger.info(f"Worker {proc_id} | {len(jobs)} images | device {device}")
            model = FastSAM(fastsam_model_path)
            start = time.time()
            count = 0
            for job in jobs:
                if test_mode and time.time() - start > 10:
                    logger.info("Test mode timeout; exiting")
                    break
                ok = EgoExo4DMaskGenerator._static_process_job(job, model, device, conf, iou, retina_masks, imgsz, logger)
                if ok:
                    count += 1
                    progress_q.put(1)
            logger.info(f"Worker {proc_id} finished. processed={count}")
        except Exception as e:
            logger.error(f"Worker {proc_id} crashed: {e}")

    @staticmethod
    def _static_process_job(job, model, device, conf, iou, retina_masks, imgsz, logger):
        img_path, out_dir, take, cam, frame = job
        masks, boxes = run_fastsam_on_image(Path(img_path), model, device, conf, iou, retina_masks, imgsz, logger)
        if masks is None or boxes is None:
            return False
        save_outputs(masks, boxes, Path(out_dir), take, cam, frame, logger)
        return True

    def run_multi_process(self, jobs: List[Tuple[str, str, str, str, str]],
                          num_processes: int, device: str,
                          conf: float, iou: float, retina_masks: bool, imgsz: int,
                          test_mode: bool, log_dir: Path):
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
                              progress_q, log_dir, test_mode))
            p.start()
            procs.append(p)
            self.logger.info(f"Spawned PID {p.pid}")

        total = len(jobs)
        processed = 0
        start_time = time.time()
        while any(p.is_alive() for p in procs):
            try:
                while not progress_q.empty():
                    progress_q.get_nowait()
                    processed += 1
                    elapsed = time.time() - start_time
                    if processed:
                        eta = (elapsed / processed) * (total - processed)
                        self.logger.info(f"Progress {processed}/{total} ({processed/total*100:.1f}%) ETA {eta/60:.1f}m")
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Interrupt detected. Terminating children...")
                for p in procs:
                    p.terminate()
                break

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

    parser.add_argument('--conf_threshold', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.9,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Input resolution for FastSAM')
    parser.add_argument('--retina_masks', action='store_true', default=True,
                        help='Use retina masks (FastSAM option)')
    parser.add_argument('--no-retina_masks', dest='retina_masks', action='store_false')

    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'],
                        help='Dataset splits to process')
    parser.add_argument('--directions', nargs='+', choices=['ego2exo', 'exo2ego'], default=['ego2exo', 'exo2ego'],
                        help='Directions to process')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip already processed frames')
    parser.add_argument('--no-resume', dest='resume', action='store_false')

    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Workers exit after 10s (debug)')

    parser.add_argument('--log_dir', type=str, default='logs_masks',
                        help='Where to store log files')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Inject FastSAM path
    sys.path.append(args.fastsam_path)

    log_dir = Path(args.log_dir)
    logger = setup_logger(log_dir)

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Initialize generator
    gen = EgoExo4DMaskGenerator(dataset_root=args.dataset_root,
                                fastsam_model_path=os.path.join(args.fastsam_path, args.fastsam_model),
                                fastsam_path=args.fastsam_path,
                                logger=logger)

    device = f"cuda:{args.gpu_device}"

    # Collect jobs
    jobs = gen.collect_image_jobs(args.splits, args.directions, args.resume)
    if not jobs:
        logger.info("Nothing to do. Exiting.")
        return
    if args.num_processes <= 1:
        gen.run_single_process(jobs, device, args.conf_threshold, args.iou_threshold,
                               args.retina_masks, args.imgsz)
    else:
        gen.run_multi_process(jobs, args.num_processes, device, args.conf_threshold,
                              args.iou_threshold, args.retina_masks, args.imgsz,
                              args.test_mode, log_dir)

    logger.info("Mask generation completed!")


if __name__ == '__main__':
    main()
