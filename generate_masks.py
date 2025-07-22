#!/usr/bin/env python3
"""
FastSAM Mask Generation Script for Ego-Exo4d Dataset

This script generates masks using FastSAM for all video frames specified in the 
Ego-Exo4d dataset JSON files, organized for the O-MaMa training pipeline.

Author: Claude Code
Date: 2025-07-22
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import logging

# FastSAM path will be set via argparse

from fastsam import FastSAM, FastSAMPrompt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EgoExo4DMaskGenerator:
    def __init__(self, 
                 dataset_root: str,
                 fastsam_model_path: str,
                 fastsam_path: str,
                 device: str = 'cuda',
                 conf_threshold: float = 0.4,
                 iou_threshold: float = 0.9,
                 retina_masks: bool = True,
                 imgsz: int = 1024):
        """
        Initialize the mask generator.
        
        Args:
            dataset_root: Path to Ego-Exo4d dataset root directory
            fastsam_model_path: Path to FastSAM model weights
            device: Device to run inference on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for FastSAM
            iou_threshold: IoU threshold for FastSAM NMS
        """
        self.dataset_root = Path(dataset_root)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        
        # Add FastSAM to Python path
        sys.path.append(fastsam_path)
        
        # Initialize FastSAM model
        logger.info(f"Loading FastSAM model from {fastsam_model_path}")
        self.model = FastSAM(fastsam_model_path)
        
        # Create output directories
        self.mask_dirs = {
            'train_ego2exo': self.dataset_root / 'Masks_TRAIN_EGO2EXO',
            'train_exo2ego': self.dataset_root / 'Masks_TRAIN_EXO2EGO', 
            'val_ego2exo': self.dataset_root / 'Masks_VAL_EGO2EXO',
            'val_exo2ego': self.dataset_root / 'Masks_VAL_EXO2EGO',
            'test_ego2exo': self.dataset_root / 'Masks_TEST_EGO2EXO',
            'test_exo2ego': self.dataset_root / 'Masks_TEST_EXO2EGO'
        }
        
        # Create directories if they don't exist
        for mask_dir in self.mask_dirs.values():
            mask_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {mask_dir}")

    def load_json_pairs(self, json_file: Path) -> list:
        """Load and parse JSON pair file."""
        logger.info(f"Loading pairs from {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract pairs - JSON format may vary, adapt as needed
        pairs = []
        if isinstance(data, list):
            pairs = data
        elif isinstance(data, dict):
            # Handle different JSON structures
            if 'pairs' in data:
                pairs = data['pairs']
            else:
                # Assume the dict values are the pairs
                pairs = list(data.values())
        
        logger.info(f"Loaded {len(pairs)} pairs from {json_file}")
        return pairs

    def convert_json_path_to_local(self, json_path: str) -> Path:
        """
        Convert JSON path format to local file system path.
        
        JSON format: "/hddsdb/segswap/trainval//take_id//camera//object_name//rgb//frame_number"
        Local format: "dataset_root/processed/take_id/camera/frame_number.jpg"
        """
        # Remove any leading path components and split
        path_parts = json_path.strip('/').split('//')
        path_parts = [p for p in path_parts if p]  # Remove empty parts
        
        # Find the take_id (UUID format) in the path
        take_id = None
        camera = None
        frame_number = None
        
        for i, part in enumerate(path_parts):
            if len(part) == 36 and part.count('-') == 4:  # UUID format
                take_id = part
                if i + 1 < len(path_parts):
                    camera = path_parts[i + 1]
                # Frame number is at the end
                frame_number = path_parts[-1]
                break
        
        if not all([take_id, camera, frame_number]):
            logger.warning(f"Could not parse JSON path: {json_path}")
            return None
        
        # Construct local path: processed/take_id/camera/frame_number.jpg
        return self.dataset_root / 'processed' / take_id / camera / f"{frame_number}.jpg"

    def generate_masks_for_image(self, image_path: Path) -> tuple:
        """
        Generate masks for a single image using FastSAM.
        
        Returns:
            tuple: (masks_array, boxes_array) or (None, None) if failed
        """
        try:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None, None
            
            # Run FastSAM inference
            everything_results = self.model(
                str(image_path),
                device=self.device,
                retina_masks=self.retina_masks,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold
            )
            
            prompt_process = FastSAMPrompt(str(image_path), everything_results, device=self.device)
            
            # Get all masks (everything mode)
            ann = prompt_process.everything_prompt()
            
            if ann is None or len(ann) == 0:
                logger.warning(f"No masks generated for {image_path}")
                return None, None
            
            # Extract masks and boxes
            masks = []
            boxes = []
            
            for mask_data in ann:
                if 'segmentation' in mask_data and 'bbox' in mask_data:
                    # Extract mask
                    mask = mask_data['segmentation']
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    masks.append(mask.astype(np.uint8))
                    
                    # Extract bounding box [x, y, w, h] -> [x1, y1, x2, y2]
                    bbox = mask_data['bbox']
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
            
            if len(masks) == 0:
                logger.warning(f"No valid masks extracted from {image_path}")
                return None, None
            
            # Convert to numpy arrays
            masks_array = np.stack(masks, axis=0)  # Shape: (N, H, W)
            boxes_array = np.array(boxes, dtype=np.float32)  # Shape: (N, 4)
            
            logger.debug(f"Generated {len(masks)} masks for {image_path}")
            return masks_array, boxes_array
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None, None

    def save_masks(self, masks: np.ndarray, boxes: np.ndarray, 
                   output_dir: Path, take_id: str, camera: str, frame_idx: str):
        """
        Save masks and boxes in the expected format.
        
        Args:
            masks: Array of masks (N, H, W)
            boxes: Array of bounding boxes (N, 4)
            output_dir: Base output directory
            take_id: Take ID (UUID)
            camera: Camera name
            frame_idx: Frame index/number
        """
        try:
            # Create subdirectories
            take_dir = output_dir / take_id
            camera_dir = take_dir / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Save masks as compressed NPZ
            masks_file = camera_dir / f"{frame_idx}_masks.npz"
            np.savez_compressed(masks_file, masks=masks)
            
            # Save boxes as NPY
            boxes_file = camera_dir / f"{frame_idx}_boxes.npy"
            np.save(boxes_file, boxes)
            
            logger.debug(f"Saved masks to {masks_file}")
            logger.debug(f"Saved boxes to {boxes_file}")
            
        except Exception as e:
            logger.error(f"Error saving masks for {take_id}/{camera}/{frame_idx}: {str(e)}")

    def parse_image_path(self, image_path: Path) -> tuple:
        """
        Parse image path to extract take_id, camera, and frame_idx.
        
        Expected format: .../processed/take_id/camera/frame.jpg
        
        Returns:
            tuple: (take_id, camera, frame_idx)
        """
        parts = image_path.parts
        
        # Find 'processed' directory in path
        try:
            processed_idx = parts.index('processed')
            take_id = parts[processed_idx + 1]
            camera = parts[processed_idx + 2]
            frame_file = parts[processed_idx + 3]
            frame_idx = frame_file.split('.')[0]  # Remove extension
            
            return take_id, camera, frame_idx
        except (ValueError, IndexError):
            logger.error(f"Could not parse path structure: {image_path}")
            return None, None, None

    def process_json_file(self, json_file: Path, output_key: str, resume: bool = True):
        """
        Process a single JSON file and generate masks for all pairs.
        
        Args:
            json_file: Path to JSON pair file
            output_key: Key for output directory (e.g., 'train_ego2exo')
            resume: Whether to skip already processed images
        """
        logger.info(f"Processing {json_file} -> {output_key}")
        
        pairs = self.load_json_pairs(json_file)
        output_dir = self.mask_dirs[output_key]
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each pair
        for pair_idx, pair in enumerate(tqdm(pairs, desc=f"Processing {json_file.name}")):
            try:
                # Handle different JSON pair formats
                if isinstance(pair, list) and len(pair) >= 2:
                    # Format: [ego_path, exo_path] or [ego_rgb, ego_mask, exo_rgb, exo_mask]
                    ego_path = pair[0]
                    exo_path = pair[2] if len(pair) > 2 else pair[1]
                elif isinstance(pair, dict):
                    # Handle dictionary format
                    ego_path = pair.get('ego_rgb', pair.get('ego', ''))
                    exo_path = pair.get('exo_rgb', pair.get('exo', ''))
                else:
                    logger.warning(f"Unknown pair format at index {pair_idx}: {pair}")
                    continue
                
                # Process both ego and exo images
                for img_path_str in [ego_path, exo_path]:
                    if not img_path_str:
                        continue
                        
                    # Convert JSON path to local path
                    image_path = self.convert_json_path_to_local(img_path_str)
                    if image_path is None:
                        continue
                    
                    # Parse path components
                    take_id, camera, frame_idx = self.parse_image_path(image_path)
                    if not all([take_id, camera, frame_idx]):
                        continue
                    
                    # Check if already processed (resume functionality)
                    masks_file = output_dir / take_id / camera / f"{frame_idx}_masks.npz"
                    if resume and masks_file.exists():
                        skipped_count += 1
                        continue
                    
                    # Generate masks
                    masks, boxes = self.generate_masks_for_image(image_path)
                    
                    if masks is not None and boxes is not None:
                        # Save masks and boxes
                        self.save_masks(masks, boxes, output_dir, take_id, camera, frame_idx)
                        processed_count += 1
                    else:
                        error_count += 1
                        
            except Exception as e:
                logger.error(f"Error processing pair {pair_idx}: {str(e)}")
                error_count += 1
        
        logger.info(f"Completed {json_file.name}: {processed_count} processed, "
                   f"{skipped_count} skipped, {error_count} errors")

    def process_all_datasets(self, splits: list = None, directions: list = None, resume: bool = True):
        """
        Process all dataset JSON files.
        
        Args:
            splits: List of splits to process ['train', 'val', 'test']
            directions: List of directions ['ego2exo', 'exo2ego']
            resume: Whether to resume from existing progress
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        if directions is None:
            directions = ['ego2exo', 'exo2ego']
        
        dataset_jsons_dir = self.dataset_root / 'dataset_jsons'
        
        # Map of JSON files to output directories
        json_mappings = {
            ('train', 'ego2exo'): ('train_egoexo_pairs.json', 'train_ego2exo'),
            ('train', 'exo2ego'): ('train_exoego_pairs.json', 'train_exo2ego'),
            ('val', 'ego2exo'): ('val_egoexo_pairs.json', 'val_ego2exo'),
            ('val', 'exo2ego'): ('val_exoego_pairs.json', 'val_exo2ego'),
            ('test', 'ego2exo'): ('test_egoexo_pairs.json', 'test_ego2exo'),
            ('test', 'exo2ego'): ('test_exoego_pairs.json', 'test_exo2ego'),
        }
        
        for split in splits:
            for direction in directions:
                key = (split, direction)
                if key not in json_mappings:
                    logger.warning(f"No mapping found for {split} {direction}")
                    continue
                
                json_file_name, output_key = json_mappings[key]
                json_file = dataset_jsons_dir / json_file_name
                
                if not json_file.exists():
                    logger.warning(f"JSON file not found: {json_file}")
                    continue
                
                self.process_json_file(json_file, output_key, resume)

def main():
    parser = argparse.ArgumentParser(description='Generate masks for Ego-Exo4d dataset using FastSAM')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to Ego-Exo4d dataset root directory')
    parser.add_argument('--fastsam_path', type=str, default='/home/compu/JinProjects/jinprojects/FastSAM',
                       help='Path to FastSAM directory')
    parser.add_argument('--fastsam_model', type=str, default='FastSAM-x.pt',
                       help='FastSAM model filename')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference (cuda/cpu)')
    parser.add_argument('--conf_threshold', type=float, default=0.4,
                       help='Confidence threshold for FastSAM')
    parser.add_argument('--iou_threshold', type=float, default=0.9,
                       help='IoU threshold for FastSAM NMS')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'],
                       default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--directions', nargs='+', choices=['ego2exo', 'exo2ego'],
                       default=['ego2exo', 'exo2ego'],
                       help='Correspondence directions to process')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from existing progress (skip already processed images)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Process all images (overwrite existing)')
    parser.add_argument('--retina_masks', action='store_true', default=True,
                       help='Use retina masks in FastSAM')
    parser.add_argument('--no-retina_masks', dest='retina_masks', action='store_false',
                       help='Disable retina masks')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Image size for FastSAM inference')
    
    args = parser.parse_args()
    
    # Initialize generator
    logger.info("Initializing FastSAM mask generator...")
    fastsam_model_path = os.path.join(args.fastsam_path, args.fastsam_model)
    generator = EgoExo4DMaskGenerator(
        dataset_root=args.dataset_root,
        fastsam_model_path=fastsam_model_path,
        fastsam_path=args.fastsam_path,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        retina_masks=args.retina_masks,
        imgsz=args.imgsz
    )
    
    # Process datasets
    logger.info(f"Starting mask generation for splits: {args.splits}, directions: {args.directions}")
    generator.process_all_datasets(
        splits=args.splits,
        directions=args.directions,
        resume=args.resume
    )
    
    logger.info("Mask generation completed!")

if __name__ == '__main__':
    main()