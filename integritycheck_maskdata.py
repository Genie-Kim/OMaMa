#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame-level data integrity verification script for O-MaMa dataset.
Checks existence of mask/box files for each frame referenced in JSON pairs.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def parse_json_paths(json_entry):
    """
    Parse JSON entry to extract frame information.
    
    Args:
        json_entry (list): 4-element list [ego_rgb, ego_mask, exo_rgb, exo_mask]
        
    Returns:
        dict: Parsed information with take_id, cameras, frames, etc.
    """
    if len(json_entry) != 4:
        return None

    try:
        # Parse path: processed//{take_id}//{camera}//{object}/rgb//{frame}
        cameras = set()

        for path in json_entry:
            parts = path.split('//')
            if len(parts) >= 6:
                take_id = parts[1]
                cameras.add(parts[2])
                frame = int(parts[5])
            else:
                return None
        
        for cam in cameras:
            if 'aria' in cam:
                ego_camera=cam
            else:
                exo_camera=cam
        
        return {
            'take_id': take_id,
            'ego_camera': ego_camera,
            'exo_camera': exo_camera,
            'frame': frame,
        }
        
    except (IndexError, ValueError) as e:
        print(f"Error parsing JSON entry: {e}")
        return None


def check_frame_files(take_id, camera, frame_number, masks_dir):
    """
    Check existence and validity of mask/box files for a specific frame.
    
    Args:
        take_id (str): UUID of the take
        camera (str): Camera name
        frame_number (int): Frame number
        masks_dir (Path): Path to masks directory
        debug_mode (bool): Enable debug output
        
    Returns:
        dict: File existence and validity status with structured missing info
    """
    # Construct expected file paths
    take_dir = masks_dir / take_id / camera
    mask_file = take_dir / f"{frame_number}_masks.npz"
    box_file = take_dir / f"{frame_number}_boxes.npy"
    
    result = {
        'take_id': take_id,
        'camera': camera,
        'frame': frame_number,
        'success': True,
    }
    
    # Check mask file
    mask_missing = False
    if mask_file.exists():
        try:
            # Try to load the npz file to verify it's valid
            with np.load(mask_file, allow_pickle=True) as data:
                if len(data.files) == 0:
                    mask_missing = True
        except Exception as e:
            mask_missing = True
    else:
        mask_missing = True
    
    # Check box file
    box_missing = False
    if box_file.exists():
        try:
            # Try to load the npy file to verify it's valid
            boxes = np.load(box_file)
            if boxes.size == 0:
                box_missing = True
        except Exception as e:
            box_missing = True
    else:
        box_missing = True
    
    result['success'] = (not box_missing) and (not mask_missing)
    
    return result


def verify_frame_integrity(dataset_dir, debug_mode=False):
    """
    Main frame-level integrity verification function.
    
    Args:
        dataset_dir (str): Path to dataset root directory
        debug_mode (bool): Enable debug mode with breakpoints
        limit (int): Limit number of entries to process (for debugging)
        output_file (file): Optional file handle for output logging
        
    Returns:
        tuple: (overall_success, detailed_results, json_data)
    """
    
    print("=" * 60)
    print("O-MaMa Frame-Level Dataset Integrity Check")
    print("=" * 60)
    
    dataset_path = Path(dataset_dir)
    dataset_jsons_dir = dataset_path / "dataset_jsons"
    
    if not dataset_jsons_dir.exists():
        print(f"✗ dataset_jsons directory not found: {dataset_jsons_dir}")
        return False, {}
    
    # JSON to mask directory mapping
    json_to_mask_mapping = {
        'train_egoexo_pairs.json': 'Masks_TRAIN_EGO2EXO',
        'train_exoego_pairs.json': 'Masks_TRAIN_EXO2EGO',
        'val_egoexo_pairs.json': 'Masks_VAL_EGO2EXO',
        'val_exoego_pairs.json': 'Masks_VAL_EXO2EGO',
        'test_egoexo_pairs.json': 'Masks_TEST_EGO2EXO',
        'test_exoego_pairs.json': 'Masks_TEST_EXO2EGO'
    }
    
    all_results = {}
    overall_success = True
    
    for json_file, mask_dir_name in json_to_mask_mapping.items():
        print(f"\n{'='*50}")
        print(f"Processing: {json_file}")
        print(f"{'='*50}")
        
        json_path = dataset_jsons_dir / json_file
        mask_dir = dataset_path / mask_dir_name
        
        if not json_path.exists():
            print(f"⚠ Skipping missing JSON file: {json_file}")
            continue
            
        if not mask_dir.exists():
            print(f"✗ Mask directory not found: {mask_dir}")
            overall_success = False
            continue
        
        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                pairs_data = json.load(f)
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
            overall_success = False
            continue
        
        print(f"Total pairs in {json_file}: {len(pairs_data)}")
        
        # Process each pair
        # missing_files = defaultdict(set)  # Use set to avoid duplicates
        structured_missing = []  # For JSON output
        
        for i, pair in enumerate(tqdm(pairs_data)):
            
            # Parse the JSON entry
            parsed = parse_json_paths(pair)
            if not parsed:
                print(f"⚠ Failed to parse entry {i}: {pair}")
                continue
            
            # Check files for both ego and exo views
            # For ego->exo direction, we check exo destination
            # For exo->ego direction, we check ego destination
            ego_result = check_frame_files(
                parsed['take_id'], parsed['ego_camera'], parsed['frame'], 
                mask_dir)
            exo_result = check_frame_files(
                parsed['take_id'], parsed['exo_camera'], parsed['frame'], 
                mask_dir)
            
            if not exo_result['success']:
                structured_missing.append(exo_result)

            if not ego_result['success']:
                structured_missing.append(ego_result)
            
            if debug_mode and i == 100000:
                break
 
        all_results[json_file] = structured_missing
        
        if structured_missing:
            overall_success = False
    
    return overall_success, all_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-level O-MaMa dataset integrity verification")
    parser.add_argument("--dataset_dir", type=str, default="Ego-Exo4d",
                       help="Path to the dataset directory")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with breakpoints")
    parser.add_argument("--json-output", type=str, default="missing_mask_files.json",
                       help="JSON output file for missing files (default: missing_mask_files.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        exit(1)
    
    # Write header with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Run verification
    success, results = verify_frame_integrity(args.dataset_dir, args.debug)
    
    # Generate report
    if not success:
        total_missing = 0
        for json_file, data in results.items():
            total_missing += len(data)
        print(f"Total missing files: {total_missing}")

        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {args.json_output}")
    else:
        print("No missing files")