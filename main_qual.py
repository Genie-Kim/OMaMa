""" Defines the main script for evaluating O-MaMa """

import torch
import argparse

import torch
import numpy as np
import sys

from descriptors.get_descriptors import DescriptorExtractor
from dataset.dataset_masks import Masks_Dataset
from model.model import Attention_projector
from pathlib import Path

import helpers
from tqdm import tqdm
import os

def compute_IoU(pred_mask, gt_mask):
    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()
    IoU = intersection / (union + 1e-6)
    return IoU

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    else:
        return obj


def save_qualitative_visualization(batch, pred_mask, data_id, frame_idx, qualitative_dir):
    """Save 3-image horizontal layout with blended masks as requested"""
    import cv2
    
    # Load source and destination images
    source_img = batch['SOURCE_img'][0].cpu().numpy().transpose(1, 2, 0)
    dest_img = batch['GT_img'][0].cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    source_img = (source_img * std + mean).clip(0, 1)
    dest_img = (dest_img * std + mean).clip(0, 1)
    
    # Get masks
    source_gt_mask = batch['SOURCE_mask'][0].squeeze().cpu().numpy()
    dest_gt_mask = batch['GT_mask'][0].squeeze().cpu().numpy()
    
    # Find max height
    source_h, source_w = source_img.shape[:2]
    dest_h, dest_w = dest_img.shape[:2]
    max_h = max(source_h, dest_h)
    
    # Resize all images and masks to max height using interpolation
    if source_h != max_h:
        new_w = int(source_w * max_h / source_h)
        source_img = cv2.resize(source_img, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        source_gt_mask = cv2.resize(source_gt_mask.astype(np.float32), (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        source_gt_mask = (source_gt_mask > 0.5).astype(np.float32)
    
    if dest_h != max_h:
        new_w = int(dest_w * max_h / dest_h)
        dest_img = cv2.resize(dest_img, (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        dest_gt_mask = cv2.resize(dest_gt_mask.astype(np.float32), (new_w, max_h), interpolation=cv2.INTER_LINEAR)
        dest_gt_mask = (dest_gt_mask > 0.5).astype(np.float32)
    
    # Create overlay mask images
    # 1. Source image with GT mask (red, 0.7 alpha)
    source_overlay = source_img.copy()
    red_mask = np.zeros_like(source_img)
    red_mask[:, :, 0] = 1.0  # Red channel
    source_overlay = np.where(source_gt_mask[..., np.newaxis], 
                                source_overlay * 0.3 + red_mask * 0.7, 
                                source_overlay)
    
    # 2. Target image with predicted mask(s)
    target_pred_overlay = dest_img.copy()
    
    # Check if pred_mask is a list (top5 masks) or single mask
    if isinstance(pred_mask, list):
        # Top 5 masks visualization with different colors
        colors = [
            (0.0, 1.0, 0.0),     # Green (rank 2)
            (1.0, 1.0, 0.0),     # Yellow (rank 3)
            (1.0, 0.5, 0.0),     # Orange (rank 4)
            (1.0, 0.0, 1.0),      # Magenta (rank 5)
            (0.53, 0.81, 0.92),  # Sky blue (rank 1)

        ]
        
        # Process each mask with decreasing alpha for lower ranks
        for i, mask_tensor in enumerate(pred_mask):
            if i >= 3:  # Only show top 3
                break
            
            # Convert tensor to numpy
            mask = mask_tensor.detach().cpu().numpy()
            
            # Resize mask if needed
            if dest_h != max_h:
                mask = cv2.resize(mask.astype(np.float32), (new_w, max_h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.float32)
            
            # Create color mask
            color_mask = np.zeros_like(dest_img)
            color_mask[:, :, 0] = colors[i][0]  # R
            color_mask[:, :, 1] = colors[i][1]  # G
            color_mask[:, :, 2] = colors[i][2]  # B
            
            # Apply with decreasing alpha for lower ranks
            alpha = 0.7 - (i * 0.1)  # 0.7, 0.6, 0.5, 0.4, 0.3
            target_pred_overlay = np.where(mask[..., np.newaxis], 
                                          target_pred_overlay * (1 - alpha) + color_mask * alpha, 
                                          target_pred_overlay)
    else:
        # Single mask visualization (original behavior)
        if dest_h != max_h:
            pred_mask = cv2.resize(pred_mask.astype(np.float32), (new_w, max_h), interpolation=cv2.INTER_LINEAR)
            pred_mask = (pred_mask > 0.5).astype(np.float32)
        
        sky_blue_mask = np.zeros_like(dest_img)
        sky_blue_mask[:, :, 0] = 0.53  # Red channel (135/255)
        sky_blue_mask[:, :, 1] = 0.81  # Green channel (206/255)
        sky_blue_mask[:, :, 2] = 0.92  # Blue channel (235/255)
        target_pred_overlay = np.where(pred_mask[..., np.newaxis], 
                                        target_pred_overlay * 0.3 + sky_blue_mask * 0.7, 
                                        target_pred_overlay)
    
    # 3. Target image with GT mask (red, 0.7 alpha)
    target_gt_overlay = dest_img.copy()
    red_mask = np.zeros_like(dest_img)
    red_mask[:, :, 0] = 1.0  # Red channel
    target_gt_overlay = np.where(dest_gt_mask[..., np.newaxis], 
                                    target_gt_overlay * 0.3 + red_mask * 0.7, 
                                    target_gt_overlay)
    
    # Combine images horizontally using numpy
    combined_image = np.hstack([source_overlay, target_pred_overlay, target_gt_overlay])
    
    # Convert to BGR for cv2 (from RGB) and scale to 0-255
    combined_image_bgr = cv2.cvtColor((combined_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Save using cv2
    save_file = qualitative_dir / f"{data_id}_frame_{frame_idx}.png"
    cv2.imwrite(str(save_file), combined_image_bgr)



"""
python main_qual.py --reverse --root Ego-Exo4d --devices 0 --checkpoint_dir pretrained_models/Exo2Ego.pt --exp_name Eval_OMAMA_0fe5b647 --data_id "1f541cce-f10b-4cd3-814f-8e2d527df8f5"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match masks from ego-exo pairs")
    parser.add_argument("--root", type=str, default="/media/maria/Datasets/Ego-Exo4d",help="Path to the dataset")
    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size of the dino transformer")
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context sizo for the object")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    parser.add_argument("--checkpoint_dir", type=str, default="pretrained_models/Exo2Ego.pt")
    parser.add_argument("--data_id", type=str, help="Specific data ID for qualitative analysis (optional)")
    parser.add_argument("--top5qual", type=bool,default=True, help="plot top 5 qualitative results")
    parser.add_argument("--qual_dir", type=str,default=None, help="path to qualitative")

    args = parser.parse_args()

    helpers.set_all_seeds(42)
    if args.devices != "cpu":
        gpus = [args.devices]  # Specify which GPUs to use
        device_ids = [f'cuda:{gpu}' for gpu in gpus]

        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    

    # Create qualitative folder
    if args.qual_dir is not None:
        # Extract the last folder name from checkpoint_dir
        last_folder_name = Path(args.checkpoint_dir).parent.name
        qualitative_dir = Path(args.qual_dir) / last_folder_name / 'qualitative_results'
    else:
        qualitative_dir = Path(os.path.join(os.path.dirname(args.checkpoint_dir), 'qualitative_results'))
    
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    descriptor_extractor = DescriptorExtractor('dinov2_vitb14_reg',  args.patch_size, args.context_size, device)
    model = Attention_projector(args.reverse).to(device)
    checkpoint_weights = torch.load(args.checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint_weights, strict=False)
    print(model)
    
    # Process all samples for qualitative visualization
    test_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, 
                                N_masks_per_batch=args.N_masks_per_batch, 
                                order=args.order, test=True)
    
    # Only load dataset if specific data_id provided or for batch processing
    if args.data_id:
        print(f"Looking for specific data_id: {args.data_id}")
        # Print some info to debug
        print(f"Dataset size: {len(test_dataset.pairs)}")
        
        # Collect ALL indices for this data_id
        found_indices = []
        frame_info = []  # Store frame numbers for each found index
        print(f"Searching through {len(test_dataset.pairs)} pairs for all frames of {args.data_id}...")
        
        for idx, pair in enumerate(test_dataset.pairs):
            # Extract take_id from the pair path structure
            if isinstance(pair, (list, tuple)) and len(pair) >= 1:
                path = str(pair[0])  # Get first path in the pair
                # Path pattern: /path/to/processed//UUID//camera//object//type//frame
                parts = path.split('//')
                if len(parts) >= 2:  # At least processed//UUID
                    take_id = parts[1]  # UUID is at index 1 after splitting by '//'
                    if take_id == args.data_id:
                        # Extract frame number from the end of the path
                        frame_idx = path.rstrip('/').split('/')[-1]
                        found_indices.append(idx)
                        frame_info.append(frame_idx)
        
        if not found_indices:
            print(f"Data ID {args.data_id} not found!")
            print("Sampling available take IDs from dataset:")
            sample_count = min(10, len(test_dataset.pairs))
            available_take_ids = set()
            
            for i in range(sample_count):
                pair = test_dataset.pairs[i]
                if isinstance(pair, (list, tuple)) and len(pair) >= 1:
                    path = str(pair[0])
                    parts = path.split('//')
                    if len(parts) >= 2:
                        take_id = parts[1]  # Extract UUID
                        available_take_ids.add(take_id)
            
            for take_id in sorted(available_take_ids):
                print(f"  {take_id}")
            
            print(f"\nTotal unique take IDs found in first {sample_count} samples: {len(available_take_ids)}")
            sys.exit(1)
        
        print(f"Found {len(found_indices)} frames for data_id {args.data_id}")
        print(f"Frame indices: {sorted(set(frame_info))[:10]}{'...' if len(set(frame_info)) > 10 else ''}")
        
        # Create dataloader with ALL found indices
        test_dataset_subset = torch.utils.data.Subset(test_dataset, found_indices)
        test_dataloader = torch.utils.data.DataLoader(test_dataset_subset, batch_size=1, shuffle=False)

        print(f"Processing data ID: {args.data_id}")
        print(f"Processing {len(found_indices)} frames...")
        
        # Process all frames for this data_id
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Processing frames")):
            # Extract frame index from the batch
            pair_idx = batch['pair_idx'][0].item()
            frame_idx = frame_info[batch_idx]  # Get the corresponding frame number
            
            # Extract and process the sample
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            
            similarities, pred_masks_idx, pred_mask, loss, top5_masks = model(
                SOURCE_descriptors, DEST_descriptors,
                SOURCE_img_feats, DEST_img_feats,
                batch['POS_mask_position'], batch['is_visible'],
                batch['DEST_SAM_masks'], test_mode=True)
            
            pred_mask = pred_mask.squeeze().detach().cpu().numpy()         

            if args.top5qual:
                top5mask = []
                for topk in top5_masks[0]:  # Use index 0 since batch_size=1
                    top5mask.append(batch['DEST_SAM_masks'][0, topk])
                # Save visualization for this frame
                save_qualitative_visualization(batch, top5mask, args.data_id, frame_idx, qualitative_dir)
            else:
                # Save visualization for this frame
                save_qualitative_visualization(batch, pred_mask, args.data_id, frame_idx, qualitative_dir)
            
        print(f"Saved {len(found_indices)} qualitative visualizations to: {qualitative_dir}")
    else:

        print(f"Processing all test samples for qualitative visualization...")
        print(f"Total samples to process: {len(test_dataset)}")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True, prefetch_factor=8)
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Processing samples")):
            # Extract path information to get take_id and frame_idx
            img_path = batch['img_pth1'][0]  # Since batch_size=1
            parts = img_path.split('//')
            if len(parts) >= 2:
                take_id = parts[1]  # Extract UUID
                frame_idx = img_path.rstrip('/').split('/')[-1]  # Extract frame number
            else:
                # Fallback if path structure is different
                take_id = f"sample_{batch_idx}"
                frame_idx = str(batch_idx)
            
            # Extract descriptors
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            
            # Get model predictions
            similarities, pred_masks_idx, pred_mask, loss, top5_masks = model(
                SOURCE_descriptors, DEST_descriptors,
                SOURCE_img_feats, DEST_img_feats,
                batch['POS_mask_position'], batch['is_visible'],
                batch['DEST_SAM_masks'], test_mode=True)
            
            # Extract predicted mask (batch_size=1, so use index 0)
            mask_idx = pred_masks_idx[0].item()
            pred_mask = batch['DEST_SAM_masks'][0, mask_idx, :, :].detach().cpu().numpy()
            
            # Extract top5 masks if requested
            if args.top5qual:
                top5mask = []
                for topk in top5_masks[0]:  # Use index 0 since batch_size=1
                    top5mask.append(batch['DEST_SAM_masks'][0, topk])
                # Save visualization with top5 masks
                save_qualitative_visualization(batch, top5mask, take_id, frame_idx, qualitative_dir)
            else:
                # Save visualization with single predicted mask
                save_qualitative_visualization(batch, pred_mask, take_id, frame_idx, qualitative_dir)
        
        print(f"Saved {len(test_dataset)} qualitative visualizations to: {qualitative_dir}")
    
    