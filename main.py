""" Defines the main script for training O-MaMa """

import torch
import argparse
from descriptors.get_descriptors import DescriptorExtractor
from dataset.dataset_masks import Masks_Dataset
from model.model import Attention_projector
from evaluation.evaluate import add_to_json, evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import helpers
from tqdm import tqdm
import os
import wandb
import numpy as np
import cv2
import random
from datetime import datetime
import itertools


def run_validation(model, descriptor_extractor, val_dataloader, val_dataset, 
                   args, qualitative_dir, exp_dir, exp_name, current_epoch, 
                   global_step, best_IoU, DEBUG_MODE):
    """Run validation and save checkpoints"""
    print(f'----- Step {global_step} (Epoch {current_epoch:.2f}) Validation -----')
    # Validation loop
    processed_epoch, pred_json_epoch, gt_json_epoch = {}, {}, {}
    epoch_loss = 0
    model.eval()
    
    # Track qualitative visualization state
    selected_data_ids = set()
    data_id_frame_counts = {}
    
    for idx, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
            SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
            similarities, pred_masks_idx, refined_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                                 SOURCE_img_feats, DEST_img_feats, 
                                                                                 batch['POS_mask_position'], batch['is_visible'],
                                                                                 batch['DEST_SAM_masks'], test_mode = False)
            pred_mask = refined_mask.squeeze().detach().cpu().numpy()
            confidence = similarities.detach().cpu().numpy()
            
            epoch_loss += loss.item()
            pred_json_epoch, gt_json_epoch = add_to_json(val_dataset, batch['pair_idx'], 
                                                        pred_mask, confidence,
                                                        processed_epoch, pred_json_epoch, gt_json_epoch)
            
            # Generate qualitative outputs for selected data_ids
            pair_idx = batch['pair_idx'][0].item()
            if args.reverse:
                img_pth1, _, img_pth2, _ = val_dataset.pairs[pair_idx]
            else:
                img_pth2, _, img_pth1, _ = val_dataset.pairs[pair_idx]
            
            # Extract take_id from path
            _, take_id, _, _, _, frame_idx = img_pth1.split('//')
            frame_idx = int(frame_idx)
            
            # Only visualize for first 3 randomly selected data_ids, max 5 frames each
            if len(selected_data_ids) < 3 and take_id not in selected_data_ids:
                if random.random() < 0.1:  # 10% chance to select new data_id
                    selected_data_ids.add(take_id)
                    data_id_frame_counts[take_id] = 0
            
            if take_id in selected_data_ids and data_id_frame_counts[take_id] < 5:
                try:
                    saved_path = create_qualitative_visualization(
                        batch, pred_mask, qualitative_dir, take_id, frame_idx
                    )
                    data_id_frame_counts[take_id] += 1
                    
                    # Log image to WandB
                    wandb.log({
                        f"qualitative/{take_id}/frame_{frame_idx}": wandb.Image(saved_path),
                        "epoch": current_epoch
                    })
                except Exception as e:
                    print(f"Warning: Failed to create qualitative visualization: {e}")
            
            # Debug mode: limit validation iterations
            if DEBUG_MODE and idx >= 2:
                break

    epoch_loss /= len(val_dataloader)
    print(f'----- Step {global_step} (Epoch {current_epoch:.2f}) metrics -----')
    out_dict = evaluate(gt_json_epoch, pred_json_epoch, args.reverse)  
    
    # Log validation metrics to WandB
    wandb.log({f"val/{k}": v for k, v in out_dict.items()})
    wandb.log({"val/loss": epoch_loss, "val/epoch": current_epoch})
    
    if out_dict['iou'] > best_IoU:
        best_IoU = out_dict['iou']
        torch.save(model.state_dict(), exp_dir / f'best_IoU_step_{global_step}_{exp_name}.pt')
    
    return best_IoU


def create_qualitative_visualization(batch, pred_mask, qualitative_dir, data_id_prefix, frame_idx):
    """Create side-by-side visualization adapted from main_qual.py"""
    # Extract and denormalize images
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
    
    # Resize to 480px height while maintaining aspect ratio
    target_height = 480
    source_h, source_w = source_img.shape[:2]
    dest_h, dest_w = dest_img.shape[:2]
    
    # Resize source image and mask
    if source_h != target_height:
        new_w = int(source_w * target_height / source_h)
        source_img = cv2.resize(source_img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        source_gt_mask = cv2.resize(source_gt_mask.astype(np.float32), (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        source_gt_mask = (source_gt_mask > 0.5).astype(np.float32)
    
    # Resize dest image and masks
    if dest_h != target_height:
        new_w = int(dest_w * target_height / dest_h)
        dest_img = cv2.resize(dest_img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        dest_gt_mask = cv2.resize(dest_gt_mask.astype(np.float32), (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        dest_gt_mask = (dest_gt_mask > 0.5).astype(np.float32)
        pred_mask_resized = cv2.resize(pred_mask.astype(np.float32), (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        pred_mask_resized = (pred_mask_resized > 0.5).astype(np.float32)
    else:
        pred_mask_resized = pred_mask
    
    # Create overlay masks
    # 1. Source image with GT mask (red, α=0.7)
    source_overlay = source_img.copy()
    red_mask = np.zeros_like(source_img)
    red_mask[:, :, 0] = 1.0
    source_overlay = np.where(source_gt_mask[..., np.newaxis], 
                              source_overlay * 0.3 + red_mask * 0.7, 
                              source_overlay)
    
    # 2. Target image with GT mask (red, α=0.5) and pred mask (blue, α=0.7)
    target_overlay = dest_img.copy()
    
    # Add GT mask (red, α=0.5)
    red_mask = np.zeros_like(dest_img)
    red_mask[:, :, 0] = 1.0
    target_overlay = np.where(dest_gt_mask[..., np.newaxis], 
                              target_overlay * 0.5 + red_mask * 0.5, 
                              target_overlay)
    
    # Add pred mask (blue, α=0.7)
    blue_mask = np.zeros_like(dest_img)
    blue_mask[:, :, 2] = 1.0  # Blue channel
    target_overlay = np.where(pred_mask_resized[..., np.newaxis], 
                              target_overlay * 0.3 + blue_mask * 0.7, 
                              target_overlay)
    
    # Combine images horizontally
    combined_image = np.hstack([source_overlay, target_overlay])
    
    # Convert to BGR for cv2 and scale to 0-255
    combined_image_bgr = cv2.cvtColor((combined_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Create data_id directory
    data_id_dir = qualitative_dir / data_id_prefix
    data_id_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    save_file = data_id_dir / f"frame_{frame_idx}.jpg"
    cv2.imwrite(str(save_file), combined_image_bgr)
    
    return str(save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match masks from ego-exo pairs")
    parser.add_argument("--root", type=str, default="Ego-Exo4d",help="Path to the dataset")
    parser.add_argument("--reverse", action="store_true", help="Flag to select exo->ego pairs")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size of the dino transformer")
    parser.add_argument("--context_size", type=int, default=20, help="Size of the context sizo for the object")
    parser.add_argument("--devices", default="0", type=str)
    parser.add_argument("--N_masks_per_batch", default=32, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--max_iterations", default=100000, type=int, help="Maximum training iterations")
    parser.add_argument("--order", default=2, type=int, help="order of adjacency matrix, 2 for 2nd order")
    parser.add_argument("--exp_name", type=str, default="Train_OMAMA")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for development")
    args = parser.parse_args()
    # making exp name with date and view change
    if args.reverse:
        view_change = "Exo2Ego"
    else:
        view_change = "Ego2Exo"
    date = datetime.now().strftime("%y%m%d%H%M")
    exp_name = f"{view_change}_{date}_{args.exp_name}"

    # Initialize WandB
    wandb.init(project="O-MaMa", name=exp_name, config=args,
               settings=wandb.Settings(disable_git=True, save_code=False))

    # Debug mode for development
    DEBUG_MODE = args.debug

    helpers.set_all_seeds(42)
    if args.devices != "cpu":
        gpus = [args.devices]  # Specify which GPUs to use
        device_ids = [f'cuda:{gpu}' for gpu in gpus]

        device = torch.device(f'cuda:{device_ids[0].split(":")[1]}') if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    # Training dataset only contains horizontal images, in order to batchify the masks
    train_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, N_masks_per_batch=args.N_masks_per_batch, order = args.order, train = True, test = False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=helpers.our_collate_fn, num_workers = 8, pin_memory = True) #16 in both
    
    # Validation dataset contains both horizontal and vertical images. Since the SAM masks are different, we use batch size 1
    # Note: the val annotations are a small subset of the full validation dataset, used for eval the training per epoch
    val_dataset = Masks_Dataset(args.root, args.patch_size, args.reverse, args.N_masks_per_batch,  order = args.order, train = False, test = False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    descriptor_extractor = DescriptorExtractor('dinov2_vitb14_reg', args.patch_size, args.context_size, device)
    model = Attention_projector(reverse = args.reverse).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    T_max = args.max_iterations
    scheduler = CosineAnnealingLR(optimizer, args.max_iterations, eta_min=1e-6)

    # Create experiment folders
    exp_dir = Path("exp_results", exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = exp_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training variables
    global_step = 0
    eval_step = 2000
    best_IoU = 0
    train_iter = itertools.cycle(train_dataloader)
    trainset_length = len(train_dataloader)
    
    print(f'Starting training for {args.max_iterations} iterations, validating every {eval_step} steps')
    
    # Training loop
    while global_step < args.max_iterations:
        model.train()
        batch = next(train_iter)
        
        DEST_descriptors, DEST_img_feats = descriptor_extractor.get_DEST_descriptors(batch)
        SOURCE_descriptors, SOURCE_img_feats = descriptor_extractor.get_SOURCE_descriptors(batch)
        best_similarities, best_masks, refined_mask, loss, top5_masks = model(SOURCE_descriptors, DEST_descriptors, 
                                                                              SOURCE_img_feats, DEST_img_feats, 
                                                                              batch['POS_mask_position'], batch['is_visible'],
                                                                              batch['DEST_SAM_masks'], test_mode = False)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        global_step += 1
        current_epoch = global_step / trainset_length
        
        # Log training metrics to WandB
        wandb.log({
            "train/loss": loss.item(), 
            "train/epoch": current_epoch, 
            "train/step": global_step
        })
        
        # Print progress every 100 steps
        if global_step % 100 == 0 or global_step == 1:
            print(f'Step {global_step}/{args.max_iterations} (Epoch {current_epoch:.2f}), Loss: {loss.item():.4f}')
        
        # Validation every eval_step
        if global_step % eval_step == 0 or global_step == 1:
            # Save checkpoint before validation
            torch.save(model.state_dict(), exp_dir / f'step_{global_step}_epoch_{current_epoch:.2f}_{exp_name}.pt')
            
            # Run validation
            best_IoU = run_validation(model, descriptor_extractor, val_dataloader, val_dataset,
                                     args, qualitative_dir, exp_dir, exp_name, current_epoch,
                                     global_step, best_IoU, DEBUG_MODE)
        
        # Debug mode: limit training iterations
        if DEBUG_MODE and global_step >= 2:
            break
    
    print(f'Training completed at step {global_step} (Epoch {current_epoch:.2f})')
    # Final checkpoint
    torch.save(model.state_dict(), exp_dir / f'final_step_{global_step}_epoch_{current_epoch:.2f}_{exp_name}.pt')
