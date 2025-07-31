""" Dataloader for the Ego-Exo4D correspondences dataset """

# Standard library imports
import json
import os
import random

# Third-party imports
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

# Local imports
from dataset.adj_descriptors import get_adj_matrix
from dataset.dataset_utils import compute_IoU, compute_IoU_bbox, bbox_from_mask
from utils.mask_converter import MaskConverter


class Masks_Dataset(Dataset):
    def __init__(self, root, patch_size, reverse, N_masks_per_batch, order, train=False, test=False, val=False):
        if not train and not test and not val:
            raise ValueError("At least one of train, test, or val must be True")
        
        self.root = root
        self.train_mode = train
        self.test_mode = test
        self.val_mode = val
        self.reverse = reverse

        # Select the pre-extracted masks directory based on the train/test mode and reverse flag
        if train:
            if reverse:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EXO2EGO')
            else:
                self.masks_dir = os.path.join(root, 'Masks_TRAIN_EGO2EXO')
        else:
            if test:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EXO2EGO')
                else:
                    self.masks_dir = os.path.join(root, 'Masks_TEST_EGO2EXO')
                    
            else:
                if reverse:
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EXO2EGO')
                else:   
                    self.masks_dir = os.path.join(root, 'Masks_VAL_EGO2EXO')

        # Preprocessed dataset directory
        self.dataset_dir = os.path.join(root, 'processed')

        # Configs for loading the features
        self.N_masks_per_batch = N_masks_per_batch
        self.patch_size = patch_size

        self.order = order

        # Load the mask annotations and pairs
        self.mask_annotations_paths = self.load_mask_annotations_path()
        self.pairs = self.load_all_pairs()
        self.takes_json = json.load(open(os.path.join(root, 'takes.json'), 'r'))

        # Transformations for the images
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
        
        # Initialize mask converter for indexed mask optimization
        self.mask_converter = MaskConverter(max_masks=256)
        
        # Memory usage statistics tracking
        self.memory_stats = {
            'total_masks_loaded': 0,
            'total_memory_saved_bytes': 0,
            'conversion_failures': 0,
            'truncated_mask_samples': 0
        }
        
        print(len(self.takes_json), 'TAKES')

    # Load the json with the pairs
    def load_all_pairs(self):
        # Select the json file based on the train/test mode and reverse flag
        if self.train_mode:
            if self.reverse:
                # pairs_json = 'train_exoego_20percent_pairs.json' # We train with 20% of the pairs of the full trainig set
                pairs_json = 'train_exoego_pairs.json' # We train with 20% of the pairs of the full trainig set
            else:
                # pairs_json = 'train_egoexo_20percent_pairs.json' # We train with 20% of the pairs of the full trainig set
                pairs_json = 'train_egoexo_pairs.json' # We train with 20% of the pairs of the full trainig set
        else:
            if self.test_mode:
                if self.reverse:
                    pairs_json = 'test_exoego_pairs.json'
                else:
                    pairs_json = 'test_egoexo_pairs.json'
            
            #Validation is just a subset of the test
            else:    
                if self.reverse:
                    pairs_json = 'custom_val_exoego_pairs_10.json' # We validate with 10% of the pairs of the full validation set
                    # pairs_json = 'val_exoego_pairs.json' # We validate with 10% of the pairs of the full validation set
                else:
                    pairs_json = 'custom_val_egoexo_pairs_10.json' # We validate with 10% of the pairs of the full validation set
                    # pairs_json = 'val_egoexo_pairs.json' # We validate with 10% of the pairs of the full validation set

        print('----------------------------We are loading: ', pairs_json, 'with the pair of images')
        pairs = []
        jsons_dir = os.path.join(self.root,'dataset_jsons') # Put the jsons generated from the data preparation here
        with open(os.path.join(jsons_dir, pairs_json), 'r') as fp:
            pairs.extend(json.load(fp))
        print('LEN OF THE DATASET:', len(pairs))
        return pairs
    
    # Load the GT mask annotations
    def load_mask_annotations_path(self):
        d = self.dataset_dir
        with open(f'{d}/split.json', 'r') as fp:
            splits = json.load(fp)
        
        # Only load annotations for the current mode to save memory
        if self.train_mode:
            valid_takes = splits['train']
        elif self.val_mode:
            valid_takes = splits['val']
        else:  # test_mode
            valid_takes = splits['test']

        annotations = {}
        for take in valid_takes:
            try:
                with open(f'{d}/{take}/annotation.json', 'r') as fp:
                    # annotations[take] = json.load(fp)
                    annotations[take] = f'{d}/{take}/annotation.json'
            except:
                continue
        return annotations
    
    def load_mask_annotations(self, take_id):
        with open(self.mask_annotations_paths[take_id], 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _load_sam_masks(self, mask_path: str, expected_size: tuple = None) -> tuple:
        """
        Load SAM masks from NPZ file and convert to indexed format for memory optimization.
        
        Args:
            mask_path: Path to the NPZ mask file
            expected_size: Expected (H, W) size for interpolation if needed
            
        Returns:
            Tuple of (indexed_masks, original_num_masks, conversion_stats)
            - indexed_masks: HxW tensor with uint8 indexed format (CPU memory)
            - original_num_masks: Number of masks in original binary format
            - conversion_stats: Dictionary with memory usage statistics
        """
        try:
            # Load binary masks from NPZ file
            sam_masks_data = np.load(mask_path)
            binary_masks = torch.from_numpy(sam_masks_data['arr_0'].astype(np.uint8))  # N, H, W
            
            # Handle edge case where masks might be 2D (single mask)
            if len(binary_masks.shape) < 3:
                # Create empty mask set if no masks available
                H, W = expected_size if expected_size else (512, 512)  # Default size fallback
                binary_masks = torch.zeros((1, H, W), dtype=torch.uint8)
            
            original_num_masks, H, W = binary_masks.shape
            
            # Apply interpolation if size mismatch (maintaining original logic)
            if expected_size and (H != expected_size[0] or W != expected_size[1]):
                binary_masks = F.interpolate(
                    binary_masks.unsqueeze(0).float(), 
                    size=expected_size, 
                    mode='nearest'
                ).squeeze(0).to(torch.uint8)
                H, W = expected_size
            
            # Convert binary masks to indexed format for memory optimization
            indexed_masks = self.mask_converter.binary_to_indexed(binary_masks)
            
            # Calculate memory usage statistics
            binary_memory = binary_masks.numel() * binary_masks.element_size()
            indexed_memory = indexed_masks.numel() * indexed_masks.element_size()
            memory_saved = binary_memory - indexed_memory
            
            conversion_stats = {
                'original_num_masks': original_num_masks,
                'original_height': H,
                'original_width': W,
                'binary_memory_bytes': binary_memory,
                'indexed_memory_bytes': indexed_memory,
                'memory_saved_bytes': memory_saved,
                'reduction_factor': binary_memory / indexed_memory if indexed_memory > 0 else 0,
                'truncated': original_num_masks > 256
            }
            
            # Update dataset-level statistics
            self.memory_stats['total_masks_loaded'] += 1
            self.memory_stats['total_memory_saved_bytes'] += memory_saved
            if original_num_masks > 256:
                self.memory_stats['truncated_mask_samples'] += 1
            
            # Return indexed masks (kept in CPU memory for optimization), num_masks, and stats
            return indexed_masks.cpu(), original_num_masks, conversion_stats
            
        except Exception as e:
            # Handle conversion failures gracefully
            self.memory_stats['conversion_failures'] += 1
            
            # Log the error with context
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to load and convert masks from {mask_path}: {e}. "
                f"Falling back to original binary format."
            )
            
            # Fallback: return original binary masks without conversion
            try:
                sam_masks_data = np.load(mask_path)
                binary_masks = torch.from_numpy(sam_masks_data['arr_0'].astype(np.uint8))
                
                if len(binary_masks.shape) < 3:
                    H, W = expected_size if expected_size else (512, 512)
                    binary_masks = torch.zeros((1, H, W), dtype=torch.uint8)
                
                if expected_size:
                    original_num_masks, H, W = binary_masks.shape
                    if H != expected_size[0] or W != expected_size[1]:
                        binary_masks = F.interpolate(
                            binary_masks.unsqueeze(0).float(), 
                            size=expected_size, 
                            mode='nearest'
                        ).squeeze(0).to(torch.uint8)
                
                # Return binary masks as fallback (no memory optimization)
                fallback_stats = {
                    'original_num_masks': binary_masks.shape[0],
                    'binary_memory_bytes': binary_masks.numel() * binary_masks.element_size(),
                    'indexed_memory_bytes': 0,  # No optimization applied
                    'memory_saved_bytes': 0,
                    'reduction_factor': 1.0,
                    'truncated': False,
                    'fallback_used': True
                }
                
                return binary_masks.cpu(), binary_masks.shape[0], fallback_stats
                
            except Exception as fallback_error:
                # Complete failure - return minimal empty mask
                logger.error(f"Complete mask loading failure: {fallback_error}")
                H, W = expected_size if expected_size else (512, 512)
                empty_mask = torch.zeros((H, W), dtype=torch.uint8)
                error_stats = {
                    'original_num_masks': 0,
                    'binary_memory_bytes': 0,
                    'indexed_memory_bytes': empty_mask.numel() * empty_mask.element_size(),
                    'memory_saved_bytes': 0,
                    'reduction_factor': 0,
                    'truncated': False,
                    'error': True
                }
                return empty_mask.cpu(), 0, error_stats

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory usage statistics for the dataset.
        
        Returns:
            Dictionary with memory optimization metrics
        """
        stats = self.memory_stats.copy()
        
        # Calculate derived metrics
        if stats['total_masks_loaded'] > 0:
            stats['average_memory_saved_per_sample'] = (
                stats['total_memory_saved_bytes'] / stats['total_masks_loaded']
            )
            stats['truncation_rate'] = (
                stats['truncated_mask_samples'] / stats['total_masks_loaded']
            )
            stats['failure_rate'] = (
                stats['conversion_failures'] / stats['total_masks_loaded']
            )
        else:
            stats['average_memory_saved_per_sample'] = 0
            stats['truncation_rate'] = 0
            stats['failure_rate'] = 0
        
        # Convert bytes to MB for readability
        stats['total_memory_saved_mb'] = stats['total_memory_saved_bytes'] / (1024 * 1024)
        stats['average_memory_saved_per_sample_mb'] = stats['average_memory_saved_per_sample'] / (1024 * 1024)
        
        return stats

    # Returns the img reshaped slightly to be divisible by 14 (dinov2 patch size)
    def reshape_img(self, img):
        h, w = img.shape[:2]
        new_h = self.patch_size * (h // self.patch_size)
        new_w = self.patch_size * (w // self.patch_size)
        size = (new_w, new_h)
        if h % self.patch_size != 0:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        return img

    def check_horizontal(self, img):
        h, w = img.shape[:2]
        if h > w:
            horiz_size = (h, w)
            img = cv2.resize(img, horiz_size, interpolation=cv2.INTER_NEAREST) # cv2.resize(img, (width, height)) expects (width, height) format
        return img

    # Select the adjacent negatives based on the adjacency matrix   
    def select_adjacent_negatives(self, adj_matrix, SAM_bboxes, SAM_masks, mask_GT):
        # Select adjacent negatives based on the adjacency matrix
        
        bbox_GT, _ = bbox_from_mask(mask_GT)
        bbox_iou = compute_IoU_bbox(SAM_bboxes, bbox_GT)
        max_index = torch.argmax(bbox_iou)
        
        # Get the neighbors of the best mask
        adj_matrix[max_index, max_index] = 0
        neighbors = torch.where(adj_matrix[max_index] == 1)[0]
        N_adjacent_indices = self.N_masks_per_batch - 1
        if len(neighbors) > N_adjacent_indices:
            random_indices = np.random.choice(neighbors, N_adjacent_indices, replace=False)
            adjacent_SAM_masks = SAM_masks[random_indices]
            adjacent_SAM_bboxes = SAM_bboxes[random_indices]
        else:
            adjacent_SAM_masks = SAM_masks[neighbors]
            adjacent_SAM_bboxes = SAM_bboxes[neighbors]
            
            # Get remaining negatives
            N_remaining_indices = N_adjacent_indices - len(neighbors)
            if SAM_masks.shape[0] < N_remaining_indices:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=True)
            else:
                remaining_indices = np.random.choice(SAM_masks.shape[0], N_remaining_indices, replace=False)
                
            
            adjacent_SAM_masks = torch.cat((adjacent_SAM_masks, SAM_masks[remaining_indices]), dim=0)
            adjacent_SAM_bboxes = torch.cat((adjacent_SAM_bboxes, SAM_bboxes[remaining_indices]), dim=0)
            
        
        return adjacent_SAM_masks, adjacent_SAM_bboxes

    # Select the best SAM mask
    def get_best_mask(self, SAM_masks, mask_GT):
        iou = compute_IoU(SAM_masks, mask_GT)
        return torch.argmax(iou)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx_sample):
        # Get the pair of images, 1 refers to the source image and 2 to the destination image
        if self.reverse:
            # img_pth2 ego, img_pth1 exo
            img_pth2, _, img_pth1, _ = self.pairs[idx_sample]
        else:
            # img_pth2 exo, img_pth1 ego
            img_pth1, _, img_pth2, _ = self.pairs[idx_sample]

        root, take_id, cam, obj, _, idx = img_pth1.split('//')
        root = self.dataset_dir
        root2, take_id2, cam2, obj2, _, idx2 = img_pth2.split('//')
        root2 = self.dataset_dir

        # Both viewpoints should have the same take_id, object and index   
        assert obj == obj2
        assert idx == idx2
        assert take_id == take_id2

        vid_idx = int(idx)
        vid_idx2 = int(idx2)

        # Image 1: SOURCE
        img1 = cv2.imread(f"{root}/{take_id}/{cam}/{vid_idx}.jpg")[..., ::-1]
        if self.train_mode or self.val_mode:
            img1 = self.check_horizontal(img1)
        img1 = self.reshape_img(img1)
        self.h1, self.w1 = img1.shape[:2]
        img1_torch = self.transform_img(img1)

        # Load the source mask
        mask_annotation_SOURCE = self.load_mask_annotations(take_id)
        mask_SOURCE = mask_utils.decode(mask_annotation_SOURCE['masks'][obj][cam][idx])
        mask_SOURCE = cv2.resize(mask_SOURCE, (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        assert mask_SOURCE.shape == img1.shape[:2]
        mask_SOURCE = torch.from_numpy(mask_SOURCE.astype(np.uint8))
        
        
        # Image 2: DESTINATION
        img2 = cv2.imread(f"{root2}/{take_id2}/{cam2}/{vid_idx2}.jpg")[..., ::-1]
        if self.train_mode or self.val_mode:
            img2 = self.check_horizontal(img2)
        img2 = self.reshape_img(img2)
        self.h2, self.w2 = img2.shape[:2]
        img2_torch = self.transform_img(img2)

                
        # Load the destination GT mask
        mask_annotation_DEST = self.load_mask_annotations(take_id2)
        if idx in mask_annotation_DEST['masks'][obj2][cam2]:  # If the object is visible in the destionation image, load it
            mask2_GT = mask_utils.decode(mask_annotation_DEST['masks'][obj2][cam2][idx])
            mask2_GT = cv2.resize(mask2_GT, (img2.shape[1],img2.shape[0]), interpolation=cv2.INTER_NEAREST)
        else: # If the object is not visible in the destination image, create an empty mask
            mask2_GT = np.zeros(img2.shape[:2])
        assert mask2_GT.shape == img2.shape[:2]
        mask2_GT = torch.from_numpy(mask2_GT.astype(np.uint8))

        # Load the proposed pre-extracted SAM masks for this pair using indexed format optimization
        mask_file_path = f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_masks.npz"
        SAM_masks_indexed, N_masks, conversion_stats = self._load_sam_masks(
            mask_file_path, 
            expected_size=(self.h2, self.w2)
        )
        
        # Store conversion statistics for debugging/monitoring (can be used by training pipeline)
        # SAM_masks_indexed is now HxW uint8 tensor stored in CPU memory for optimization
        # N_masks contains the original number of masks for compatibility with existing logic
        
        # Get the adjacent matrix (only needed for training)
        # For adjacency calculation, we need binary masks temporarily
        if self.train_mode:
            # Convert indexed masks back to binary format for adjacency matrix calculation
            # This is needed only during training and only temporarily
            SAM_masks_binary = self.mask_converter.indexed_to_binary(SAM_masks_indexed, N_masks)
            adj_matrix = get_adj_matrix(SAM_masks_binary, order=self.order)
            # Clean up temporary binary masks to save memory
            del SAM_masks_binary
        else:
            adj_matrix = None
        
        SAM_bboxes_dest = np.load(f"{self.masks_dir}/{take_id2}/{cam2}/{vid_idx2}_boxes.npy")
        SAM_bboxes_dest = torch.from_numpy(SAM_bboxes_dest.astype(np.float32)) # x1, y1, w, h
        
        # For bbox scaling, we need the original mask dimensions from conversion stats
        original_H = conversion_stats.get('original_height', self.h2)  
        original_W = conversion_stats.get('original_width', self.w2)
        h_factor = self.h2 / original_H if original_H > 0 else 1.0
        w_factor = self.w2 / original_W if original_W > 0 else 1.0
        
        if h_factor != 1 or w_factor != 1:
            SAM_bboxes_dest[:, 0] = SAM_bboxes_dest[:, 0] * w_factor
            SAM_bboxes_dest[:, 1] = SAM_bboxes_dest[:, 1] * h_factor
            SAM_bboxes_dest[:, 2] = SAM_bboxes_dest[:, 2] * w_factor
            SAM_bboxes_dest[:, 3] = SAM_bboxes_dest[:, 3] * h_factor       

        if self.train_mode:  # Training: Construct 1 positive + 31 negative batch for contrastive learning
            visible_pixels = mask2_GT.sum()  # Check if object is visible in destination image
            
            # Convert indexed masks to binary format for training operations that need individual masks
            SAM_masks_binary = self.mask_converter.indexed_to_binary(SAM_masks_indexed, N_masks)
            
            # If the object is visible in the destination image, we select the best SAM mask as the positive mask
            if visible_pixels > 0:  # Case 1: Object is visible - use GT as strong positive
                NEG_SAM_masks, NEG_SAM_bboxes = self.select_adjacent_negatives(adj_matrix, SAM_bboxes_dest, SAM_masks_binary, mask2_GT)  # Hard negative mining using adjacency
                is_visible = torch.tensor(1.) # True
                POS_SAM_masks = mask2_GT # Choose the GT (Strong Positive) or the best SAM mask (Weak Positive) TODO
                POS_SAM_bboxes, _ = bbox_from_mask(mask2_GT) # x1, y1, w, h
            else:  # Case 2: Object not visible - random selection for negative learning
                N_remaining_indices = self.N_masks_per_batch - 1  # Need 31 negatives
                if N_masks < N_remaining_indices:  # If not enough masks, use replacement
                    random_indices = np.random.choice(N_masks, N_remaining_indices, replace=True)
                else:  # Sufficient masks, no replacement needed
                    random_indices = np.random.choice(N_masks, N_remaining_indices, replace=False)
                
                NEG_SAM_masks = SAM_masks_binary[random_indices]  # Random negatives
                NEG_SAM_bboxes = SAM_bboxes_dest[random_indices]
                is_visible = torch.tensor(0.) #False
                random_idx = np.random.randint(N_masks)  # Pick random mask as fake positive
                POS_SAM_masks = SAM_masks_binary[random_idx]
                POS_SAM_bboxes = SAM_bboxes_dest[random_idx]
            
            # Clean up binary masks after use to save memory
            del SAM_masks_binary

            POS_mask_position = random.randint(0, self.N_masks_per_batch - 1) # Random position of the positive mask in the batch (prevent position bias)
            NEG_part1 = NEG_SAM_masks[:POS_mask_position]  # Split negatives: before positive
            NEG_part2 = NEG_SAM_masks[POS_mask_position:]  # Split negatives: after positive
            DEST_SAM_masks = torch.cat((NEG_part1, POS_SAM_masks.unsqueeze(0), NEG_part2), dim=0)  # Final batch: [neg...pos...neg]

            NEG_part1_bboxes = NEG_SAM_bboxes[:POS_mask_position]  # Corresponding bbox splits
            NEG_part2_bboxes = NEG_SAM_bboxes[POS_mask_position:]
            DEST_SAM_bboxes = torch.cat((NEG_part1_bboxes, POS_SAM_bboxes.unsqueeze(0), NEG_part2_bboxes), dim=0)  # Final bbox batch

        # In validation or test modes, we just return the SAM masks, and precompute which is the best SAM mask
        elif self.val_mode:
            # For GPU conversion optimization, we'll select mask indices without converting to binary
            # This requires temporarily converting to compute IoU for best mask selection
            visible_pixels = mask2_GT.sum()
            if visible_pixels > 0:  # Object is visible - ensure best mask is included
                is_visible = torch.tensor(1.) # True
                # Temporarily convert to binary just for IoU computation
                with torch.no_grad():
                    SAM_masks_binary_temp = self.mask_converter.indexed_to_binary(SAM_masks_indexed, N_masks)
                    best_original = self.get_best_mask(SAM_masks_binary_temp, mask2_GT)
                    del SAM_masks_binary_temp  # Clean up immediately
                
                # Sample remaining masks (N_masks_per_batch - 1) excluding best_original
                remaining_indices = np.setdiff1d(np.arange(N_masks), [best_original])
                N_remaining = self.N_masks_per_batch - 1
                
                if len(remaining_indices) >= N_remaining:
                    sampled_remaining = np.random.choice(remaining_indices, N_remaining, replace=False)
                else:
                    sampled_remaining = np.random.choice(remaining_indices, N_remaining, replace=True)
                
                # Insert best mask at random position
                POS_mask_position = random.randint(0, self.N_masks_per_batch - 1)
                indices_part1 = sampled_remaining[:POS_mask_position]
                indices_part2 = sampled_remaining[POS_mask_position:]
                random_indices = np.concatenate([indices_part1, [best_original], indices_part2])
                
            else:  # Object not visible - random sampling
                is_visible = torch.tensor(0.) # False
                POS_mask_position = 0
                if N_masks >= self.N_masks_per_batch:
                    random_indices = np.random.choice(N_masks, self.N_masks_per_batch, replace=False)
                else:
                    random_indices = np.random.choice(N_masks, self.N_masks_per_batch, replace=True)
            
            # Store the selected indices and pass indexed masks
            DEST_SAM_masks = SAM_masks_indexed  # Keep as indexed format
            DEST_SAM_masks_indices = random_indices
            DEST_SAM_bboxes = SAM_bboxes_dest[random_indices]
                
        else:  # Test mode
            # For test mode, we need all masks, so just pass the indexed format
            DEST_SAM_masks = SAM_masks_indexed  # Keep as indexed format
            DEST_SAM_masks_indices = np.arange(N_masks)  # All mask indices
            
            visible_pixels = mask2_GT.sum()
            if visible_pixels > 0:
                is_visible = torch.tensor(1.) # True
                # Temporarily convert to find best mask
                with torch.no_grad():
                    SAM_masks_binary_temp = self.mask_converter.indexed_to_binary(SAM_masks_indexed, N_masks)
                    POS_mask_position = self.get_best_mask(SAM_masks_binary_temp, mask2_GT)
                    del SAM_masks_binary_temp
            else:
                is_visible = torch.tensor(0.) # False
                POS_mask_position = 0
            DEST_SAM_bboxes = SAM_bboxes_dest
            if len(DEST_SAM_bboxes.shape) == 1:
                DEST_SAM_bboxes = torch.zeros((1, 4))
        
        # # Memory usage analysis of __getitem__ variables
        # locals_dict = locals()
        # print(f"\n=== __getitem__ Variables Memory Analysis (idx={idx_sample}) ===")
        # total_size = 0
        
        # # Key variables to check
        # key_vars = ['img1', 'img2', 'img1_torch', 'img2_torch', 
        #             'mask_SOURCE', 'mask2_GT', 'SAM_masks', 'SAM_bboxes_dest',
        #             'DEST_SAM_masks', 'DEST_SAM_bboxes', 'adj_matrix',
        #             'mask_annotation_SOURCE', 'mask_annotation_DEST']
        
        # for var_name in key_vars:
        #     if var_name in locals_dict:
        #         var_value = locals_dict[var_name]
        #         size = check_size(var_value, var_name)
        #         total_size += size
        
        # print(f"Total __getitem__ variables: {total_size / 1024 / 1024:.2f} MB")
        # print("-" * 60)

        # Memory cleanup: Remove large unnecessary variables before return
        
        # Original images (largest memory usage ~1.4MB each)
        del img1, img2
        # Annotation dictionaries (large JSON data)
        del mask_annotation_SOURCE, mask_annotation_DEST
        
        # Force garbage collection to actually free memory
        gc.collect()

        # Prepare return data based on mode
        return_data = {
            'SOURCE_img': img1_torch, 'SOURCE_mask': mask_SOURCE, 'SOURCE_bbox': bbox_from_mask(mask_SOURCE)[0], 'SOURCE_img_size': torch.tensor([self.h1, self.w1]),
            'GT_img': img2_torch, 'GT_mask': mask2_GT, 
            'DEST_SAM_masks': DEST_SAM_masks, 'DEST_SAM_bbox': DEST_SAM_bboxes, 'DEST_img_size': torch.tensor([self.h2, self.w2]),
            'is_visible': is_visible, 'POS_mask_position': torch.tensor(POS_mask_position),
            'pair_idx': torch.tensor(idx_sample),
            'img_pth1': img_pth1, 'img_pth2': img_pth2
        }
        
        # Add metadata for GPU conversion if not in training mode
        if not self.train_mode:
            return_data['DEST_SAM_masks_indices'] = torch.tensor(DEST_SAM_masks_indices)
            return_data['DEST_SAM_masks_num'] = torch.tensor(N_masks)
            return_data['DEST_SAM_masks_indexed'] = True  # Flag to indicate indexed format
        
        return return_data


def check_size(var, varname):
    import sys
    size_bytes = sys.getsizeof(var)
    size_mb = size_bytes / 1024 / 1024
    # print(f"{varname} The size : {size_mb:.2f} MB")
    return size_bytes


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import helpers
    from torch.utils.data import DataLoader
    
    print("Testing Masks_Dataset iteration...")
    
    # Test parameters
    root = "Ego-Exo4d"  # Update this path to your actual dataset
    mode = "train"  # Options: "train", "val", "test"
    
    # Test single sample
    try:
        # Test different modes - change this to test different dataset modes
        
        if mode == "train":
            dataset = Masks_Dataset(root, patch_size=14, reverse=False, N_masks_per_batch=32, order=2, train=True)
        elif mode == "val":
            dataset = Masks_Dataset(root, patch_size=14, reverse=False, N_masks_per_batch=32, order=2, val=True)
        elif mode == "test":
            dataset = Masks_Dataset(root, patch_size=14, reverse=False, N_masks_per_batch=32, order=2, test=True)
        
        print(f"{mode.capitalize()} dataset created with {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Shapes - SOURCE: {sample['SOURCE_img'].shape}, GT: {sample['GT_img'].shape}, MASKS: {sample['DEST_SAM_masks'].shape}")
        print(f"is_visible: {sample['is_visible']}, POS_mask_position: {sample['POS_mask_position']}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit()
    
    # Test DataLoader
    try:
        dataloader = DataLoader(dataset, batch_size=24, collate_fn=helpers.our_collate_fn, num_workers=0, pin_memory=True)
        batch = next(iter(dataloader))
        
        print(f"\nBatch test successful!")
        print(f"Batch shapes - SOURCE: {batch['SOURCE_img'].shape}, GT: {batch['GT_img'].shape}")
        print(f"DEST_SAM_masks: {batch['DEST_SAM_masks'].shape}")
        
    except Exception as e:
        print(f"DataLoader error: {e}")
    
    print("Test completed!")






