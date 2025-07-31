import os 
import numpy as np 
import torch
import random


def set_all_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def our_collate_fn(batch):
    """Custom collate function that handles both standard tensors and indexed mask format"""
    collated = {}
    
    for key in batch[0]:
        # Special handling for indexed masks in val/test mode
        if key == 'DEST_SAM_masks' and 'DEST_SAM_masks_indexed' in batch[0] and batch[0]['DEST_SAM_masks_indexed']:
            # For indexed masks, we need to handle them differently since they may have different shapes
            # Just pass them as a list to be processed later
            collated[key] = [sample[key] for sample in batch]
        elif key == 'DEST_SAM_masks_indexed':
            # Just take the first value since it should be the same for all samples in the batch
            collated[key] = batch[0][key]
        else:
            # Standard stacking for other tensors
            collated[key] = torch.stack([sample[key] for sample in batch])
    
    return collated

