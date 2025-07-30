import json
import random
from pathlib import Path
from collections import defaultdict

# Custom validation subset sampling strategy:
# 1. Groups frame pairs by (take_id, ego_camera, exo_camera, object_name)
# 2. Randomly selects 60% of unique combinations
# 3. Applies 1/16 temporal subsampling with random start + consecutive window
# 4. Filters groups with <5 frames after reduction
def extract_info_from_path(path):
    """Extract take_id, camera name, and object name from the path."""
    parts = path.split('//')
    if len(parts) >= 6:
        take_id = parts[1]
        camera_name = parts[2]
        object_name = parts[3]
        return take_id, camera_name, object_name
    return None, None, None

def main(input_file_name, output_file_name):
    # Read the original validation pairs
    input_file = Path(f"Ego-Exo4d/dataset_jsons/{input_file_name}")
    output_file = Path(f"Ego-Exo4d/dataset_jsons/{output_file_name}")
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        all_pairs = json.load(f)
    
    print(f"Total pairs: {len(all_pairs)}")
    
    # Group pairs by (take_id, ego_camera, exo_camera, object_name)
    grouped_pairs = defaultdict(list)
    
    for pair in all_pairs:
        # Extract camera and object info from both paths
        take_id1, camera1, object1 = extract_info_from_path(pair[0])
        take_id2, camera2, object2 = extract_info_from_path(pair[2])
        
        # Only group pairs with matching take_id and object_name
        if take_id1 == take_id2 and object1 == object2 and take_id1:
            # Determine ego/exo based on "aria" in camera name
            if "aria" in camera1.lower():
                key = (take_id1, camera1, camera2, object1)
                grouped_pairs[key].append(pair)
            elif "aria" in camera2.lower():
                # Reorder to keep ego first
                key = (take_id1, camera2, camera1, object1)
                grouped_pairs[key].append([pair[2], pair[3], pair[0], pair[1]])
    
    # Get unique camera pairs
    unique_pairs = list(grouped_pairs.keys())
    print(f"Unique (take_id, ego_camera, exo_camera, object_name) combinations: {len(unique_pairs)}")
    
    # Randomly select 60% of unique pairs
    num_to_select = max(1, int(len(unique_pairs) * 0.6))
    selected_pairs = random.sample(unique_pairs, num_to_select)
    
    print(f"Selected {num_to_select} unique combinations (60%)")
    
    # Collect pairs with 1/16 reduction using random start + consecutive window
    selected_data = []
    skipped_groups = 0
    
    for key in selected_pairs:
        # Sort frames by frame number
        all_frames = sorted(grouped_pairs[key], key=lambda x: int(x[0].split('//')[-1]))
        total_frames = len(all_frames)
        window_size = total_frames // 16
        
        # Skip if window size is less than 5
        if window_size < 5:
            skipped_groups += 1
            continue
            
        # Random start within first 16 frames (or less if total frames < 16)
        max_start = min(16, total_frames - window_size)
        start_idx = random.randint(0, max_start - 1) if max_start > 1 else 0
        
        # Select consecutive window
        selected_frames = all_frames[start_idx:start_idx + window_size]
        selected_data.extend(selected_frames)
    
    print(f"Total pairs in selected subset: {len(selected_data)}")
    print(f"Skipped {skipped_groups} groups with less than 5 frames after 1/16 reduction")
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(selected_data, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    # Print detailed statistics
    print("\nSelected combinations with frame details:")
    actual_groups = 0
    for i, key in enumerate(selected_pairs[:10]):
        take_id, ego_cam, exo_cam, object_name = key
        all_frames = sorted(grouped_pairs[key], key=lambda x: int(x[0].split('//')[-1]))
        total_frames = len(all_frames)
        window_size = total_frames // 16
        
        if window_size < 5:
            continue
            
        actual_groups += 1
        
        # Find the actual selected frames for this group
        max_start = min(16, total_frames - window_size)
        # Note: we can't know exact start_idx here as it was random, but we can show the window size
        
        print(f"  {actual_groups}. Take: {take_id[:8]}..., Ego: {ego_cam}, Exo: {exo_cam}, Object: {object_name}")
        print(f"      Original frames: {total_frames}, Window size: {window_size} (1/16 reduction)")
        print(f"      Frame range: {all_frames[0][0].split('//')[-1]} - {all_frames[-1][0].split('//')[-1]}")
        
    if len(selected_pairs) - skipped_groups > 10:
        print(f"  ... and {len(selected_pairs) - skipped_groups - 10} more combinations")
    
    # Overall statistics
    actual_combinations = len(selected_pairs) - skipped_groups
    print(f"\nOverall statistics:")
    print(f"  Total unique combinations selected: {len(selected_pairs)}")
    print(f"  Combinations after filtering (≥5 frames): {actual_combinations}")
    print(f"  Total frame pairs in subset: {len(selected_data)}")
    if actual_combinations > 0:
        print(f"  Average frames per combination: {len(selected_data) / actual_combinations:.1f}")
    print(f"  Data reduction: {len(selected_data)} frames from original dataset")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility


    main("val_exoego_pairs.json", "custom_val_exoego_pairs_10.json")
    main("val_egoexo_pairs.json", "custom_val_egoexo_pairs_10.json")