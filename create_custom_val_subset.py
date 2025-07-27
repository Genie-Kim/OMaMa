import json
import random
from pathlib import Path
from collections import defaultdict

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
    
    # Randomly select 10% of unique pairs
    num_to_select = max(1, int(len(unique_pairs) * 0.1))
    selected_pairs = random.sample(unique_pairs, num_to_select)
    
    print(f"Selected {num_to_select} unique combinations (10%)")
    
    # Collect all pairs belonging to selected combinations
    selected_data = []
    for key in selected_pairs:
        selected_data.extend(grouped_pairs[key])
    
    print(f"Total pairs in selected subset: {len(selected_data)}")
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(selected_data, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    # Print detailed statistics
    print("\nSelected combinations with frame details:")
    for i, key in enumerate(selected_pairs[:10]):
        take_id, ego_cam, exo_cam, object_name = key
        pairs = grouped_pairs[key]
        
        # Extract frame numbers from the paths
        frames = []
        for pair in pairs[:10]:  # Show first 10 frames
            frame_num = pair[0].split('//')[-1]  # Get frame number from path
            frames.append(frame_num)
        
        print(f"  {i+1}. Take: {take_id[:8]}..., Ego: {ego_cam}, Exo: {exo_cam}, Object: {object_name}")
        print(f"      Total frames: {len(pairs)}")
        print(f"      Sample frames: {', '.join(frames[:5])}", end="")
        if len(frames) > 5:
            print(f", ... ({len(pairs) - 5} more frames)")
        else:
            print()
    
    if len(selected_pairs) > 5:
        print(f"  ... and {len(selected_pairs) - 5} more combinations")
    
    # Overall statistics
    print(f"\nOverall statistics:")
    print(f"  Total unique combinations selected: {len(selected_pairs)}")
    print(f"  Total frame pairs in subset: {len(selected_data)}")
    print(f"  Average frames per combination: {len(selected_data) / len(selected_pairs):.1f}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility


    main("val_exoego_pairs.json", "custom_val_exoego_pairs_10.json")
    main("val_egoexo_pairs.json", "custom_val_egoexo_pairs_10.json")