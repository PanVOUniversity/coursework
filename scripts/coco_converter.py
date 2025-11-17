"""Convert instance masks to COCO format.

Extracts contours from instance masks and converts to COCO JSON format.
"""

import argparse
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as coco_mask
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.color_mapping import color_to_id


def load_metadata(meta_path: Path) -> dict:
    """Load metadata JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_instance_from_mask(mask_image: np.ndarray, instance_id: int) -> np.ndarray:
    """Extract binary mask for a specific instance ID.
    
    Args:
        mask_image: RGB mask image (H, W, 3)
        instance_id: Instance ID to extract
        
    Returns:
        Binary mask (H, W) with 255 for instance pixels, 0 for background
    """
    binary_mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
    
    # Find pixels matching this instance ID
    for y in range(mask_image.shape[0]):
        for x in range(mask_image.shape[1]):
            r, g, b = mask_image[y, x]
            pixel_id = color_to_id(int(r), int(g), int(b))
            if pixel_id == instance_id:
                binary_mask[y, x] = 255
    
    return binary_mask


def mask_to_coco_annotation(mask_image: np.ndarray, instance_id: int, 
                           annotation_id: int, image_id: int, 
                           metadata: dict) -> dict:
    """Convert instance mask to COCO annotation format.
    
    Args:
        mask_image: RGB mask image (H, W, 3)
        instance_id: Instance ID
        annotation_id: Unique annotation ID
        image_id: Image ID
        metadata: Metadata dictionary with frame info
        
    Returns:
        COCO annotation dictionary
    """
    # Extract binary mask for this instance
    binary_mask = extract_instance_from_mask(mask_image, instance_id)
    
    if np.sum(binary_mask) == 0:
        return None
    
    # Find frame metadata
    frame_meta = None
    for frame in metadata['frames']:
        if frame['id'] == instance_id:
            frame_meta = frame
            break
    
    if frame_meta is None:
        return None
    
    # Extract contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour (but preserve rounded corners)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to polygon format [x1, y1, x2, y2, ...]
    polygon = []
    for point in simplified_contour:
        polygon.extend([float(point[0][0]), float(point[0][1])])
    
    # Convert to RLE format (more compact)
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    
    # Compute bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = [float(x), float(y), float(w), float(h)]
    area = float(cv2.contourArea(largest_contour))
    
    # Create annotation
    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': 1,  # "frame" category
        'segmentation': {
            'size': [mask_image.shape[1], mask_image.shape[0]],
            'counts': rle['counts']
        },
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'z_index': frame_meta.get('z_index', 0),
        'border_radius': frame_meta.get('border_radius', 0),
    }
    
    return annotation


def convert_to_coco(mask_dir: Path, meta_dir: Path, output_dir: Path, split: str = 'train'):
    """Convert instance masks to COCO format.
    
    Args:
        mask_dir: Directory with instance mask PNG files
        meta_dir: Directory with metadata JSON files
        output_dir: Output directory for COCO dataset
        split: Dataset split ('train' or 'val')
    """
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    images_dir = output_dir / split
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mask files
    mask_files = sorted(mask_dir.glob('page_*_instance_mask.png'))
    
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return
    
    # Split into train/val (80/20)
    if split == 'train':
        mask_files = mask_files[:int(len(mask_files) * 0.8)]
    else:
        mask_files = mask_files[int(len(mask_files) * 0.8):]
    
    print(f"Processing {len(mask_files)} masks for {split} split...")
    
    # COCO data structures
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'frame', 'supercategory': 'none'}]
    
    annotation_id = 1
    
    for mask_path in tqdm(mask_files, desc=f"Converting {split}"):
        try:
            # Extract page ID
            page_id = int(mask_path.stem.replace('page_', '').replace('_instance_mask', ''))
            
            # Load mask
            mask_image = cv2.imread(str(mask_path))
            if mask_image is None:
                continue
            
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            height, width = mask_image.shape[:2]
            
            # Load metadata
            meta_path = meta_dir / f"page_{page_id}.json"
            if not meta_path.exists():
                continue
            
            metadata = load_metadata(meta_path)
            
            # Add image entry
            image_entry = {
                'id': page_id,
                'file_name': f"page_{page_id}.png",
                'width': width,
                'height': height,
            }
            images.append(image_entry)
            
            # Copy screenshot to COCO images directory
            screenshot_path = Path('data/screenshots') / f"page_{page_id}.png"
            if screenshot_path.exists():
                shutil.copy(screenshot_path, images_dir / f"page_{page_id}.png")
            
            # Process each frame in metadata
            for frame in metadata['frames']:
                instance_id = frame['id']
                annotation = mask_to_coco_annotation(
                    mask_image, instance_id, annotation_id, page_id, metadata
                )
                
                if annotation:
                    annotations.append(annotation)
                    annotation_id += 1
        
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue
    
    # Create COCO JSON
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    
    # Save COCO JSON
    output_json = annotations_dir / f"instances_{split}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Done! Created COCO dataset:")
    print(f"  - Images: {len(images)}")
    print(f"  - Annotations: {len(annotations)}")
    print(f"  - JSON: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Convert instance masks to COCO format')
    parser.add_argument('--mask-dir', type=str, default='data/masks', help='Directory with instance mask PNG files')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory with metadata JSON files')
    parser.add_argument('--output-dir', type=str, default='data/coco', help='Output directory for COCO dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
    
    args = parser.parse_args()
    
    mask_dir = Path(args.mask_dir)
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    
    # Convert both splits
    print("Converting train split...")
    convert_to_coco(mask_dir, meta_dir, output_dir, 'train')
    
    print("\nConverting val split...")
    convert_to_coco(mask_dir, meta_dir, output_dir, 'val')
    
    print("\nDone! COCO dataset created.")


if __name__ == '__main__':
    main()

