"""Convert instance masks to COCO format.

Builds COCO annotations directly from metadata (like masks are built).
"""

import argparse
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as coco_mask
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import draw_rounded_rectangle_mask


def load_metadata(meta_path: Path) -> dict:
    """Load metadata JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_binary_mask_from_metadata(frame_meta: dict, width: int, height: int) -> np.ndarray:
    """Create binary mask for a frame directly from metadata.
    
    Optimized: creates mask only in ROI area instead of full image.
    
    Args:
        frame_meta: Frame metadata dictionary with x, y, w, h, border_radius
        width: Image width
        height: Image height
        
    Returns:
        Binary mask (H, W) with 255 for frame pixels, 0 for background
    """
    x = frame_meta['x']
    y = frame_meta['y']
    w = frame_meta['w']
    h = frame_meta['h']
    border_radius = frame_meta.get('border_radius', 0)
    
    # Clamp coordinates to image bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Create mask only in ROI area (with padding for border_radius)
    padding = border_radius + 2
    roi_x = max(0, x - padding)
    roi_y = max(0, y - padding)
    roi_w = min(width - roi_x, w + 2 * padding)
    roi_h = min(height - roi_y, h + 2 * padding)
    
    # Create ROI mask
    roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    
    # Adjust coordinates relative to ROI
    rel_x = x - roi_x
    rel_y = y - roi_y
    
    # Draw rounded rectangle directly on binary mask
    if border_radius <= 0:
        # Simple rectangle - much faster
        roi_mask[rel_y:rel_y + h, rel_x:rel_x + w] = 255
    else:
        # Use RGB temp mask only for ROI (much smaller)
        temp_rgb = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        draw_rounded_rectangle_mask(temp_rgb, rel_x, rel_y, w, h, border_radius, (255, 255, 255))
        roi_mask = (np.sum(temp_rgb, axis=2) > 0).astype(np.uint8) * 255
    
    # Create full mask and place ROI
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    binary_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask
    
    return binary_mask


def create_rle_for_rectangle(x: int, y: int, w: int, h: int, width: int, height: int) -> dict:
    """Create RLE for a simple rectangle by creating minimal mask.
    
    Optimized: creates mask only for the rectangle area, not the full image.
    
    Args:
        x, y, w, h: Rectangle coordinates
        width, height: Image dimensions
        
    Returns:
        RLE dictionary with 'size' and 'counts'
    """
    # Create minimal binary mask only for the rectangle (much faster than full image)
    # Create full-size mask but only fill the rectangle area
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    binary_mask[y:y+h, x:x+w] = 1
    
    # Encode to RLE
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def metadata_to_coco_annotation(frame_meta: dict, annotation_id: int, 
                                image_id: int, width: int, height: int) -> dict:
    """Convert frame metadata directly to COCO annotation format.
    
    Args:
        frame_meta: Frame metadata dictionary
        annotation_id: Unique annotation ID
        image_id: Image ID
        width: Image width
        height: Image height
        
    Returns:
        COCO annotation dictionary
    """
    # Get coordinates directly from metadata
    x = float(frame_meta['x'])
    y = float(frame_meta['y'])
    w = float(frame_meta['w'])
    h = float(frame_meta['h'])
    border_radius = frame_meta.get('border_radius', 0)
    
    # Clamp to image bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return None
    
    # Create RLE segmentation
    if border_radius <= 0:
        # Simple rectangle - create RLE directly without drawing mask
        rle = create_rle_for_rectangle(int(x), int(y), int(w), int(h), width, height)
        rle['counts'] = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
    else:
        # Rounded rectangle - need to draw mask for accurate RLE
        binary_mask = create_binary_mask_from_metadata(frame_meta, width, height)
        if not np.any(binary_mask):
            return None
        rle = coco_mask.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
    
    # RLE size from pycocotools is [height, width] (numpy format)
    # COCO format uses [width, height], so we need to convert
    rle_height, rle_width = rle['size']
    coco_size = [rle_width, rle_height]  # Convert to [width, height]
    
    # Bounding box from metadata
    bbox = [x, y, w, h]
    
    # Compute area
    if border_radius <= 0:
        area = w * h
    else:
        border_radius = min(border_radius, min(w, h) // 2)
        if border_radius <= 0:
            area = w * h
        else:
            area = w * h - 4 * border_radius * border_radius + np.pi * border_radius * border_radius
            area = max(0, area)
    
    # Create annotation
    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': 1,
        'segmentation': {
            'size': coco_size,
            'counts': rle['counts']
        },
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'z_index': frame_meta.get('z_index', 0),
        'border_radius': frame_meta.get('border_radius', 0),
    }
    
    return annotation


def convert_to_coco(meta_dir: Path, output_dir: Path, split: str = 'train'):
    """Convert metadata directly to COCO format.
    
    Creates only JSON annotations file, no image copying.
    
    Args:
        meta_dir: Directory with metadata JSON files
        output_dir: Output directory for COCO dataset (only JSON annotations)
        split: Dataset split ('train' or 'val')
    """
    # Create output directory only for annotations
    annotations_dir = output_dir / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all metadata files
    meta_files = sorted(meta_dir.glob('page_*.json'))
    
    if not meta_files:
        print(f"No metadata files found in {meta_dir}")
        return
    
    # Split into train/val (80/20)
    if split == 'train':
        meta_files = meta_files[:int(len(meta_files) * 0.8)]
    else:
        meta_files = meta_files[int(len(meta_files) * 0.8):]
    
    print(f"Processing {len(meta_files)} pages for {split} split...")
    
    # COCO data structures
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'frame', 'supercategory': 'none'}]
    
    annotation_id = 1
    
    for meta_path in tqdm(meta_files, desc=f"Converting {split}"):
        try:
            # Load metadata
            metadata = load_metadata(meta_path)
            page_id = metadata['page_id']
            
            # Get image dimensions from metadata
            width = metadata.get('page_width', 1920)
            height = metadata.get('page_height', 1080)
            
            # Add image entry (images stay in original location, not copied)
            image_entry = {
                'id': page_id,
                'file_name': f"page_{page_id}.png",  # Relative path from COCO root
                'width': width,
                'height': height,
            }
            images.append(image_entry)
            
            # Process each frame in metadata
            for frame in metadata['frames']:
                annotation = metadata_to_coco_annotation(
                    frame, annotation_id, page_id, width, height
                )
                
                if annotation:
                    annotations.append(annotation)
                    annotation_id += 1
        
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")
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
    parser = argparse.ArgumentParser(description='Convert metadata to COCO format')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory with metadata JSON files')
    parser.add_argument('--output-dir', type=str, default='data/coco', help='Output directory for COCO dataset (only JSON annotations)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
    
    args = parser.parse_args()
    
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    
    # Convert both splits
    print("Converting train split...")
    convert_to_coco(meta_dir, output_dir, 'train')
    
    print("\nConverting val split...")
    convert_to_coco(meta_dir, output_dir, 'val')
    
    print("\nDone! COCO dataset created.")


if __name__ == '__main__':
    main()

