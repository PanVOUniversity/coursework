"""Inference and postprocessing for frame segmentation.

Loads trained model, performs inference, and computes shifts to resolve overlaps.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import (
    min_shift_to_resolve_overlap,
    compute_overlap_pixels,
    bbox_from_mask
)
from utils.color_mapping import color_to_id


def load_metadata(meta_path: Path) -> dict:
    """Load metadata JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_z_index_from_metadata(instance_id: int, metadata: dict) -> int:
    """Get z-index for instance from metadata."""
    for frame in metadata.get('frames', []):
        if frame['id'] == instance_id:
            return frame.get('z_index', 0)
    return 0


def determine_top_bottom(mask1: np.ndarray, mask2: np.ndarray, 
                        id1: int, id2: int, metadata: dict = None) -> Tuple[int, int]:
    """Determine which instance is on top.
    
    Args:
        mask1: First instance mask
        mask2: Second instance mask
        id1: First instance ID
        id2: Second instance ID
        metadata: Optional metadata dictionary
        
    Returns:
        (top_id, bottom_id)
    """
    # Try to use z_index from metadata
    if metadata:
        z1 = get_z_index_from_metadata(id1, metadata)
        z2 = get_z_index_from_metadata(id2, metadata)
        if z1 != z2:
            if z1 > z2:
                return (id1, id2)
            else:
                return (id2, id1)
    
    # Fallback: use overlap coverage
    # Count pixels where both masks overlap
    overlap = np.logical_and(mask1 > 0, mask2 > 0)
    if np.sum(overlap) == 0:
        return (id1, id2)  # No overlap, arbitrary
    
    # Check which mask covers more of the overlap area
    mask1_coverage = np.sum(np.logical_and(overlap, mask1 > 0))
    mask2_coverage = np.sum(np.logical_and(overlap, mask2 > 0))
    
    if mask1_coverage > mask2_coverage:
        return (id1, id2)
    else:
        return (id2, id1)


def process_overlaps(predictions: List[Dict], metadata: dict = None) -> List[Dict]:
    """Process predictions to find overlaps and compute shifts.
    
    Args:
        predictions: List of prediction dictionaries with 'mask', 'bbox', 'instance_id'
        metadata: Optional metadata dictionary
        
    Returns:
        List of overlap dictionaries
    """
    overlaps = []
    
    # Convert predictions to binary masks
    masks = []
    bboxes = []
    ids = []
    
    for pred in predictions:
        mask = pred['mask'].astype(np.uint8)
        masks.append(mask)
        bboxes.append(pred['bbox'])
        ids.append(pred['instance_id'])
    
    # Check all pairs
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            mask1 = masks[i]
            mask2 = masks[j]
            id1 = ids[i]
            id2 = ids[j]
            bbox1 = bboxes[i]
            bbox2 = bboxes[j]
            
            # Compute overlap
            overlap_pixels = compute_overlap_pixels(mask1, mask2)
            
            if overlap_pixels == 0:
                continue
            
            # Determine which is on top
            top_id, bottom_id = determine_top_bottom(mask1, mask2, id1, id2, metadata)
            
            # Compute shifts
            dx_top, dy_top, dx_bottom, dy_bottom = min_shift_to_resolve_overlap(bbox1, bbox2)
            
            # Adjust shifts based on which is top/bottom
            if top_id == id1:
                shift_top = [float(dx_top), float(dy_top)]
                shift_bottom = [float(dx_bottom), float(dy_bottom)]
            else:
                shift_top = [float(dx_bottom), float(dy_bottom)]
                shift_bottom = [float(dx_top), float(dy_top)]
            
            overlaps.append({
                'top_id': int(top_id),
                'bottom_id': int(bottom_id),
                'overlap_pixels': int(overlap_pixels),
                'shift_top': shift_top,
                'shift_bottom': shift_bottom,
            })
    
    return overlaps


def visualize_overlaps(image: np.ndarray, overlaps: List[Dict], predictions: List[Dict], 
                       output_path: Path):
    """Visualize overlaps on image.
    
    Args:
        image: Input image
        overlaps: List of overlap dictionaries
        predictions: List of predictions
        output_path: Path to save visualization
    """
    vis_image = image.copy()
    
    # Draw overlap regions
    for overlap in overlaps:
        top_id = overlap['top_id']
        bottom_id = overlap['bottom_id']
        
        # Find corresponding masks
        top_mask = None
        bottom_mask = None
        
        for pred in predictions:
            if pred['instance_id'] == top_id:
                top_mask = pred['mask']
            if pred['instance_id'] == bottom_id:
                bottom_mask = pred['mask']
        
        if top_mask is None or bottom_mask is None:
            continue
        
        # Highlight overlap region
        overlap_region = np.logical_and(top_mask > 0, bottom_mask > 0)
        vis_image[overlap_region] = [255, 0, 0]  # Red for overlap
    
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Inference and postprocessing')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                       help='Config file name')
    parser.add_argument('--input-dir', type=str, default='data/screenshots', help='Input directory with screenshots')
    parser.add_argument('--output-dir', type=str, default='data/results', help='Output directory for results')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory with metadata JSON files')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    
    if args.gpu and torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else:
        cfg.MODEL.DEVICE = 'cpu'
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Find all screenshots
    screenshot_files = sorted(input_dir.glob('page_*.png'))
    
    if not screenshot_files:
        print(f"No screenshot files found in {input_dir}")
        return
    
    print(f"Processing {len(screenshot_files)} screenshots...")
    
    for screenshot_path in tqdm(screenshot_files, desc="Inference"):
        try:
            # Extract page ID
            page_id = int(screenshot_path.stem.replace('page_', ''))
            
            # Load image
            image = cv2.imread(str(screenshot_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            outputs = predictor(image)
            
            # Extract predictions
            instances = outputs['instances']
            predictions = []
            
            for i in range(len(instances)):
                mask = instances.pred_masks[i].cpu().numpy().astype(np.uint8) * 255
                bbox = instances.pred_boxes[i].tensor[0].cpu().numpy().tolist()
                score = instances.scores[i].cpu().item()
                
                # Convert bbox from (x1, y1, x2, y2) to (x, y, w, h)
                x, y, x2, y2 = bbox
                bbox_xywh = [float(x), float(y), float(x2 - x), float(y2 - y)]
                
                predictions.append({
                    'instance_id': i + 1,
                    'mask': mask,
                    'bbox': bbox_xywh,
                    'score': score,
                })
            
            # Load metadata if available
            metadata = None
            meta_path = meta_dir / f"page_{page_id}.json"
            if meta_path.exists():
                metadata = load_metadata(meta_path)
            
            # Process overlaps
            overlaps = process_overlaps(predictions, metadata)
            
            # Save results
            result = {
                'page_id': page_id,
                'num_detections': len(predictions),
                'overlaps': overlaps,
            }
            
            result_path = output_dir / f"page_{page_id}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            # Visualize overlaps
            if overlaps:
                vis_path = output_dir / f"page_{page_id}_overlap_mask.png"
                visualize_overlaps(image_rgb, overlaps, predictions, vis_path)
        
        except Exception as e:
            print(f"Error processing {screenshot_path}: {e}")
            continue
    
    print(f"Done! Results saved in {output_dir}")


if __name__ == '__main__':
    main()

