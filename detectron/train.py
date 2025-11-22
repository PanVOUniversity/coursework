"""Train Mask R-CNN model with Detectron2.

Registers synthetic frames dataset and trains Mask R-CNN model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from pycocotools import mask as mask_utils

try:
    from torch.fx import _symbolic_trace as fx_symbolic_trace
except ImportError:  # pragma: no cover
    fx_symbolic_trace = None

# Ensure RLE decodes produce writable arrays to avoid PyTorch warnings
_original_decode = mask_utils.decode


def _decode_writable(rle):
    decoded = _original_decode(rle)
    if isinstance(decoded, np.ndarray):
        decoded = np.array(decoded, copy=True, order='C')
        decoded.flags.writeable = True
    return decoded


mask_utils.decode = _decode_writable

# Provide default indexing argument for torch.meshgrid to avoid deprecation warning
_original_meshgrid = torch.meshgrid


def _meshgrid_with_indexing(*tensors, **kwargs):
    if "indexing" not in kwargs:
        kwargs["indexing"] = "ij"
    return _original_meshgrid(*tensors, **kwargs)


torch.meshgrid = _meshgrid_with_indexing

# Align with new torch.fx API to avoid compatibility warnings
if fx_symbolic_trace and hasattr(fx_symbolic_trace, "is_fx_tracing_symbolic_tracing"):
    fx_symbolic_trace.is_fx_tracing = fx_symbolic_trace.is_fx_tracing_symbolic_tracing


def register_dataset(coco_dir: Path, split: str = 'train'):
    """Register COCO dataset in Detectron2.
    
    Args:
        coco_dir: Directory containing COCO annotations
        split: Dataset split ('train' or 'val')
    """
    json_file = coco_dir / 'annotations' / f'instances_{split}.json'
    image_root = coco_dir / split
    
    # Check if files exist
    if not json_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {json_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    
    dataset_name = f"synthetic_frames_{split}"
    
    # Use closure to properly capture variables
    def dataset_func():
        return load_coco_json(str(json_file), str(image_root), dataset_name)
    
    DatasetCatalog.register(dataset_name, dataset_func)
    
    MetadataCatalog.get(dataset_name).set(
        thing_classes=['frame'],
        evaluator_type='coco',
        json_file=str(json_file),
        image_root=str(image_root),
    )
    
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN model')
    parser.add_argument('--coco-dir', type=str, default='data/coco', help='COCO dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory for checkpoints')
    parser.add_argument('--config', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 
                       help='Config file name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (optimized for Tesla T4 16GB)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger()
    
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Register datasets
    print("Registering datasets...")
    try:
        train_dataset = register_dataset(coco_dir, 'train')
        val_dataset = register_dataset(coco_dir, 'val')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run coco_converter.py first to create COCO dataset.")
        return
    
    # Get dataset size for proper MAX_ITER calculation
    train_data = DatasetCatalog.get(train_dataset)
    num_train_images = len(train_data)
    print(f"Training images: {num_train_images}")
    
    # Setup config
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config)
    
    # Modify config for our dataset
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "frame" class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    # Calculate MAX_ITER properly: iterations per epoch * number of epochs
    # One epoch = dataset_size / batch_size iterations
    iterations_per_epoch = max(1, num_train_images // args.batch_size)
    cfg.SOLVER.MAX_ITER = iterations_per_epoch * args.epochs
    
    cfg.SOLVER.BASE_LR = 0.01  # Calculated as 0.02 * (batch_size / 16) for batch_size=8
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = []
    
    # Output directory
    cfg.OUTPUT_DIR = str(output_dir)
    
    # Device
    if args.gpu and torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        cfg.MODEL.DEVICE = 'cpu'
        print("Using CPU")
    
    print(f"Training configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Iterations per epoch: {iterations_per_epoch}")
    print(f"  - Total iterations (MAX_ITER): {cfg.SOLVER.MAX_ITER}")
    
    # Create trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Train
    print(f"Starting training...")
    print(f"Output directory: {output_dir}")
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved in: {output_dir}")


if __name__ == '__main__':
    main()

