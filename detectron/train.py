"""Train Mask R-CNN model with Detectron2.

Registers synthetic frames dataset and trains Mask R-CNN model.
"""

import argparse
import os
import json
from pathlib import Path

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger


def register_dataset(coco_dir: Path, split: str = 'train'):
    """Register COCO dataset in Detectron2.
    
    Args:
        coco_dir: Directory containing COCO annotations
        split: Dataset split ('train' or 'val')
    """
    json_file = coco_dir / 'annotations' / f'instances_{split}.json'
    image_root = coco_dir / split
    
    dataset_name = f"synthetic_frames_{split}"
    
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(str(json_file), str(image_root), dataset_name)
    )
    
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
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
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
    train_dataset = register_dataset(coco_dir, 'train')
    val_dataset = register_dataset(coco_dir, 'val')
    
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config)
    
    # Modify config for our dataset
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "frame" class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.epochs * 1000  # Approximate epochs
    cfg.SOLVER.BASE_LR = 0.00025
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
    
    # Create trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Train
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved in: {output_dir}")


if __name__ == '__main__':
    main()

