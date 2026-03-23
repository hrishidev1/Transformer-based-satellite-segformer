"""
Configuration management system for satellite segmentation
Supports loading from YAML files and command-line arguments
"""
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


class Config:
    """
    Configuration class that supports both dict and attribute access
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        # FIXED: Removed expensive to_dict() call to prevent overhead in hot loops
        return f"Config({self.__dict__})"
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def update(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Config) and isinstance(value, dict):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)


def load_config(config_path: str) -> Config:
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty config file: {config_path}")
    
    print(f"✓ Loaded config from {config_path}")
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Saved config to {save_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description='Satellite Imagery Semantic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'inference'],
        help='Mode: train, eval, or inference'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for resume/eval/inference'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,  # FIXED: Was 'cuda', which silently overwrote YAML configs
        choices=['cuda', 'cpu'],
        help='Override device for training/inference'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode (smaller dataset, fewer epochs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    return parser.parse_args()


def merge_args_with_config(config: Config, args):
    if args.device:
        config.training.device = args.device
    
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.lr:
        config.optimizer.lr = args.lr
    
    if args.seed:
        config.seed = args.seed
    
    if args.debug:
        print("[DEBUG MODE] Reducing dataset and epochs")
        config.training.epochs = 2
        config.training.batch_size = 2
    
    return config


def print_config(config: Config, indent=0):
    for key, value in config.__dict__.items():
        if isinstance(value, Config):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")