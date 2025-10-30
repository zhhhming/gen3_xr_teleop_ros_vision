#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert robot manipulation data from run folders to LeRobot dataset format.

Each run folder represents one episode with:
- One CSV file containing robot state, action, and image paths
- Image folders for different camera views and tactile sensors

CSV Schema:
- index, current_joint0-6_deg, current_gripper_0to1 (robot state)
- target_joint0-6_deg, target_gripper_0to1 (robot action) 
- cam1_image_file (third-person camera)
- cam2_image_file (wrist camera)
- digit_D20583_file, digit_D20584_file (tactile sensors)

LeRobot Dataset Mappings:
- observation.state -> 8D: 7 joints (deg) + gripper (0-1)
- observation.action -> 8D: 7 target joints (deg) + target gripper (0-1)
- action -> Same as observation.action (for compatibility)
- observation.images.third_people -> cam1 RGB images
- observation.images.wrist_left -> cam2 RGB images  
- observation.digist1 -> digit_D20583 RGB images
- observation.digist2 -> digit_D20584 RGB images

Metadata per frame:
- language_instruction, is_first, is_last, is_terminal
- is_episode_successful, timestamp, frame_index
- episode_index, index, task_index
"""

import argparse
import sys
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Try importing LeRobot
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Install with: pip install lerobot")

@dataclass
class Config:
    """Configuration for dataset conversion"""
    runs_root: str
    dataset_root: str
    repo_id: str
    fps: int = 15
    
    # Task metadata
    instruction: str = "Put the corn onto the right tray."
    task_index: int = 0
    unsuccessful: bool = False
    
    # CSV settings
    csv_name: Optional[str] = "joint_gripper_log.csv"  # Fixed filename or None for glob
    glob_csv: str = "*.csv"
    
    # Column mappings
    cam1_col: str = "cam1_image_file"
    cam2_col: str = "cam2_image_file"
    digit1_col: str = "digit_D20583_file"
    digit2_col: str = "digit_D20584_file"
    
    # Processing options
    strict_images: bool = False  # If True, skip entire episode on missing images
    verbose: bool = True


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image as RGB numpy array"""
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            return np.array(img_rgb)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def infer_image_shape(paths: List[Path], default_shape=(224, 224, 3)) -> Tuple[int, int, int]:
    """Infer image shape from sample paths"""
    for path in paths[:10]:  # Check first 10 images
        arr = load_image_rgb(path)
        if arr is not None and arr.ndim == 3:
            return tuple(map(int, arr.shape))
    return default_shape


def collect_run_folders(runs_root: Path, csv_name: Optional[str], glob_pattern: str) -> List[Tuple[Path, Path]]:
    """Collect all run folders with their CSV files"""
    run_pairs = []
    
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
            
        # Find CSV file
        csv_path = None
        if csv_name:
            # Look for specific filename
            candidate = run_dir / csv_name
            if candidate.exists():
                csv_path = candidate
        else:
            # Use glob pattern
            csv_files = sorted(run_dir.glob(glob_pattern))
            if csv_files:
                csv_path = csv_files[0]
        
        if csv_path:
            run_pairs.append((run_dir, csv_path))
        else:
            print(f"Warning: No CSV found in {run_dir.name}, skipping")
    
    return run_pairs


def build_dataset_features(shapes: Dict[str, Tuple[int, int, int]]) -> Dict:
    """Build feature definitions for LeRobot dataset"""
    joint_names = [f"joint_{i}_deg" for i in range(7)] + ["gripper_0to1"]
    
    features = {
        # Robot state and action
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": joint_names,
        },
        "observation.action": {
            "dtype": "float32",
            "shape": (8,),
            "names": joint_names,
        },
        "action": {  # Mirror for compatibility
            "dtype": "float32",
            "shape": (8,),
            "names": joint_names,
        },
        
        # Camera images
        "observation.images.wrist_left": {
            "dtype": "video",
            "shape": shapes["wrist"],
            "names": ["height", "width", "channels"],
        },
        "observation.images.third_people": {
            "dtype": "video",
            "shape": shapes["third"],
            "names": ["height", "width", "channels"],
        },
        
        # Tactile sensor images
        "observation.digist1": {
            "dtype": "video",
            "shape": shapes["digit1"],
            "names": ["height", "width", "channels"],
        },
        "observation.digist2": {
            "dtype": "video",
            "shape": shapes["digit2"],
            "names": ["height", "width", "channels"],
        },
        
        # Metadata
        "language_instruction": {"dtype": "string", "shape": ()},
        "is_first": {"dtype": "bool", "shape": ()},
        "is_last": {"dtype": "bool", "shape": ()},
        "is_terminal": {"dtype": "bool", "shape": ()},
        "is_episode_successful": {"dtype": "bool", "shape": ()},
        "timestamp": {"dtype": "float64", "shape": ()},
        "frame_index": {"dtype": "int64", "shape": ()},
        "episode_index": {"dtype": "int64", "shape": ()},
        "index": {"dtype": "int64", "shape": ()},
        "task_index": {"dtype": "int64", "shape": ()},
    }
    
    return features


def create_or_load_dataset(dataset_root: Path, repo_id: str, fps: int, features: Dict) -> "LeRobotDataset":
    """Create new dataset or load existing one for appending"""
    dataset_exists = dataset_root.exists() and any(dataset_root.iterdir())
    
    if dataset_exists:
        try:
            # Try to open existing dataset
            if hasattr(LeRobotDataset, "open"):
                dataset = LeRobotDataset.open(str(dataset_root))
            else:
                dataset = LeRobotDataset(root=str(dataset_root))
            print(f"Opened existing dataset at {dataset_root}")
            return dataset
        except Exception as e:
            print(f"Could not open existing dataset: {e}")
            print("Creating new dataset...")
    
    # Create new dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(dataset_root),
        features=features,
        use_videos=True,
    )
    print(f"Created new dataset: {repo_id} at {dataset_root}")
    return dataset


def process_episode(
    dataset: "LeRobotDataset",
    csv_path: Path,
    episode_idx: int,
    config: Config
) -> bool:
    """Process one episode (run folder) and add to dataset"""
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return False
    
    # Column names for state and action
    state_cols = [f"current_joint{i}_deg" for i in range(7)] + ["current_gripper_0to1"]
    action_cols = [f"target_joint{i}_deg" for i in range(7)] + ["target_gripper_0to1"]
    
    # Check required columns
    required_cols = state_cols + action_cols + [
        config.cam1_col, config.cam2_col, 
        config.digit1_col, config.digit2_col
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns in {csv_path.name}: {missing_cols}")
        return False
    
    # Add index column if missing
    if 'index' not in df.columns:
        df['index'] = np.arange(len(df), dtype=np.int64)
    
    # Process frames
    frames_kept = 0
    frames_dropped = 0
    episode_frames = []
    
    for row_idx, row in tqdm(df.iterrows(), total=len(df), 
                              desc=f"Episode {episode_idx}", disable=not config.verbose):
        
        # Extract state and action
        try:
            state = np.array([row[col] for col in state_cols], dtype=np.float32)
            action = np.array([row[col] for col in action_cols], dtype=np.float32)
        except Exception as e:
            frames_dropped += 1
            continue
        
        # Load images
        base_path = csv_path.parent
        img_paths = {
            "third": base_path / str(row[config.cam1_col]),
            "wrist": base_path / str(row[config.cam2_col]),
            "digit1": base_path / str(row[config.digit1_col]),
            "digit2": base_path / str(row[config.digit2_col]),
        }
        
        images = {}
        for key, path in img_paths.items():
            img = load_image_rgb(path)
            if img is None:
                if config.strict_images:
                    print(f"Missing image {path}, skipping entire episode")
                    return False
                frames_dropped += 1
                break
            images[key] = img
        
        if len(images) != 4:
            continue
        
        # Create frame data
        frame_index = frames_kept
        timestamp = float(frame_index) / float(config.fps)
        
        frame_data = {
            # State and action
            "observation.state": state,
            "observation.action": action,
            "action": action,  # Mirror for compatibility
            
            # Images
            "observation.images.third_people": images["third"],
            "observation.images.wrist_left": images["wrist"],
            "observation.digist1": images["digit1"],
            "observation.digist2": images["digit2"],
            
            # Metadata
            "language_instruction": config.instruction,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": episode_idx,
            "index": int(row["index"]),
            "task_index": config.task_index,
            "is_first": (frame_index == 0),
            "is_last": False,  # Will be updated for last frame
            "is_terminal": False,  # Will be updated for last frame
            "is_episode_successful": not config.unsuccessful,
        }
        
        episode_frames.append(frame_data)
        frames_kept += 1
    
    if frames_kept == 0:
        print(f"No valid frames in episode {episode_idx}")
        return False
    
    # Update last frame metadata
    episode_frames[-1]["is_last"] = True
    episode_frames[-1]["is_terminal"] = True
    
    # Add all frames to dataset
    for frame in episode_frames:
        dataset.add_frame(frame)
    
    # Save episode
    try:
        dataset.save_episode()
    except Exception as e:
        print(f"Error saving episode: {e}")
        return False
    
    print(f"Episode {episode_idx}: {frames_kept} frames saved, {frames_dropped} dropped")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert robot manipulation data to LeRobot dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python to_lerobot_converter.py \\
    --runs-root ./data/runs \\
    --dataset-root ./output/my_dataset \\
    --repo-id myusername/robot_manipulation \\
    --fps 15 \\
    --instruction "Pick up the object and place it in the bin"
        """
    )
    
    # Required arguments
    parser.add_argument("--runs-root", type=str, required=True,
                        help="Root folder containing run subfolders")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Output dataset root directory")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Repository ID (e.g., username/dataset_name)")
    
    # Optional arguments
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second (default: 15)")
    parser.add_argument("--instruction", type=str, 
                        default="Put the corn onto the right tray.",
                        help="Language instruction for the task")
    parser.add_argument("--task-index", type=int, default=0,
                        help="Task index (default: 0)")
    parser.add_argument("--unsuccessful", action="store_true",
                        help="Mark all episodes as unsuccessful")
    parser.add_argument("--csv-name", type=str, default=None,
                        help="Fixed CSV filename (e.g., episode.csv)")
    parser.add_argument("--glob-csv", type=str, default="*.csv",
                        help="Glob pattern for finding CSV files")
    parser.add_argument("--strict-images", action="store_true",
                        help="Skip entire episode if any image is missing")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Check LeRobot availability
    if not LEROBOT_AVAILABLE:
        print("ERROR: LeRobot is not installed!")
        print("Please install it with: pip install lerobot")
        sys.exit(1)
    
    # Create configuration
    config = Config(
        runs_root=args.runs_root,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        fps=args.fps,
        instruction=args.instruction,
        task_index=args.task_index,
        unsuccessful=args.unsuccessful,
        csv_name=args.csv_name,
        glob_csv=args.glob_csv,
        strict_images=args.strict_images,
        verbose=not args.quiet
    )
    
    # Validate paths
    runs_root = Path(config.runs_root)
    if not runs_root.exists():
        print(f"ERROR: Runs root directory does not exist: {runs_root}")
        sys.exit(1)
    
    dataset_root = Path(config.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    # Collect run folders
    print(f"\nScanning for run folders in: {runs_root}")
    run_pairs = collect_run_folders(runs_root, config.csv_name, config.glob_csv)
    
    if not run_pairs:
        print("ERROR: No valid run folders found!")
        sys.exit(1)
    
    print(f"Found {len(run_pairs)} run folders to process")
    
    # Infer image shapes from first few runs
    print("\nInferring image shapes...")
    sample_images = {"cam1": [], "cam2": [], "digit1": [], "digit2": []}
    
    for run_dir, csv_path in run_pairs[:3]:  # Sample first 3 runs
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                base_path = csv_path.parent
                if config.cam1_col in df.columns:
                    sample_images["cam1"].extend([
                        base_path / str(p) for p in df[config.cam1_col].dropna().head(5)
                    ])
                if config.cam2_col in df.columns:
                    sample_images["cam2"].extend([
                        base_path / str(p) for p in df[config.cam2_col].dropna().head(5)
                    ])
                if config.digit1_col in df.columns:
                    sample_images["digit1"].extend([
                        base_path / str(p) for p in df[config.digit1_col].dropna().head(5)
                    ])
                if config.digit2_col in df.columns:
                    sample_images["digit2"].extend([
                        base_path / str(p) for p in df[config.digit2_col].dropna().head(5)
                    ])
        except Exception as e:
            print(f"Warning: Could not sample from {csv_path}: {e}")
    
    shapes = {
        "third": infer_image_shape(sample_images["cam1"], (224, 224, 3)),
        "wrist": infer_image_shape(sample_images["cam2"], (224, 224, 3)),
        "digit1": infer_image_shape(sample_images["digit1"], (240, 320, 3)),
        "digit2": infer_image_shape(sample_images["digit2"], (240, 320, 3)),
    }
    
    print(f"Image shapes:")
    print(f"  Third-person camera: {shapes['third']}")
    print(f"  Wrist camera: {shapes['wrist']}")
    print(f"  Digit sensor 1: {shapes['digit1']}")
    print(f"  Digit sensor 2: {shapes['digit2']}")
    
    # Build features and create/load dataset
    print(f"\nInitializing LeRobot dataset...")
    features = build_dataset_features(shapes)
    dataset = create_or_load_dataset(dataset_root, config.repo_id, config.fps, features)
    
    # Process all episodes
    print(f"\nProcessing {len(run_pairs)} episodes...")
    successful_episodes = 0
    
    for episode_idx, (run_dir, csv_path) in enumerate(run_pairs):
        print(f"\n[{episode_idx + 1}/{len(run_pairs)}] Processing {run_dir.name}")
        
        if process_episode(dataset, csv_path, episode_idx, config):
            successful_episodes += 1
        else:
            print(f"Failed to process episode from {run_dir.name}")
    
    # Finalize dataset
    print("\nFinalizing dataset...")
    dataset.consolidate()
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION COMPLETE")
    print("="*50)
    print(f"Successfully processed: {successful_episodes}/{len(run_pairs)} episodes")
    print(f"Dataset saved to: {dataset_root}")
    print(f"Repository ID: {config.repo_id}")
    print(f"FPS: {config.fps}")
    print(f"Task: {config.instruction}")
    
    # Print usage instructions
    print("\nTo use this dataset:")
    print("  from lerobot.datasets import LeRobotDataset")
    print(f"  dataset = LeRobotDataset('{dataset_root}')")
    print(f"  # or: dataset = LeRobotDataset.from_pretrained('{config.repo_id}')")
    

if __name__ == "__main__":
    main()