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
except Exception:
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
    for run_dir in sorted(r for r in runs_root.iterdir() if r.is_dir()):
        csv_path = None
        if csv_name:
            candidate = run_dir / csv_name
            if candidate.exists():
                csv_path = candidate
        else:
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
        "observation.state": {"dtype": "float32", "shape": (8,), "names": joint_names},
        "observation.action": {"dtype": "float32", "shape": (8,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (8,), "names": joint_names},  # Mirror for compatibility

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
        "observation.digist1": {"dtype": "video", "shape": shapes["digit1"], "names": ["height", "width", "channels"]},
        "observation.digist2": {"dtype": "video", "shape": shapes["digit2"], "names": ["height", "width", "channels"]},

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


def is_probably_lerobot_dataset(root: Path) -> bool:
    """Heuristic check for an existing LeRobot dataset directory."""
    if not root.exists():
        return False
    try:
        entries = {p.name for p in root.iterdir()}
    except Exception:
        return False
    # Common anchors across versions
    anchors = {"meta", "videos", "data", "parquet"}
    return len(entries & anchors) > 0


def create_or_load_dataset(dataset_root: Path, repo_id: str, fps: int, features: Dict, append: bool = False) -> "LeRobotDataset":
    """
    - If append=True: require path exists and open it; error if not openable.
    - If append=False (default new-or-open):
        * path missing -> create()
        * path exists & empty -> rmdir then create()
        * path exists & looks like dataset -> open()
        * path exists & non-empty & not dataset-like -> raise to avoid clobbering.
    """
    if append:
        if not dataset_root.exists():
            raise FileNotFoundError(f"--append specified but dataset directory does not exist: {dataset_root}")
        try:
            if hasattr(LeRobotDataset, "open"):
                ds = LeRobotDataset.open(str(dataset_root))
            else:
                ds = LeRobotDataset(root=str(dataset_root))
            print(f"Opened existing dataset for appending at {dataset_root}")
            return ds
        except Exception as e:
            raise RuntimeError(f"Failed to open dataset for appending at {dataset_root}\n{e}")

    # New-or-open path
    if not dataset_root.exists():
        ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=str(dataset_root), features=features, use_videos=True)
        print(f"Created new dataset: {repo_id} at {dataset_root}")
        return ds

    # Path exists
    entries = list(dataset_root.iterdir())
    if len(entries) == 0:
        # Empty directory -> remove then create
        dataset_root.rmdir()
        ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=str(dataset_root), features=features, use_videos=True)
        print(f"Created new dataset at previously-empty path: {dataset_root}")
        return ds

    # Non-empty path
    if is_probably_lerobot_dataset(dataset_root):
        try:
            if hasattr(LeRobotDataset, "open"):
                ds = LeRobotDataset.open(str(dataset_root))
            else:
                ds = LeRobotDataset(root=str(dataset_root))
            print(f"Opened existing dataset at {dataset_root}")
            return ds
        except Exception as e:
            raise RuntimeError(f"Path looks like a dataset but failed to open: {dataset_root}\n{e}")
    else:
        raise FileExistsError(
            f"Target directory exists and is non-empty but does not look like a LeRobot dataset: {dataset_root}\n"
            f"Refusing to overwrite. Choose a new --dataset-root or clean the directory."
        )


def process_episode(dataset: "LeRobotDataset", csv_path: Path, episode_idx: int, config: Config) -> bool:
    """Process one episode (run folder) and add to dataset"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return False

    state_cols = [f"current_joint{i}_deg" for i in range(7)] + ["current_gripper_0to1"]
    action_cols = [f"target_joint{i}_deg" for i in range(7)] + ["target_gripper_0to1"]

    required_cols = state_cols + action_cols + [config.cam1_col, config.cam2_col, config.digit1_col, config.digit2_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing columns in {csv_path.name}: {missing_cols}")
        return False

    if "index" not in df.columns:
        df["index"] = np.arange(len(df), dtype=np.int64)

    frames_kept = 0
    frames_dropped = 0
    episode_frames = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Episode {episode_idx}", disable=not config.verbose):
        try:
            state = np.asarray([row[c] for c in state_cols], dtype=np.float32)
            action = np.asarray([row[c] for c in action_cols], dtype=np.float32)
        except Exception:
            frames_dropped += 1
            continue

        base_path = csv_path.parent
        paths = {
            "third": base_path / str(row[config.cam1_col]),
            "wrist": base_path / str(row[config.cam2_col]),
            "digit1": base_path / str(row[config.digit1_col]),
            "digit2": base_path / str(row[config.digit2_col]),
        }

        imgs: Dict[str, Optional[np.ndarray]] = {}
        missing = False
        for k, p in paths.items():
            img = load_image_rgb(p)
            if img is None:
                if config.strict_images:
                    print(f"Missing image {p}, skipping entire episode")
                    return False
                frames_dropped += 1
                missing = True
                break
            imgs[k] = img
        if missing:
            continue

        frame_index = frames_kept
        timestamp = float(frame_index) / float(config.fps)

        frame = {
            # low-dim
            "observation.state": state,
            "observation.action": action,
            "action": action,  # mirror

            # images
            "observation.images.third_people": imgs["third"],
            "observation.images.wrist_left": imgs["wrist"],
            "observation.digist1": imgs["digit1"],
            "observation.digist2": imgs["digit2"],

            # meta
            "language_instruction": config.instruction,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": int(episode_idx),
            "index": int(row["index"]),
            "task_index": int(config.task_index),
            "is_first": (frame_index == 0),
            "is_last": False,
            "is_terminal": False,
            "is_episode_successful": (not config.unsuccessful),
        }

        episode_frames.append(frame)
        frames_kept += 1

    if frames_kept == 0:
        print(f"No valid frames in episode {episode_idx}")
        return False

    # mark last
    episode_frames[-1]["is_last"] = True
    episode_frames[-1]["is_terminal"] = True

    for f in episode_frames:
        dataset.add_frame(f)

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
Example:
  python data_convert_to_lerobot.py \\
    --runs-root ./data/runs \\
    --dataset-root ./output/my_dataset \\
    --repo-id myusername/robot_manipulation \\
    --fps 15 \\
    --instruction "Pick up the object and place it in the bin"
        """
    )

    # Required arguments
    parser.add_argument("--runs-root", type=str, required=True, help="Root folder containing run subfolders")
    parser.add_argument("--dataset-root", type=str, required=True, help="Output dataset root directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID (e.g., username/dataset_name)")

    # Optional arguments
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--instruction", type=str, default="Put the corn onto the right tray.", help="Language instruction for the task")
    parser.add_argument("--task-index", type=int, default=0, help="Task index (default: 0)")
    parser.add_argument("--unsuccessful", action="store_true", help="Mark all episodes as unsuccessful")
    parser.add_argument("--csv-name", type=str, default=None, help="Fixed CSV filename (e.g., episode.csv)")
    parser.add_argument("--glob-csv", type=str, default="*.csv", help="Glob pattern for finding CSV files")
    parser.add_argument("--strict-images", action="store_true", help="Skip entire episode if any image is missing")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--append", action="store_true", help="Append episodes to an existing local dataset")

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
        verbose=not args.quiet,
    )

    # Validate paths
    runs_root = Path(config.runs_root)
    if not runs_root.exists():
        print(f"ERROR: Runs root directory does not exist: {runs_root}")
        sys.exit(1)

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
    for run_dir, csv_path in run_pairs[:3]:
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                base = csv_path.parent
                if config.cam1_col in df.columns:
                    sample_images["cam1"].extend([base / str(p) for p in df[config.cam1_col].dropna().head(5)])
                if config.cam2_col in df.columns:
                    sample_images["cam2"].extend([base / str(p) for p in df[config.cam2_col].dropna().head(5)])
                if config.digit1_col in df.columns:
                    sample_images["digit1"].extend([base / str(p) for p in df[config.digit1_col].dropna().head(5)])
                if config.digit2_col in df.columns:
                    sample_images["digit2"].extend([base / str(p) for p in df[config.digit2_col].dropna().head(5)])
        except Exception as e:
            print(f"Warning: Could not sample from {csv_path}: {e}")

    shapes = {
        "third": infer_image_shape(sample_images["cam1"], (224, 224, 3)),
        "wrist": infer_image_shape(sample_images["cam2"], (224, 224, 3)),
        "digit1": infer_image_shape(sample_images["digit1"], (240, 320, 3)),
        "digit2": infer_image_shape(sample_images["digit2"], (240, 320, 3)),
    }
    print("Image shapes:")
    print(f"  Third-person camera: {shapes['third']}")
    print(f"  Wrist camera: {shapes['wrist']}")
    print(f"  Digit sensor 1: {shapes['digit1']}")
    print(f"  Digit sensor 2: {shapes['digit2']}")

    # Build features and create/load dataset (do NOT pre-create the directory here)
    print("\nInitializing LeRobot dataset...")
    dataset_root = Path(config.dataset_root)
    features = build_dataset_features(shapes)
    dataset = create_or_load_dataset(dataset_root, config.repo_id, config.fps, features, append=args.append)

    # Process all episodes
    print(f"\nProcessing {len(run_pairs)} episodes...")
    successful_episodes = 0
    for episode_idx, (run_dir, csv_path) in enumerate(run_pairs):
        print(f"\n[{episode_idx + 1}/{len(run_pairs)}] Processing {run_dir.name}")
        ok = process_episode(dataset, csv_path, episode_idx, config)
        if ok:
            successful_episodes += 1
        else:
            print(f"Failed to process episode from {run_dir.name}")

    # Finalize dataset
    print("\nFinalizing dataset...")
    try:
        dataset.finalize()
    except AttributeError:
        # Older versions
        dataset.consolidate()

    # Summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"Successfully processed: {successful_episodes}/{len(run_pairs)} episodes")
    print(f"Dataset saved to: {dataset_root}")
    print(f"Repository ID: {config.repo_id}")
    print(f"FPS: {config.fps}")
    print(f"Task: {config.instruction}")
    print("\nTo use this dataset:")
    print("  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  dataset = LeRobotDataset('{dataset_root}')")
    print(f"  # or: dataset = LeRobotDataset.from_pretrained('{config.repo_id}')")


if __name__ == "__main__":
    main()