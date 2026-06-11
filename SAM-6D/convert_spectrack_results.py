#!/usr/bin/env python3
"""
Convert per-frame SAM-6D detection_pem.json results to per-sequence numpy arrays.

Output: results_dir/numpy/{depth_mode}/{dataset_type}_{reflectivity}/{seq}.npy
Each .npy has shape (N_frames, 4, 4), float64, containing the best-scoring
pose for each frame in temporal order.
"""

import os
import glob
import json
import argparse
import numpy as np


def load_best_pose(pem_json_path):
    """Load the highest-scoring pose from a detection_pem.json file.

    Returns a (4, 4) float64 array, or None if the file is missing/empty.
    """
    if not os.path.exists(pem_json_path):
        return None
    try:
        with open(pem_json_path) as f:
            detections = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if not detections:
        return None

    # Pick detection with highest score
    best = max(detections, key=lambda d: d.get("score", 0.0))

    R = np.array(best["R"], dtype=np.float64).reshape(3, 3)
    # SAM-6D stores t in mm; convert to meters to match results_pnp format
    t = np.array(best["t"], dtype=np.float64).reshape(3) / 1000.0

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def convert_sequence(seq_result_dir):
    """Convert all frames for one sequence.

    seq_result_dir layout: seq_result_dir/LF_XXXX/sam6d_results/detection_pem.json

    Returns (N_frames, 4, 4) numpy array (NaN pose for missing frames).
    """
    frame_dirs = sorted(glob.glob(os.path.join(seq_result_dir, "LF_*")))
    if not frame_dirs:
        return None

    poses = []
    for fd in frame_dirs:
        pem_path = os.path.join(fd, "sam6d_results", "detection_pem.json")
        pose = load_best_pose(pem_path)
        if pose is None:
            pose = np.full((4, 4), np.nan, dtype=np.float64)
        poses.append(pose)

    return np.stack(poses, axis=0)  # (N, 4, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Path to results_sam6d/SpecTrack/")
    args = parser.parse_args()

    results_dir = args.results_dir
    numpy_root = os.path.join(results_dir, "numpy")

    # Expected structure: results_dir/{depth_mode}/{dataset_type}_{refl}/{seq}/LF_*/
    depth_modes = [d for d in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, d)) and d != "numpy"]

    total_seqs = 0
    total_missing = 0

    for depth_mode in sorted(depth_modes):
        dm_dir = os.path.join(results_dir, depth_mode)
        dataset_dirs = [d for d in os.listdir(dm_dir)
                        if os.path.isdir(os.path.join(dm_dir, d))]

        for dataset_key in sorted(dataset_dirs):
            dk_dir = os.path.join(dm_dir, dataset_key)
            seq_dirs = [d for d in os.listdir(dk_dir)
                        if os.path.isdir(os.path.join(dk_dir, d))]

            out_dir = os.path.join(numpy_root, depth_mode, dataset_key)
            os.makedirs(out_dir, exist_ok=True)

            for seq in sorted(seq_dirs):
                seq_result_dir = os.path.join(dk_dir, seq)
                poses = convert_sequence(seq_result_dir)
                if poses is None:
                    print(f"  SKIP (no frames): {depth_mode}/{dataset_key}/{seq}")
                    continue

                nan_count = int(np.any(np.isnan(poses), axis=(1, 2)).sum())
                out_path = os.path.join(out_dir, f"{seq}.npy")
                np.save(out_path, poses)

                status = f"shape={poses.shape}"
                if nan_count:
                    status += f"  MISSING={nan_count}/{len(poses)}"
                    total_missing += nan_count
                print(f"  {depth_mode}/{dataset_key}/{seq}: {status}")
                total_seqs += 1

    print(f"\nConverted {total_seqs} sequences.")
    if total_missing:
        print(f"WARNING: {total_missing} frames have no pose (saved as NaN).")
    print(f"Numpy results saved to: {numpy_root}")


if __name__ == "__main__":
    main()
