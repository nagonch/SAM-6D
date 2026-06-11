#!/usr/bin/env python3
"""
Prepare SAM-6D templates from SpecTrack reference_views.

For each reference_views/OBJECT_REFLECTIVITY/ directory, creates a templates/
subfolder with rgb_N.png, mask_N.png, and xyz_N.npy (3D point cloud in mm).
"""

import os
import glob
import shutil
import numpy as np
from PIL import Image

REFERENCE_VIEWS_DIR = "/home/ngoncharov/SpecTrack_dataset/reference_views"


def read_camera_matrix(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float64)


def depth_to_xyz_mm(depth_mm, K):
    """Backproject 16-bit depth (mm) to (H, W, 3) XYZ array in mm."""
    H, W = depth_mm.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth_mm.astype(np.float32)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    return np.stack([X, Y, Z], axis=-1)  # (H, W, 3)


def main():
    ref_dirs = sorted(glob.glob(os.path.join(REFERENCE_VIEWS_DIR, "*")))
    ref_dirs = [d for d in ref_dirs if os.path.isdir(d) and os.path.basename(d) != "gifs"]

    total = len(ref_dirs)
    for dir_idx, ref_dir in enumerate(ref_dirs):
        name = os.path.basename(ref_dir)
        templates_dir = os.path.join(ref_dir, "templates")

        rgb_files = sorted(glob.glob(os.path.join(ref_dir, "rgb", "*.png")))
        existing_xyz = glob.glob(os.path.join(templates_dir, "xyz_*.npy"))

        if len(existing_xyz) == len(rgb_files) and len(rgb_files) > 0:
            print(f"[{dir_idx+1}/{total}] {name}: already done ({len(rgb_files)} views)", flush=True)
            continue

        os.makedirs(templates_dir, exist_ok=True)

        K = read_camera_matrix(os.path.join(ref_dir, "camera_matrix.txt"))
        depth_files = sorted(glob.glob(os.path.join(ref_dir, "depth_gt", "*.png")))
        mask_files = sorted(glob.glob(os.path.join(ref_dir, "mask", "*.png")))

        if not (len(rgb_files) == len(depth_files) == len(mask_files)):
            print(f"[{dir_idx+1}/{total}] {name}: MISMATCH rgb={len(rgb_files)} "
                  f"depth={len(depth_files)} mask={len(mask_files)}, skipping", flush=True)
            continue

        print(f"[{dir_idx+1}/{total}] {name}: preparing {len(rgb_files)} views ...", flush=True)

        for idx, (rgb_src, depth_src, mask_src) in enumerate(zip(rgb_files, depth_files, mask_files)):
            shutil.copy2(rgb_src, os.path.join(templates_dir, f"rgb_{idx}.png"))
            shutil.copy2(mask_src, os.path.join(templates_dir, f"mask_{idx}.png"))

            depth_mm = np.array(Image.open(depth_src), dtype=np.float32)
            xyz = depth_to_xyz_mm(depth_mm, K)
            np.save(os.path.join(templates_dir, f"xyz_{idx}.npy"), xyz.astype(np.float16))

        print(f"  -> done", flush=True)

    print(f"\nAll {total} template sets prepared.")


if __name__ == "__main__":
    main()
