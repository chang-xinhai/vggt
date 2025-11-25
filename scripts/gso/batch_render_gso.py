#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch rendering script for GSO dataset.

This script processes multiple GLB files in parallel or sequentially,
rendering them according to the Instant3D protocol.

Usage:
    python batch_render_gso.py --glb_folder /path/to/glb_folder --out_path /path/to/output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch render GSO dataset following Instant3D protocol'
    )
    
    # Path configuration
    parser.add_argument(
        '--glb_folder',
        type=str,
        required=True,
        help='Folder containing GLB files to render'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        required=True,
        help='Output directory for rendered images'
    )
    
    # Camera configuration
    parser.add_argument(
        '--distance',
        type=float,
        default=2.0,
        help='Camera distance from object center (default: 2.0)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Image resolution (default: 512)'
    )
    parser.add_argument(
        '--elevations',
        type=float,
        nargs='+',
        default=[0, 20, 40, 60],
        help='Elevation angles in degrees (default: [0, 20, 40, 60])'
    )
    parser.add_argument(
        '--num_azimuths',
        type=int,
        default=16,
        help='Number of azimuth angles per elevation (default: 16)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=128,
        help='Number of rendering samples (default: 128)'
    )
    
    # Processing configuration
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, sequential processing)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip objects that have already been rendered'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start rendering from this index (useful for resuming)'
    )
    parser.add_argument(
        '--max_objects',
        type=int,
        default=None,
        help='Maximum number of objects to render (default: all)'
    )
    
    # Blender configuration
    parser.add_argument(
        '--blender_path',
        type=str,
        default='blender',
        help='Path to Blender executable (default: "blender" in PATH)'
    )
    
    return parser.parse_args()


def check_blender(blender_path: str) -> bool:
    """Check if Blender is available."""
    try:
        result = subprocess.run(
            [blender_path, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_object_rendered(out_path: Path, obj_name: str, num_views: int) -> bool:
    """Check if an object has already been rendered."""
    obj_dir = out_path / obj_name
    if not obj_dir.exists():
        return False
    
    images_dir = obj_dir / "images"
    cameras_file = obj_dir / "cameras.json"
    
    if not images_dir.exists() or not cameras_file.exists():
        return False
    
    # Check if all views are present
    image_files = list(images_dir.glob("*.png"))
    return len(image_files) == num_views


def render_single_object(
    glb_path: Path,
    out_path: Path,
    script_path: Path,
    blender_path: str,
    args
) -> tuple:
    """
    Render a single object.
    
    Returns:
        (success: bool, obj_name: str, message: str)
    """
    obj_name = glb_path.stem
    
    try:
        # Build command
        cmd = [
            blender_path,
            '--background',
            '--python', str(script_path),
            '--',
            '--glb_path', str(glb_path),
            '--out_path', str(out_path),
            '--distance', str(args.distance),
            '--resolution', str(args.resolution),
            '--samples', str(args.samples),
            '--elevations', *[str(e) for e in args.elevations],
            '--num_azimuths', str(args.num_azimuths),
        ]
        
        if args.use_gpu:
            cmd.append('--use_gpu')
        
        # Run rendering
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per object
        )
        
        if result.returncode == 0:
            return (True, obj_name, "Success")
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return (False, obj_name, f"Rendering failed: {error_msg}")
    
    except subprocess.TimeoutExpired:
        return (False, obj_name, "Rendering timed out (>1 hour)")
    except Exception as e:
        return (False, obj_name, f"Error: {str(e)}")


def main():
    """Main batch rendering function."""
    args = parse_args()
    
    # Add use_gpu to args for compatibility
    args.use_gpu = True
    
    print("="*60)
    print("GSO Dataset Batch Rendering")
    print("="*60)
    
    # Check Blender
    print(f"Checking Blender at: {args.blender_path}")
    if not check_blender(args.blender_path):
        print(f"Error: Blender not found at '{args.blender_path}'")
        print("Please install Blender 4.x and ensure it's in your PATH")
        print("or specify the path with --blender_path")
        sys.exit(1)
    print("✓ Blender found")
    
    # Setup paths
    glb_folder = Path(args.glb_folder)
    out_path = Path(args.out_path)
    script_path = Path(__file__).parent / "render_gso_dataset.py"
    
    if not glb_folder.exists():
        print(f"Error: GLB folder not found: {glb_folder}")
        sys.exit(1)
    
    if not script_path.exists():
        print(f"Error: Rendering script not found: {script_path}")
        sys.exit(1)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all GLB files
    glb_files = sorted(glb_folder.glob("*.glb"))
    
    if not glb_files:
        print(f"Error: No GLB files found in {glb_folder}")
        sys.exit(1)
    
    print(f"Found {len(glb_files)} GLB files")
    
    # Apply start_index and max_objects filters
    if args.start_index > 0:
        glb_files = glb_files[args.start_index:]
        print(f"Starting from index {args.start_index}")
    
    if args.max_objects is not None:
        glb_files = glb_files[:args.max_objects]
        print(f"Limiting to {args.max_objects} objects")
    
    # Filter already rendered objects
    if args.skip_existing:
        num_views = len(args.elevations) * args.num_azimuths
        to_render = []
        for glb_path in glb_files:
            if not is_object_rendered(out_path, glb_path.stem, num_views):
                to_render.append(glb_path)
        
        skipped = len(glb_files) - len(to_render)
        if skipped > 0:
            print(f"Skipping {skipped} already rendered objects")
        glb_files = to_render
    
    if not glb_files:
        print("No objects to render!")
        return
    
    print(f"\nRendering {len(glb_files)} objects...")
    print(f"Output directory: {out_path}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Views per object: {len(args.elevations) * args.num_azimuths}")
    print(f"Parallel workers: {args.num_workers}")
    print("="*60)
    
    # Render objects
    success_count = 0
    failed_objects = []
    
    if args.num_workers == 1:
        # Sequential processing
        for glb_path in tqdm(glb_files, desc="Rendering"):
            success, obj_name, message = render_single_object(
                glb_path, out_path, script_path, args.blender_path, args
            )
            
            if success:
                success_count += 1
            else:
                failed_objects.append((obj_name, message))
                print(f"\n⚠ Failed: {obj_name} - {message}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    render_single_object,
                    glb_path, out_path, script_path, args.blender_path, args
                ): glb_path
                for glb_path in glb_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                success, obj_name, message = future.result()
                
                if success:
                    success_count += 1
                else:
                    failed_objects.append((obj_name, message))
                    print(f"\n⚠ Failed: {obj_name} - {message}")
    
    # Print summary
    print("\n" + "="*60)
    print("Rendering Summary")
    print("="*60)
    print(f"Total objects: {len(glb_files)}")
    print(f"Successfully rendered: {success_count}")
    print(f"Failed: {len(failed_objects)}")
    
    if failed_objects:
        print("\nFailed objects:")
        for obj_name, message in failed_objects:
            print(f"  - {obj_name}: {message}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
