#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to download and prepare GSO dataset.

This script automates the process of:
1. Checking prerequisites
2. Downloading GSO dataset (optional)
3. Converting to GLB format
4. Setting up directory structure

Usage:
    python prepare_gso.py --output_dir /path/to/gso_workspace
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run(
            [cmd, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    
    issues = []
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        issues.append(f"Python 3.8+ required (found {py_version.major}.{py_version.minor})")
    else:
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # Check Blender
    if check_command('blender'):
        print("✓ Blender found")
    else:
        issues.append("Blender not found. Install from https://www.blender.org/download/")
    
    # Check Python packages
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        issues.append("numpy not installed: pip install numpy")
    
    try:
        import PIL
        print("✓ Pillow")
    except ImportError:
        issues.append("Pillow not installed: pip install Pillow")
    
    try:
        from tqdm import tqdm
        print("✓ tqdm")
    except ImportError:
        issues.append("tqdm not installed: pip install tqdm")
    
    return issues


def clone_gso_utils(output_dir: Path):
    """Clone GSO-Data-Utils repository."""
    utils_dir = output_dir / "GSO-Data-Utils"
    
    if utils_dir.exists():
        print(f"GSO-Data-Utils already exists at {utils_dir}")
        return utils_dir
    
    print("Cloning GSO-Data-Utils repository...")
    try:
        subprocess.run(
            ['git', 'clone', 'https://github.com/TO-Hitori/GSO-Data-Utils.git', str(utils_dir)],
            check=True
        )
        print(f"✓ Cloned to {utils_dir}")
        return utils_dir
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return None


def print_instructions(output_dir: Path):
    """Print next steps instructions."""
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Download GSO dataset:")
    print(f"   cd {output_dir}/GSO-Data-Utils")
    print("   python download_collection.py -d ../GSO_data")
    print("\n   Alternative: Download from Google Drive")
    print("   https://drive.google.com/drive/folders/1Dtqiyt0QP9dabiaTN5qONdb8avc0aNg6")
    
    print("\n2. Extract downloaded .zip files to GSO_data/")
    
    print("\n3. Convert to GLB format:")
    print(f"   cd {output_dir}/GSO-Data-Utils")
    print("   python obj2glb_batch.py --input_path ../GSO_data --output_path ../GSO_GLB")
    
    print("\n4. Render images:")
    print(f"   cd {Path(__file__).parent.parent.parent}")
    print(f"   python scripts/gso/batch_render_gso.py \\")
    print(f"       --glb_folder {output_dir}/GSO_GLB \\")
    print(f"       --out_path {output_dir}/GSO_rendered")
    
    print("\n5. Evaluate VGGT-NVS:")
    print("   python eval_nvs_gso.py \\")
    print(f"       --gso_dir {output_dir}/GSO_rendered \\")
    print("       --checkpoint /path/to/checkpoint.pt")
    
    print("\n" + "="*60)
    print(f"\nWorkspace created at: {output_dir}")
    print("\nFor detailed instructions, see:")
    print("  - scripts/gso/README.md")
    print("  - docs/GSO_Dataset_Preparation.md")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare GSO dataset workspace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/gso/prepare_gso.py --output_dir ~/gso_workspace
  python scripts/gso/prepare_gso.py --output_dir /data/gso --skip_clone
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to setup GSO workspace'
    )
    parser.add_argument(
        '--skip_clone',
        action='store_true',
        help='Skip cloning GSO-Data-Utils (if already exists)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    
    print("="*60)
    print("GSO Dataset Preparation Helper")
    print("="*60)
    print(f"\nWorkspace: {output_dir}\n")
    
    # Check prerequisites
    issues = check_prerequisites()
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease resolve these issues before continuing.")
        sys.exit(1)
    
    print("\n✓ All prerequisites met!")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created workspace directory: {output_dir}")
    
    # Clone GSO-Data-Utils
    if not args.skip_clone:
        utils_dir = clone_gso_utils(output_dir)
        if utils_dir is None:
            print("\nError: Failed to clone GSO-Data-Utils")
            print("You can manually clone it:")
            print(f"  cd {output_dir}")
            print("  git clone https://github.com/TO-Hitori/GSO-Data-Utils.git")
    
    # Create subdirectories
    for subdir in ['GSO_data', 'GSO_GLB', 'GSO_rendered']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Print instructions
    print_instructions(output_dir)


if __name__ == "__main__":
    main()
