#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test to verify GSO rendering and evaluation logic.

This script tests:
1. View ID mapping logic
2. Input view selection (Instant3D protocol)
3. Camera parameter format
"""

import numpy as np
import json


def test_view_id_mapping():
    """Test view ID calculation."""
    print("Testing view ID mapping...")
    
    # Test parameters
    num_elevations = 4
    num_azimuths = 16
    elevations = [0, 20, 40, 60]
    
    # Calculate view IDs for input views (elevation 20°, azimuths [45°, 135°, 225°, 315°])
    elevation_idx = 1  # 20° is second elevation
    azimuth_indices = [2, 6, 10, 14]  # For azimuths [45°, 135°, 225°, 315°]
    
    expected_views = [18, 22, 26, 30]
    calculated_views = [elevation_idx * num_azimuths + az_idx for az_idx in azimuth_indices]
    
    assert calculated_views == expected_views, f"View ID mismatch: {calculated_views} != {expected_views}"
    print(f"  ✓ Input view IDs: {calculated_views}")
    
    # Verify azimuth angles
    azimuth_step = 360.0 / num_azimuths
    calculated_azimuths = [az_idx * azimuth_step for az_idx in azimuth_indices]
    expected_azimuths = [45.0, 135.0, 225.0, 315.0]
    
    assert calculated_azimuths == expected_azimuths, f"Azimuth mismatch: {calculated_azimuths} != {expected_azimuths}"
    print(f"  ✓ Input azimuths: {calculated_azimuths}")
    
    # Test total views
    total_views = num_elevations * num_azimuths
    assert total_views == 64, f"Total views should be 64, got {total_views}"
    print(f"  ✓ Total views: {total_views}")
    
    print("View ID mapping test passed!\n")


def test_camera_format():
    """Test camera parameter format."""
    print("Testing camera parameter format...")
    
    # Example intrinsic matrix
    fx, fy = 576.0, 576.0
    cx, cy = 256.0, 256.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    assert K.shape == (3, 3), f"Intrinsics should be 3x3, got {K.shape}"
    print(f"  ✓ Intrinsics shape: {K.shape}")
    
    # Example extrinsic matrix (identity camera at origin)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0, 0, 2.0], dtype=np.float32)
    RT = np.concatenate([R, t[:, None]], axis=1)
    
    assert RT.shape == (3, 4), f"Extrinsics should be 3x4, got {RT.shape}"
    print(f"  ✓ Extrinsics shape: {RT.shape}")
    
    # Test JSON serialization
    camera_data = {
        "intrinsics": K.tolist(),
        "extrinsics": RT.tolist(),
        "azimuth": 0.0,
        "elevation": 0.0,
        "distance": 2.0
    }
    
    json_str = json.dumps(camera_data)
    loaded = json.loads(json_str)
    
    assert np.allclose(loaded["intrinsics"], K), "Intrinsics serialization failed"
    assert np.allclose(loaded["extrinsics"], RT), "Extrinsics serialization failed"
    print("  ✓ JSON serialization works")
    
    print("Camera format test passed!\n")


def test_input_view_selection():
    """Test input view selection logic."""
    print("Testing input view selection...")
    
    # Simulate 64 views
    all_views = [f"{i:03d}" for i in range(64)]
    
    # Input views (following Instant3D protocol)
    input_view_indices = [18, 22, 26, 30]
    input_view_ids = [f"{i:03d}" for i in input_view_indices]
    
    print(f"  Input views: {input_view_ids}")
    
    # Verify all input views exist
    for vid in input_view_ids:
        assert vid in all_views, f"Input view {vid} not in available views"
    print("  ✓ All input views exist")
    
    # Get remaining views for target selection
    remaining_views = [v for v in all_views if v not in input_view_ids]
    assert len(remaining_views) == 60, f"Should have 60 remaining views, got {len(remaining_views)}"
    print(f"  ✓ Remaining views: {len(remaining_views)}")
    
    # Simulate target view sampling
    np.random.seed(42)
    num_target_views = 10
    target_view_ids = list(np.random.choice(remaining_views, size=num_target_views, replace=False))
    
    print(f"  Target views (first 5): {target_view_ids[:5]}")
    
    # Verify no overlap
    for vid in target_view_ids:
        assert vid not in input_view_ids, f"Target view {vid} overlaps with input views"
    print("  ✓ No overlap between input and target views")
    
    print("Input view selection test passed!\n")


def test_spherical_to_cartesian():
    """Test spherical to cartesian conversion."""
    print("Testing spherical to cartesian conversion...")
    
    import math
    
    def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        
        x = distance * math.cos(elevation) * math.cos(azimuth)
        y = distance * math.cos(elevation) * math.sin(azimuth)
        z = distance * math.sin(elevation)
        
        return (x, y, z)
    
    # Test known cases
    # Azimuth 0°, Elevation 0° -> (distance, 0, 0)
    x, y, z = spherical_to_cartesian(0, 0, 2.0)
    assert np.isclose(x, 2.0) and np.isclose(y, 0.0) and np.isclose(z, 0.0), \
        f"Failed for (0°, 0°): got ({x}, {y}, {z})"
    print(f"  ✓ (0°, 0°, 2.0) -> ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Azimuth 90°, Elevation 0° -> (0, distance, 0)
    x, y, z = spherical_to_cartesian(90, 0, 2.0)
    assert np.isclose(x, 0.0) and np.isclose(y, 2.0) and np.isclose(z, 0.0), \
        f"Failed for (90°, 0°): got ({x}, {y}, {z})"
    print(f"  ✓ (90°, 0°, 2.0) -> ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Azimuth 0°, Elevation 90° -> (0, 0, distance)
    x, y, z = spherical_to_cartesian(0, 90, 2.0)
    assert np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, 2.0), \
        f"Failed for (0°, 90°): got ({x}, {y}, {z})"
    print(f"  ✓ (0°, 90°, 2.0) -> ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Verify distance is preserved
    for azimuth in [0, 45, 90, 135, 180, 225, 270, 315]:
        for elevation in [0, 20, 40, 60]:
            x, y, z = spherical_to_cartesian(azimuth, elevation, 2.0)
            distance = math.sqrt(x*x + y*y + z*z)
            assert np.isclose(distance, 2.0), \
                f"Distance not preserved for ({azimuth}°, {elevation}°): {distance}"
    
    print("  ✓ Distance preserved for all test angles")
    print("Spherical to cartesian test passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("GSO Dataset Rendering Logic Tests")
    print("="*60)
    print()
    
    try:
        test_view_id_mapping()
        test_camera_format()
        test_input_view_selection()
        test_spherical_to_cartesian()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
