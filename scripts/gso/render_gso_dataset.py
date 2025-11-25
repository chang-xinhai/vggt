#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GSO Dataset Rendering Script

This script renders GSO objects following the Instant3D/GS-LRM/LVSM protocol:
- 64 views total: 4 elevations (0°, 20°, 40°, 60°) × 16 azimuths each
- Resolution: 512×512
- Structured for easy evaluation with eval_nvs_gso.py

Requirements:
    - Blender 4.x installed and available in PATH
    - GSO dataset downloaded and converted to .glb format

Usage:
    blender --background --python render_gso_dataset.py -- --glb_path /path/to/object.glb --out_path /path/to/output
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Generator, Optional

try:
    import bpy
    import numpy as np
    from mathutils import Vector
except ImportError:
    print("Error: This script must be run with Blender's Python interpreter")
    print("Usage: blender --background --python render_gso_dataset.py -- [args]")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Render GSO dataset following Instant3D protocol'
    )
    
    # Path configuration
    parser.add_argument(
        '--glb_path', 
        type=str, 
        required=True,
        help='Path to the GLB file to render'
    )
    parser.add_argument(
        '--out_path', 
        type=str, 
        required=True,
        help='Output directory for rendered images'
    )
    
    # Camera configuration (following Instant3D)
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
        help='Image resolution (width and height, default: 512)'
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
    
    # Rendering configuration
    parser.add_argument(
        '--samples', 
        type=int, 
        default=128,
        help='Number of rendering samples (default: 128)'
    )
    parser.add_argument(
        '--use_gpu', 
        action='store_true',
        default=True,
        help='Use GPU for rendering (default: True)'
    )
    
    # Parse arguments (handle Blender's -- separator)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]
    
    return parser.parse_args(argv)


def reset_scene():
    """Reset the Blender scene, keeping only camera and lights."""
    # Delete all objects except camera and lights
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.select_by_type(type='CAMERA', extend=False)
    bpy.ops.object.select_by_type(type='LIGHT', extend=True)
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()
    
    # Clear unused data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    
    for texture in bpy.data.textures:
        if texture.users == 0:
            bpy.data.textures.remove(texture)
    
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)
    
    bpy.ops.object.select_all(action='DESELECT')


def load_object(glb_path: str):
    """Load a GLB file into the scene."""
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")
    
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=glb_path)


def scene_meshes() -> Generator["bpy.types.Object", None, None]:
    """Get all mesh objects in the scene."""
    for obj in bpy.context.scene.objects.values(): 
        if isinstance(obj.data, (bpy.types.Mesh)):  
            yield obj  


def scene_root_objects() -> Generator["bpy.types.Object", None, None]:
    """Get all root objects in the scene."""
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox(single_obj: Optional["bpy.types.Object"] = None, ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    """
    Compute the bounding box of the scene or a single object.
    Returns (min_point, max_point).
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    
    objects = [single_obj] if single_obj is not None else scene_meshes()
    
    for obj in objects:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix: 
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
            
    if not found: 
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    """
    Normalize the scene by centering and scaling all objects.
    Matches the logic from render_cli.py:
    - Scale so that the max dimension of the bounding box is 1.0
    - Center the object at (0,0,0)
    """
    # Get scene bounding box
    bbox_min, bbox_max = scene_bbox()
    
    # Calculate scale factor
    # render_cli.py uses 1 / max_dimension
    scale = 1 / max(bbox_max - bbox_min)

    # Apply scale to all root objects
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
        
    # Update view layer to apply transformations
    bpy.context.view_layer.update()

    # Recalculate bounding box after scaling
    bbox_min, bbox_max = scene_bbox()
    
    # Calculate offset to center the scene
    offset = -(bbox_min + bbox_max) / 2
    
    # Apply translation to all root objects
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
        
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera_and_rendering(resolution: int, samples: int, use_gpu: bool):
    """Setup camera and rendering settings."""
    scene = bpy.context.scene
    camera = scene.objects["Camera"]
    
    # Camera settings
    camera.data.lens = 35  # Focal length in mm
    camera.data.sensor_width = 32  # Sensor width in mm
    
    # Add track-to constraint
    camera.constraints.clear()
    cam_constraint = camera.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
    # Create empty object for camera to track
    empty = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)
    cam_constraint.target = empty
    
    # Rendering settings
    render = scene.render
    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    
    # Cycles settings
    scene.cycles.samples = samples
    scene.cycles.tile_size = 8192
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    
    # GPU settings
    if use_gpu:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    
    # Setup world environment (uniform lighting)
    world_tree = scene.world.node_tree
    world_tree.nodes.clear()
    
    # Add background node
    bg_node = world_tree.nodes.new('ShaderNodeBackground')
    output_node = world_tree.nodes.new('ShaderNodeOutputWorld')
    
    # Set uniform lighting
    env_light = 0.5
    bg_node.inputs['Color'].default_value = (env_light, env_light, env_light, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0
    
    # Link nodes
    world_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    return camera, empty


def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, distance: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        azimuth_deg: Azimuth angle in degrees (0° is +X axis, increases counter-clockwise)
        elevation_deg: Elevation angle in degrees (0° is XY plane, positive is up)
        distance: Distance from origin
    
    Returns:
        (x, y, z) coordinates
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    
    return (x, y, z)


def get_camera_intrinsics(camera: bpy.types.Object) -> np.ndarray:
    """
    Get camera intrinsic matrix (K).
    
    Returns:
        3x3 intrinsic matrix
    """
    scene = bpy.context.scene
    
    # Get camera parameters
    f_in_mm = camera.data.lens
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    
    # Calculate pixels per millimeter
    if camera.data.sensor_fit == 'VERTICAL':
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    
    # Calculate focal length in pixels
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    
    # Principal point (image center)
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    
    # Construct intrinsic matrix
    K = np.array([
        [alpha_u, 0, u_0],
        [0, alpha_v, v_0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def get_camera_extrinsics(camera: bpy.types.Object) -> np.ndarray:
    """
    Get camera extrinsic matrix (world to camera transformation).
    
    Returns:
        3x4 extrinsic matrix [R|t]
    """
    # Update the scene to ensure matrix_world is current
    bpy.context.view_layer.update()
    
    # Get camera's world transformation
    location, rotation = camera.matrix_world.decompose()[0:2]
    
    # Convert rotation to matrix
    R = np.array(rotation.to_matrix(), dtype=np.float32)
    t = np.array(location, dtype=np.float32)
    
    # Coordinate system conversion (Blender to OpenCV)
    cam_rec = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    
    # Transform to world-to-camera
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    
    # Construct 3x4 extrinsic matrix
    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], axis=1)
    
    return RT


def render_views(
    glb_path: str,
    out_path: str,
    elevations: List[float],
    num_azimuths: int,
    distance: float,
    resolution: int,
    samples: int,
    use_gpu: bool
):
    """
    Render all views of the GSO object.
    
    Args:
        glb_path: Path to GLB file
        out_path: Output directory
        elevations: List of elevation angles
        num_azimuths: Number of azimuth angles per elevation
        distance: Camera distance from object
        resolution: Image resolution
        samples: Number of rendering samples
        use_gpu: Whether to use GPU
    """
    # Get object name from filename
    obj_name = Path(glb_path).stem
    obj_out_path = Path(out_path) / obj_name
    img_out_path = obj_out_path / "images"
    img_out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {obj_name}...")
    
    # Reset and setup scene
    reset_scene()
    load_object(glb_path)
    normalize_scene()
    camera, empty = setup_camera_and_rendering(resolution, samples, use_gpu)
    
    # Generate azimuth angles (evenly distributed from 0° to 360°, excluding 360°)
    azimuths = [i * 360.0 / num_azimuths for i in range(num_azimuths)]
    
    # Prepare camera data structure
    cameras_data = {}
    view_id = 0
    
    # Render all views
    for elevation in elevations:
        for azimuth in azimuths:
            # Calculate camera position
            cam_pos = spherical_to_cartesian(azimuth, elevation, distance)
            camera.location = cam_pos
            
            # Get camera matrices
            K = get_camera_intrinsics(camera)
            RT = get_camera_extrinsics(camera)
            
            # Store camera parameters
            view_name = f"{view_id:03d}"
            cameras_data[view_name] = {
                "intrinsics": K.tolist(),
                "extrinsics": RT.tolist(),
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance
            }
            
            # Render and save image
            img_path = img_out_path / f"{view_name}.png"
            bpy.context.scene.render.filepath = str(img_path)
            bpy.ops.render.render(write_still=True)
            
            print(f"  Rendered view {view_id:03d}: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")
            view_id += 1
    
    # Save camera parameters
    cameras_file = obj_out_path / "cameras.json"
    with open(cameras_file, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    
    print(f"Saved {view_id} views to {obj_out_path}")
    print(f"Camera parameters saved to {cameras_file}")


def main():
    """Main rendering function."""
    args = parse_args()
    
    print("="*60)
    print("GSO Dataset Rendering Script")
    print("Following Instant3D/GS-LRM/LVSM Protocol")
    print("="*60)
    print(f"Input GLB: {args.glb_path}")
    print(f"Output directory: {args.out_path}")
    print(f"Elevations: {args.elevations}")
    print(f"Azimuths per elevation: {args.num_azimuths}")
    print(f"Total views: {len(args.elevations) * args.num_azimuths}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Camera distance: {args.distance}")
    print(f"Samples: {args.samples}")
    print(f"Use GPU: {args.use_gpu}")
    print("="*60)
    
    render_views(
        glb_path=args.glb_path,
        out_path=args.out_path,
        elevations=args.elevations,
        num_azimuths=args.num_azimuths,
        distance=args.distance,
        resolution=args.resolution,
        samples=args.samples,
        use_gpu=args.use_gpu
    )
    
    print("\nRendering complete!")


if __name__ == "__main__":
    main()
