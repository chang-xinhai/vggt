import argparse
import shutil
from typing import List, Generator, Tuple, Optional
import sys
import bpy
import numpy as np
import math
from mathutils import Vector
import os
import json
import yaml




"""
--------------------------------------------------------命令行参数定义--------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# 路径配置
parser.add_argument("--glb_path", type=str, default=r"D:\MyProject\GSO-Data-Utils\GSO_GLB1\2_of_Jenga_Classic_Game.glb", help="要渲染的glb文件")
parser.add_argument("--out_path", type=str, default=r"D:\MyProject\GSO-Data-Utils\MV_IMG", help="渲染图片的保存路径，渲染图片保存在该文件夹的子文件夹中")

# 相机配置
parser.add_argument("--distances",    type=float, default=2.0, help="相机距离",)
parser.add_argument("--resolution_x", type=int,   default=512, help="水平分辨率")
parser.add_argument("--resolution_y", type=int,   default=512, help="垂直分辨率")
parser.add_argument("--resolution_percentage", type=int, default=100, help="渲染比例")
parser.add_argument("--azi_list", type=List[float], help="相机方位角(角度)", default=[30,  90, 150, 210, 270, 330])
parser.add_argument("--ele_list", type=List[float], help="相机仰角(角度)",   default=[ 0,   0,   0,   0,   0,   0])

# 渲染配置
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)






"""
--------------------------------------------------------相机、渲染器与 Cycles 设定--------------------------------------------------------
"""
context = bpy.context   # 获取当前 Blender 环境的上下文。
scene = context.scene   # 获取当前的场景对象（scene）表示 Blender 中的活动场景。
render = scene.render   # 获取当前场景的渲染设置（即 RenderSettings 对象）


# -------------------------------相机设置--------------------------------
cam = scene.objects["Camera"]  # 获取场景中的相机对象
cam.location = (0, 1.2, 0)     # 设置相机的位置
cam.data.lens = 35             # 设置相机的镜头焦距 （mm）
cam.data.sensor_width = 32     # 设置相机的传感器宽度（mm）

cam_constraint = cam.constraints.new(type="TRACK_TO")  # 为相机添加一个新的约束类型，TRACK_TO 约束使得相机始终指向一个目标（通常是空对象）。
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"         # 相机的负Z轴将始终指向目标
cam_constraint.up_axis = "UP_Y"                        # 相机的Y轴将始终指向目标

# 渲染器设置
render.engine = "CYCLES"                   # 设置渲染引擎
render.image_settings.file_format = "PNG"  # 设置渲染输出的文件格式为 PNG
render.image_settings.color_mode = "RGBA"  # 设置渲染图像的颜色模式为 RGBA

render.resolution_x = args.resolution_x    # 水平分辨率
render.resolution_y = args.resolution_y    # 垂直分辨率
render.resolution_percentage = args.resolution_percentage  # 渲染比例

# Cycles 渲染器配置
scene.cycles.device = "GPU"               # 设置渲染设备为 GPU
scene.cycles.samples = 128                # 设置 Cycles 渲染器的采样次数为 128
scene.cycles.tile_size = 8192             # 设置 Cycles 渲染器的瓦片大小为 8192
scene.cycles.diffuse_bounces = 1          # 设置漫反射反弹次数为 1
scene.cycles.glossy_bounces = 1           # 设置镜面反射反弹次数为 1
scene.cycles.transparent_max_bounces = 3  # 设置透明对象的最大反弹次数为 3
scene.cycles.transmission_bounces = 3     # 设置透射反弹次数为 3
scene.cycles.filter_width = 0.01          # 设置滤波宽度为 0.01
scene.cycles.use_denoising = True         # 启用去噪功能
scene.render.film_transparent = True      # 设置渲染输出图像的背景为透明

# 上下文设置
# 获取当前 Cycles 渲染器支持的所有设备
context.preferences.addons["cycles"].preferences.get_devices()
# 设置渲染设备类型
context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" 





"""
--------------------------------------------------------工具函数--------------------------------------------------------
"""
# 1.清空场景
def reset_scene() -> None:
    """
    用于清理Blender场景，仅保留相机和灯光，清空其它对象与资源，避免脏数据影响。
    在开始新的渲染任务或场景前使用，确保场景中没有不必要的干扰物体或资源。
    """
    # 删除非相机和灯光的对象
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # 删除所有材质
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # 删除所有纹理
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # 删除所有图像
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

# 2.加载模型
def load_object(glb_path: str) -> None:
    """
    将 glb 文件加载到Blender场景中。
    支持文件格式：.glb 和 .fbx
    """
    if glb_path.endswith(".glb"):
        # merge_vertices=True：在导入时合并顶点，这有助于减少冗余数据并优化模型。
        bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=True)
    elif glb_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=glb_path)
    else:
        raise ValueError(f"Unsupported file type: {glb_path}")

# 3.1 获取场景中所有网格对象
def scene_meshes() -> Generator[bpy.types.Object, None, None]:
    # 遍历场景中的所有对象
    for obj in bpy.context.scene.objects.values(): 
        # 只处理网格类型
        if isinstance(obj.data, (bpy.types.Mesh)):  
            yield obj  

# 3.2 获取场景中所有根对象
def scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

# 3.3 获取场景中所有对象的包围盒
def scene_bbox(single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    """
    计算场景中一个或多个对象的包围盒（Bounding Box）。
    返回包围盒的最小点和最大点。

    :param single_obj: 如果指定，则仅计算该单个对象的包围盒；否则，计算场景中所有网格对象的包围盒。
    :param ignore_matrix: 如果为 True，则忽略物体的世界矩阵，计算其局部坐标系下的包围盒。
    :return: 返回两个 Vector 对象，分别表示包围盒的最小点和最大点坐标。
    """
    # 初始化包围盒的最小点和最大点
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3

    # 初始化一个标志，用于检查是否找到任何对象
    found = False
    # 遍历场景中所有网格对象
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        # 如果找到对象，则将标志设置为 True
        found = True
        # 遍历当前对象的边界框顶点（包围盒的 8 个角点）
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix: 
                # 将顶点坐标从局部坐标系转换为世界坐标系。
                coord = obj.matrix_world @ coord
            # 更新包围盒的最小和最大点
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found: 
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

# 3.归一化到单位盒并居中
def normalize_scene() -> None:
    """
    将场景中的所有对象进行归一化处理，使得场景的大小和位置适配到一个标准范围内。
    """
    # 获取场景中所有对象的包围盒
    bbox_min, bbox_max = scene_bbox()
    
    # 计算包围盒的最大尺寸 
    # max(bbox_max - bbox_min)：包围盒在各个轴向的最大距离。
    scale = 1 / max(bbox_max - bbox_min)

    # 遍历所有根对象，并根据缩放因子调整其缩放比例
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # 更新视图层
    bpy.context.view_layer.update()

    # 重新计算场景的包围盒
    bbox_min, bbox_max = scene_bbox()
    # 计算场景中心的偏移量，使得场景的中心移动到原点。
    offset = -(bbox_min + bbox_max) / 2
    # 遍历所有根对象，并根据偏移量调整其位置
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    # 取消场景中所有对象的选择，避免后续操作受到影响。
    bpy.ops.object.select_all(action="DESELECT")


"""
--------------------------------------------------------坐标处理函数--------------------------------------------------------
"""

# 角度 → 单位球坐标
def az_el_to_points(azimuths: np.ndarray, elevations: np.ndarray) -> np.ndarray:
    """
    将方位角（azimuth）和仰角（elevation）转换为三维空间中位于单位球上的点,
    """
    x = np.cos(azimuths) * np.cos(elevations)
    y = np.sin(azimuths) * np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x, y, z], -1)  


# 放置相机到给定坐标，返回相机对象
def set_camera_location(cam_pt: np.ndarray) -> bpy.types.Object:
    # 解包坐标点
    x, y, z = cam_pt  # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    # 获取相机对象
    camera = bpy.data.objects["Camera"]
    # 设置相机的位置
    camera.location = x, y, z
    return camera


# 获取相机的内参矩阵
def get_calibration_matrix_K_from_blender(camera: bpy.types.Object) -> np.ndarray:
    # 获取相机的焦距，单位是毫米（mm）
    f_in_mm = camera.data.lens 
    
    # 获取场景分辨率和缩放比例  
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100

    # 获取相机的传感器尺寸和像素宽高比
    sensor_width_in_mm = camera.data.sensor_width   # 宽度
    sensor_height_in_mm = camera.data.sensor_height # 高度
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    
    # 根据相机传感器的适配方式计算 s_u 和 s_v
    # s_u 表示在水平方向上，每毫米的像素数量。s_v 表示在垂直方向上，每毫米的像素数量。
    if camera.data.sensor_fit == 'VERTICAL':
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: 
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    alpha_u = f_in_mm * s_u # 计算水平焦距（单位：像素）
    alpha_v = f_in_mm * s_v # 计算垂直焦距（单位：像素）
    u_0 = resolution_x_in_px * scale / 2 # 中心点水平坐标（单位：像素）
    v_0 = resolution_y_in_px * scale / 2 # 中心点垂直坐标（单位：像素）
    skew = 0  # 表示像素的斜率，通常为 0，表示像素是矩形的。
    # 构建内参矩阵
    K = np.asarray(((alpha_u,    skew, u_0),
                    (      0, alpha_v, v_0),
                    (      0,       0,   1)), np.float32)
    return K


# 获取相机的外参矩阵
def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> np.ndarray:
    # 更新当前视图层,确保变换矩阵是最新的。
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    # cam.matrix_world: 获取相机在世界坐标系中的变换矩阵。
    # decompose()：将 4x4 变换矩阵分解为位置 (location)、旋转 (rotation) 和缩放 (scale)。
    
    # 将旋转转换为 3x3 矩阵 R
    R = np.asarray(rotation.to_matrix())
    # 获取平移向量
    t = np.asarray(location)
    # 转换矩阵,用于将 Blender 的坐标系转换为计算机视觉中的坐标系
    cam_rec = np.asarray([[1,  0,  0], 
                          [0, -1,  0], 
                          [0,  0, -1]], np.float32)
    
    
    # 转置矩阵，R 会变成从世界坐标系到相机坐标系的变换
    R = R.T
    # 从 Blender 坐标系转换到计算机视觉坐标系
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    # 拼接旋转矩阵和位移向量，得到 3x4 外参矩阵
    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)
    return RT


"""
--------------------------------------------------------渲染主函数--------------------------------------------------------
"""


def randerning_and_save_images(glb_path: str, out_path: str) -> None:
    # 从文件名中提取对象的唯一标识符，作为输出文件夹名
    glb_name = os.path.basename(glb_path).split(".")[0]
    image_path = os.path.join(out_path, glb_name)
    os.makedirs(image_path, exist_ok=True)
    
    # 1. 清理Blender场景，
    reset_scene()
    # 2. 加载3D模型文件。
    load_object(glb_path)
    # 3. 场景中的所有对象进行归一化处理
    normalize_scene()

    # 创建一个空对象（Empty），作为相机的跟踪目标。
    empty = bpy.data.objects.new("Empty", None)
    # 将空对象链接到场景。
    scene.collection.objects.link(empty)
    # 设置相机的跟踪约束目标为这个空对象。使得相机始终指向该空对象。
    cam_constraint.target = empty

    # 获取场景的世界节点树。
    world_tree = bpy.context.scene.world.node_tree
    # 获取背景节点
    back_node = world_tree.nodes['Background']
    # 设置环境光的颜色强度为 0.5（灰色）。
    env_light = 0.5
    # 设置环境光的颜色为灰色（0.5, 0.5, 0.5），最后一个值为透明度（0.0 表示透明）。
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 0.0])
    # 设置环境光的强度为 1.0。
    back_node.inputs['Strength'].default_value = 1.0


    assert len(args.azi_list) == len(args.ele_list), "The lengths of azi_list and ele_list must be the same."
    num_images = len(args.azi_list)
    distances = np.asarray([args.distances for _ in range(num_images)])

    # 用于修正视角的偏差
    azimuths_bias = np.deg2rad(np.array([90] * num_images, dtype=np.float32))
    azimuths_ori = np.deg2rad(np.array(args.azi_list, dtype=np.float32))
    azimuths = azimuths_ori - azimuths_bias
    elevations = np.deg2rad(np.array(args.ele_list, dtype=np.float32))

    # 根据方位角、仰角和距离计算 各个视角的相机位置坐标
    cam_pts = az_el_to_points(azimuths, elevations) * distances[:, None]
    # 记录相机矩阵
    cam_poses = []
    # 创建图片输出的文件夹
    for i in range(num_images):
        # 设置相机的位置。
        camera = set_camera_location(cam_pts[i])
        # 获取相机的3x4齐次变换矩阵。
        RT = get_3x4_RT_matrix_from_blender(camera)
        # 收集相机矩阵
        cam_poses.append(RT)

        # 生成当前渲染图像的输出路径
        render_path = os.path.join(args.out_path, glb_name, f"{i:03d}_azi{args.azi_list[i]}_ele{args.ele_list[i]}.png")
        # 路径存在则删除
        if os.path.exists(render_path):
            shutil.rmtree(render_path)
        # 将相对路径转换为绝对路径
        scene.render.filepath = os.path.abspath(render_path)
        # 渲染当前场景并保存为单张静态图像
        bpy.ops.render.render(write_still=True)

    """
    参数保存
    """
    # 获取相机的内参矩阵。
    K = get_calibration_matrix_K_from_blender(camera)
    # 将所有相机的外参矩阵堆叠成一个数组
    cam_poses = np.stack(cam_poses, 0)
    #  保存相机参数
    Camera_parameters_dict = {
        "intrinsic": K.tolist(),
        "azimuths": np.degrees(azimuths_ori).tolist(),
        "elevations": np.degrees(elevations).tolist(),
        "distances": distances.tolist(),
        "extrinsic": cam_poses.tolist()
    }
    # 保存字典为 JSON 文件
    json_path = os.path.join(args.out_path, glb_name, "data.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(Camera_parameters_dict, file, ensure_ascii=False, indent=4)

    yaml_path = os.path.join(args.out_path, glb_name, "data.yaml")
    with open(yaml_path, "w") as file:
        yaml.dump(Camera_parameters_dict, file, default_flow_style=False, indent=4, sort_keys=False)



"""
blender --background --python render_cli.py -- --glb_path "D:\MyProject\GSO-Data-Utils\GSO_GLB1\50_BLOCKS.glb" --out_path "D:\MyProject\GSO-Data-Utils\MV_IMG"
"""

if __name__ == "__main__":
    '''渲染单张图片'''
    randerning_and_save_images(args.glb_path, args.out_path)