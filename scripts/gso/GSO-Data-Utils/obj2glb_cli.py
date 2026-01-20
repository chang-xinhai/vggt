
import bpy # Blender的Python API模块
import os
import sys
import math
import argparse

def clear_scene():
    """清除当前场景中的所有对象"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
def reset_rotations_to_identity():
    """将所有网格对象的欧拉旋转重置为 (0, 0, 0)。返回被重置的对象数量。"""
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        print("警告：未找到网格对象")
        return 0
    for o in meshes:
        o.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return len(meshes)

# def reset_rotations_to_identity():
    
#     rotated = False
#     for obj in bpy.context.scene.objects:
#         if obj.type == 'MESH':
#             obj.rotation_euler[0] = math.radians(0)  
#             obj.rotation_euler[1] = math.radians(0)  
#             obj.rotation_euler[2] = math.radians(0)  

#             rotated = True
#     if not rotated:
#         print("警告：未找到网格对象，无法旋转")
#     # 更新场景以确保变换生效
#     bpy.context.view_layer.update()

def set_texture(texture_path):
    """
    将场景中所有材质的 Image Texture 节点替换为指定纹理图片。

    参数:
    - texture_path: 纹理图片的路径（绝对或相对路径）
    """
    abs_tex = os.path.abspath(texture_path)
    if not os.path.exists(abs_tex):
        print(f"错误：指定纹理文件未找到：{abs_tex}")
        return

    try:
        # Blender 会自动复用已加载的同路径 Image 数据块
        tex_img = bpy.data.images.load(abs_tex, check_existing=True)
    except RuntimeError as e:
        print(f"错误：加载纹理失败：{abs_tex}，{e}")
        return

    for mat in bpy.data.materials:
        if not mat or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                node.image = tex_img
                node.image.filepath = abs_tex
                try:
                    node.image.reload()
                except Exception:
                    pass
                print(f"替换纹理 → 材质[{mat.name}] 节点[{node.name}] = {abs_tex}")


def convert_obj_to_glb(obj_path, tex_path, glb_path):
    """将 OBJ 文件转换为 GLB 格式"""
    obj_path = os.path.abspath(obj_path)
    glb_path = os.path.abspath(glb_path)

    if not os.path.exists(obj_path):
        print(f"错误：OBJ 文件未找到：{obj_path}")
        return False

    clear_scene()

    # 导入obj
    try:
        bpy.ops.wm.obj_import(filepath=obj_path, use_image_search=False)  # Blender 4.x
    except TypeError:
        # 有些版本没有该参数，就退化
        try:
            bpy.ops.wm.obj_import(filepath=obj_path)
        except Exception:
            # 兼容老导入器
            bpy.ops.import_scene.obj(filepath=obj_path, use_image_search=False)
    print(f"OBJ 导入成功：{obj_path}")

    set_texture(tex_path)

    num = reset_rotations_to_identity()
    print(f"重置旋转对象数量：{num}")
    # bpy.context.view_layer.update()

    try:
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            use_selection=False,  # 导出全部对象
            export_materials='EXPORT',
            export_yup=True,
            export_draco_mesh_compression_enable=False,
            export_image_format='AUTO'
        )
        print(f"转换完成：{glb_path}")
        return True
    except Exception as e:
        print(f"导出 GLB 文件时出错：{e}")
        return False
# 执行转换
'''
将gso数据集的单个样本obj文件转换为glb文件
# 通过blender在命令行中执行
blender --background --python obj2glb_cli.py    
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert OBJ to GLB.')
    parser.add_argument('-i', '--input_path', type=str, default="GSO_data/5_HTP", help='The path to the input OBJ folder.')
    parser.add_argument('-o', '--output_path', type=str, default="./GSO_GLB", help='The path to the output GLB folder.')
    # When running Blender with `--python script.py -- <args>`, arguments after `--` are passed to the script.
    # If the caller does not include `--` (or passes no extra args), `--` won't be in sys.argv.
    # Fall back to an empty list (use defaults) if `--` is not present.
    try:
        argv = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        argv = []
    # Helpful logging for debugging — this shows the CLI args that will be parsed.
    if argv:
        print(f"Received CLI args: {argv}")
    else:
        print("No extra CLI args found; using default values for input and output paths.")
    args = parser.parse_args(argv)
    
    os.makedirs(args.output_path, exist_ok=True)
    
  
    obj_file = os.path.join(args.input_path, "meshes", "model.obj")
    tex_file = os.path.join(args.input_path, "materials", "textures", "texture.png")
    glb_file = os.path.join(args.output_path, "{}.glb".format(os.path.basename(args.input_path)))
    
    if convert_obj_to_glb(obj_file, tex_file, glb_file):
        sys.exit(0)  # 成功退出
    else:
        sys.exit(1)  # 失败退出



