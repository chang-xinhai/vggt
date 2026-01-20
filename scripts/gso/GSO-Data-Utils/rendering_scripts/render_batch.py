import subprocess
import os
import argparse

def run_blender_script(glb_path, out_path, script_path="render_cli.py"):
    """
    在命令行中运行 Blender 脚本。

    参数:
    - object_path: glb文件路径
    - output_path: 渲染图片的保存路径
    """
    # 构造命令
    command = ["blender", "--background", "--python", script_path, "--"]
    command += ["--glb_path", glb_path]
    command += ["--out_path", out_path]
    print("=-"*50)
    # 打印要执行的命令（可选）
    print("Executing command:", " ".join(command))
    # 执行命令
    try:
        # 使用 Popen 替代 run
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              encoding='utf-8') as process:
            # 读取输出
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print("Blender script executed successfully!")
                print("Blender Output:\n", stdout)
    except Exception as e:
        print("An error occurred while executing the Blender script:")
        print(e)
    
    print("=-"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a batch of GLB files.')
    parser.add_argument('--glb_folder', type=str, default=r"E:\Dataset\GSO\selected_GLB", help='The folder of the GLB files.')
    parser.add_argument('--out_path', type=str, default=r"E:\Dataset\GSO\selected_renderning\hunyuan", help='The path to save the rendered images.')
    args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    
    for glb_file in os.listdir(args.glb_folder):
        glb_path = os.path.join(args.glb_folder, glb_file)
        run_blender_script(glb_path, args.out_path)
    print("=-"*50)
    print("=-"*50)
    print()
    print()
    print()
    print("Rendering completed!")
    print()
    print()
    print()
    print("=-"*50)
    print("=-"*50)