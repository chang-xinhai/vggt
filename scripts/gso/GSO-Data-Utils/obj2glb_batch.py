import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import subprocess
import argparse
import re


"""
直接运行的python脚本
python obj2glb_batch.py
"""


# 过滤掉 Blender/导入器常见的告警行
_WARN_LINE = re.compile(r'^\s*(WARN\b|\[?WARN\]?|Warning\b|警告\b|\(warning\))', re.IGNORECASE)

def strip_warnings(text: str) -> str:
    if not text:
        return text
    return "\n".join(
        line for line in text.splitlines()
        if not _WARN_LINE.match(line)
    )



def run_blender_script(input_path, output_path, show_stdout, script_path="obj2glb_cli.py"):
    """
    在命令行中运行 Blender 脚本。

    参数:
    - object_path: glb文件路径
    - output_path: 渲染图片的保存路径
    """
    # 构造命令
    command = ["blender", "--background", "--python", script_path, "--"]
    command += ["--input_path", input_path]
    command += ["--output_path", output_path]
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
                if show_stdout:
                    clean_out = strip_warnings(stdout)
                    if clean_out.strip():  # 只在还有内容时打印
                        print("Blender Output:\n", clean_out)
                else:
                    print("Error executing Blender script:")
                    print("Return code:", process.returncode)
                    # 错误信息里也去掉告警，只保留真正的报错
                    clean_err = strip_warnings(stderr)
                    print("Error message:\n", clean_err)
    except Exception as e:
        print("An error occurred while executing the Blender script:")
        print(e)
    
    print("=-"*50)

'''
批量转换文件夹下的所有obj文件为glb文件
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert OBJ to GLB.')
    parser.add_argument('-i', '--input_path', type=str, default="GSO_data/GSO" , help='The path to the input OBJ file.')
    parser.add_argument('-o', '--output_path', type=str, default="GSO_GLB", help='The path to the output GLB file.')
    parser.add_argument('-s', '--show_stdout', type=bool, default=True, help='是否显示Blender的输出')
    args = parser.parse_args()
    
    
    # GSO数据集的主路径
    GSO_folder = args.input_path
    # 输出glb的主路径
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    for gso_file in os.listdir(GSO_folder):
        input_path = os.path.join(GSO_folder, gso_file)
        run_blender_script(input_path, output_path, args.show_stdout)
    