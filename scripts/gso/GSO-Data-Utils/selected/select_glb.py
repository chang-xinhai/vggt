import os
import shutil
import tqdm


selected_img_path = r"E:\Dataset\GSO\selected_preview"
selected_glb_path = r"E:\Dataset\GSO\selected_GLB"
GLB_path = r"E:\Dataset\GSO\GSO_GLB"

os.makedirs(selected_glb_path, exist_ok=True)

for img in tqdm.tqdm(os.listdir(selected_img_path)):
    name = img.split(".")[0]
    from_path = os.path.join(GLB_path, name + ".glb")
    to_path = os.path.join(selected_glb_path, name + ".glb")
    shutil.copy(from_path, to_path)

print("Done")
    