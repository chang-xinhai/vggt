import os, shutil
import tqdm

if __name__ == "__main__":
    GSO_folder = r"E:\Dataset\GSO\GSO_data"
    output_path = r"E:\Dataset\GSO\prview_image0"
    os.makedirs(output_path, exist_ok=True)
    
    for gso_file in tqdm.tqdm(os.listdir(GSO_folder)):
        from_path = os.path.join(GSO_folder, gso_file, "thumbnails", "0.jpg")
        to_path = os.path.join(output_path, gso_file + ".jpg")
        
        shutil.copy(from_path, to_path)
    
    print("Done")