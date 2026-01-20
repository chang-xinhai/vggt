import os, shutil, tqdm


render_path = r"E:\Dataset\GSO\selected_renderning\zero123++"
img_path = r"E:\Dataset\GSO\selected_input\a10e10"


for glb_name in tqdm.tqdm(os.listdir(render_path)):
    from_path = os.path.join(render_path, glb_name, "007_azi10_ele10.png")
    to_path = os.path.join(img_path, glb_name+".png")
    shutil.move(from_path, to_path)



'''
from： 

E:\Dataset\GSO\selected_renderning\zero123++\glbname\006_azi5_ele5.png

to:

E:\Dataset\GSO\selected_input\a5e5\glbname.png


from： 

E:\Dataset\GSO\selected_renderning\zero123++\glbname\007_azi10_ele10.png

to:

E:\Dataset\GSO\selected_input\a10e10\glbname.png

'''