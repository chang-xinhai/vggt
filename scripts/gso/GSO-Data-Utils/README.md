# GSO-Data-Utils
[![Dataset](https://img.shields.io/badge/GSO%20Dataset-Official%20Website-blue)](https://goo.gle/scanned-objects)     [![ArXiv](https://img.shields.io/badge/ArXiv-2204.11918-red?logo=arxiv&logoColor=fff)](https://arxiv.org/abs/2204.11918)     [![Google Drive](https://img.shields.io/badge/Google%20Drive-Download%20Data-4285F4?logo=googledrive&logoColor=fff)](https://drive.google.com/drive/folders/1Dtqiyt0QP9dabiaTN5qONdb8avc0aNg6?usp=sharing)

A toolkit for working with the GSO dataset.

## Repository Features

This repository provides a set of tools and scripts to help you work with the GSO dataset. The main features include:

- **Downloading the Dataset**: Scripts to download the GSO dataset from the official source.
- **Processing the Dataset**: Tools to preprocess the data, specifically converting files to the `.glb` format.
- **Rendering the Dataset**: Scripts to visualize the data, generating charts, images, or other visualizations.



## Getting Started

**1. Clone the Repository**:

```bash
git clone https://github.com/yourusername/GSO-Dataset-Helper.git
cd GSO-Dataset-Helper
```

**2. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**3. Install Blender 4.X and Set Up Environment Variables:**

- Download and install Blender from the [official Blender website](https://docs.blender.org/manual/zh-hans/dev/getting_started/installing/index.html).

- To launch Blender from the command line, follow the [official tutorial](https://docs.blender.org/manual/zh-hans/2.80/advanced/command_line/launch/index.html) to add Blender to the system environment variables.

- Verify that Blender is successfully installed

  ```bash
  blender --version
  ```

  

## Downloading the Dataset

#### Option 1: Using the Download Script (Modified Version)

This option uses a modified version of the official download script to download the dataset from the official source.

```bash
python download_collection.py -d "./" 
```

If the download process is interrupted due to network issues, you can resume the download by specifying the last successfully downloaded data ID.

```bash
python download_collection.py -d "./" -c [data ID]
```

#### Option 2: Using the Provided Google Drive Link:

If the above method encounters network issues, you can obtain the data from my shared [Google Drive](https://drive.google.com/drive/folders/1Dtqiyt0QP9dabiaTN5qONdb8avc0aNg6?usp=sharing).



## Processing the Dataset

Before processing the dataset, first unzip each downloaded .zip file into separate folders. 

The file structure for each 3D object is as follows:

```bash
├─metadata.pbtxt  Detailed Metadata of 3D Models      
├─model.config    A metadata configuration file that defines the basic information 
|                 and license of the model.
├─model.sdf       The model.sdf file defines the simulation properties and structure of a 3D object,
|                 enabling its loading and interaction in simulation platforms like Ignition Gazebo. 
|
│
├─materials
│  └─textures
│      └─texture.png        Texture images used for rendering
├─meshes
│  ├─model.mtl              Material properties of the geometric structure
│  └─model.obj              The geometric structure of the model
└─thumbnails
    └─0.jpg ... 4.jpg       Multiple perspective renderings used for quick content previews

```

To extract a `.glb` file from an `.obj` file and its texture image(s), run:

```bash
python obj2glb_batch.py --input_path "GSO_data" --output_path "GSO_GLB"
```

You’ll find the converted `.glb` files in the `GSO_GLB` folder.



## Rendering the Dataset

#### Render a single GLB file.

```bash
cd ./rendering_scripts
blender --background --python render_cli.py -- --glb_path "/root/autodl-tmp/vggt/GSO-Data-Utils/test/GSO_GLB/5_HTP.glb" --out_path "/root/autodl-tmp/vggt/GSO-Data-Utils/test/output"
```

#### Batch render GLB files.

```bash
python render_batch.py --glb_folder "path/to/glb_folder" --out_path "path/to/out_folder"
```












































