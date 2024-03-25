
---

# Object Detection Pipeline

This repository provides scripts and notebooks to set up and train object detection models using Faster R-CNN (Detectron2) and YOLO.

## 1. Faster R-CNN (Detectron2)

### 1.1. Dataset Preparation

Download the datasets from CVAT in COCO 1.1 format for the following classes:
- Drinks
- Utensils
- Laptops

Instructions to download the dataset

Run `python download.py --download_data True`

The files will be named `instances_default.json` for each class. Change the categories as specified in the instructions and merge the datasets using the following commands:

```bash
python -m COCO_merger.merge --src "/path/to/drinks_instances_default.json" "/path/to/utensils_instances_default.json" --out "/path/to/output.json"
python -m COCO_merger.merge --src "/path/to/output.json" "/path/to/laptops_instances_default.json" --out "/path/to/output.json"
```


### 1.2. Annotations and Images

- Annotations: `COCOAnnotations`
- Images: `/MergedDataset`

### 1.3. Dataset Splitting

Split the dataset into train and test using `cocosplit.py`:

```bash
python cocosplit.py --having-annotations -s 0.8 COCOAnnotations/output.json COCOAnnotations/trainnew.json COCOAnnotations/testnew.json
```

### 1.4. Model Training

Refer to `Detectron2.ipynb` for training on a single dataset and `Detectron2Multiple.ipynb` for training on multiple classes.

## 2. YOLO

### 2.1. Dataset Creation and Training

Run the cells in the provided notebook to create the dataset from the Faster R-CNN output and train the YOLO model. This notebook includes a train/test/val split.

### 2.2. Configuration

Configure the dataset with the following details:
- Names: Nothing, Drinks, Utensils, Laptop
- Number of Classes: 4
- Paths: 
    - Training Images: `YOLO_dataset/images/train`
    - Validation Images: `YOLO_dataset/images/test`

## Installation Instructions

### Windows

```bash

# For all at once (WORKS!)
conda create --prefix "C:\\Users\\rs659\\Desktop\\Object-Detection\\wincondaprojenv" python=3.9
conda activate "C:\\Users\\rs659\\Desktop\\Object-Detection\\wincondaprojenv"
pip install gdown cython==3.0.9 numpy==1.23.5 ninja ultralytics==8.1.29 pillow pylabel==0.1.55 kiwisolver==1.4.5 pandas==2.2.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install git+https://github.com/facebookresearch/detectron2.git

# For yaml file
conda env create -f environment.yml
conda activate object-detection-env


# For custom installation
conda create --prefix "C:\\Users\\rs659\\Desktop\\Object-Detection\\wincondaprojenv" python=3.9
conda activate "C:\\Users\\rs659\\Desktop\\Object-Detection\\wincondaprojenv"
pip install cython==3.0.9
pip install numpy==1.23.5
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install ninja
pip install ultralytics==8.1.29
pip install pillow
pip install pylabel==0.1.55
pip install kiwisolver==1.4.5
pip install pandas==2.2.1
pip install git+https://github.com/facebookresearch/detectron2.git
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install opencv-python
```

### macOS

```bash
python3 -m venv proj2env
source proj2env/bin/activate
pip install --upgrade pip
pip install torch
pip install torchvision
pip install pybind11
brew install pybind11
```
