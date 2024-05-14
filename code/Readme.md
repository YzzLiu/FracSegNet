# How to use FracSegNet?

FracSegNet is a specialized neural network model tailored for fracture segmentation in medical imaging. This project adapts and extends the nn-UNet framework to enhance fracture segmentation performance with advanced features such as distance map weight calculations and optimized 3D UNet architectures.

## Table of Contents
  - [Installation](#installation)
  - [Usage](#usage)
    * [Experiment Planning and Preprocessing](#experiment-planning-and-preprocessing)
    * [Model Training](#model-training)
    * [Run inference](#run-inference)
    * [Using our models for quick start](#using-our-models-for-quick-start)


## Installation

FracSegNet is developed and tested on Linux (Ubuntu 20.04) with a minimum GPU requirement of 10 GB VRAM.

1. Install PyTorch. Follow the installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally/):
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0
```
2. Install nn-UNet, which is used as a standardized baseline for 3D UNet and to quickly set up FracSegNet:
```bash
pip install nnunet==1.7.0
```
3. Overwrite the nn-UNet installation with the FracSegNet files:
```bash
cp -r ...Training/FracSegNet/* .../envs_name/python3.7/site-packages/nnunet/ 
```
## Usage
### Experiment Planning and Preprocessing
Create a new task for training FracSegNet. Before training, the dataset must be normalized (standardized, resampled, and distance map weights calculated). Execute the following command to prepare the dataset:
```bash
nnUNet_plan_and_preprocess -t TASK_ID --verify_dataset_integrity
```
### Anatomical Segmentation Model Training
On the basis of [CTpelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K), we continue to train the model on our PENGWIN data set to optimize the internal holes, lumbar sacralization and other problems.

### Fracture Segmentation Model Training
To train the Fracture Segmentation Model, use the command below:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TASK_ID_taskName FOLD
```
### Run inference
To infer the result of fracture segmentation from extracted CT region, use the command below:
```bash
nnUNet_predict -m 3d_fullres -t TASK_ID -f FOLD -i INPUT_FOLDER -o OUTPUT_FOLDER
```
## Using our models to extract fragments from CT images:
1. Download [Anatomical Segmentation and fracture Segmentation Models](tmp_url).
2. Copy Anatomical Segmentation and fracture Segmentation Models into NNUNET_RESULTS_FOLDER.
3. Execute the anatomical segmentation command:
```bashe
nnUNet_predict -m 3d_cascade_fullres --disable_tta -t TASK_ID -f all -i INPUT_DIR -o OUTPUT_DIR
```
4. Extract CT Regions from anatomical segmentation result using [extract_ct_regions.py](https://github.com/YzzLiu/FracSegNet/tree/main/code/inference)
5. Execute the fracture segmentation command:
```bashe
nnUNet_predict -m 3d_fullres TASK_ID -f all -i INPUT_DIR -o OUTPUT_DIR
```
