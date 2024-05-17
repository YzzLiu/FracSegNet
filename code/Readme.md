
# FracSegNet

FracSegNet is a specialized neural network model designed for fracture segmentation in medical imaging. This project adapts and extends the nn-UNet framework, incorporating advanced features like distance map weight calculations and optimized 3D UNet architectures to enhance segmentation performance.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [Experiment Planning and Preprocessing](#experiment-planning-and-preprocessing)
  * [Model Training](#model-training)
  * [Run Inference](#run-inference)
  * [Using Pre-trained Models to Extract Fragments from CT Images](#using-pre-trained-models-to-extract-fragments-from-ct-images)

## Installation

FracSegNet is developed and tested on Linux (Ubuntu 20.04) with a minimum GPU requirement of 10 GB VRAM.

1. **Install PyTorch**  
   Follow the installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally/):
   ```bash
   pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0
   ```
2. **Install nn-UNet**  
   Install nn-UNet, which serves as a standardized baseline for 3D UNet and facilitates the setup of FracSegNet:
   ```bash
   pip install nnunet==1.7.0
   ```
3. **Overwrite nn-UNet with FracSegNet Files**  
   Replace the nn-UNet installation with FracSegNet files:
   ```bash
   cp -r /path_to_FracSegNet/Training/FracSegNet/* /path_to_python_envs/env_name/python3.7/site-packages/nnunet/ 
   ```

## Usage

### Experiment Planning and Preprocessing
Before training, create a new task for FracSegNet and preprocess the dataset. This involves normalization (standardization, resampling) and distance map weight calculation. Execute the following command to prepare the dataset:
```bash
nnUNet_plan_and_preprocess -t TASK_ID --verify_dataset_integrity
```

### Model Training

#### Anatomical Segmentation Model Training
Based on the [CTpelvic1K dataset](https://github.com/MIRACLE-Center/CTPelvic1K), further train the model on the PENGWIN dataset to address internal holes, lumbar sacralization, and other issues.

#### Fracture Segmentation Model Training
To train the fracture segmentation model, use the command below:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TASK_ID_taskName FOLD
```

### Run Inference
To perform fracture segmentation on extracted CT regions, use the following command:
```bash
nnUNet_predict -m 3d_fullres -t TASK_ID -f FOLD -i INPUT_FOLDER -o OUTPUT_FOLDER
```

### Using Pre-trained Models to Extract Fragments from CT Images
1. **Download Models**  
   Download the [Anatomical Segmentation and Fracture Segmentation Models](https://github.com/YzzLiu/FracSegNet/tree/main/code/inference).
2. **Copy Models**  
   Copy the models into `NNUNET_RESULTS_FOLDER`.
3. **Anatomical Segmentation**  
   Execute the anatomical segmentation command:
   ```bash
   nnUNet_predict -m 3d_cascade_fullres --disable_tta -t TASK_ID -f all -i INPUT_DIR -o OUTPUT_DIR
   ```
4. **Extract CT Regions**  
   Extract CT regions from the anatomical segmentation results using `/inference/extract_ct_regions.py`:
   ```bash
   python /inference/extract_ct_regions.py -i INPUT_IMAGE -o OUTPUT_IMAGE
   ```
5. **Fracture Segmentation**  
   Perform fracture segmentation:
   ```bash
   nnUNet_predict -m 3d_fullres -t TASK_ID -f all -i INPUT_DIR -o OUTPUT_DIR
   ```
