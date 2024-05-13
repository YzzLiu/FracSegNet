# DataSet for Pelvic Fracture Segmentation

Pelvic fractures are critical high-energy injuries. Manual delineation of the fracture surface from 3D CT images can be time-consuming and prone to errors. This dataset aims to facilitate the development of automated algorithms, especially for pelvic research and reduction planning.

## Abstract

Pelvic fracture is a severe type of high-energy injury. The segmentation of pelvic fractures from 3D CT images plays a pivotal role in trauma diagnosis, evaluation, and subsequent treatment planning. While manual delineation can be conducted in a slice-by-slice manner, it is inherently slow and often fraught with errors. Automatic segmentation remains a challenge due to the intricate structure of pelvic bones and the vast variability in fracture types and shapes. 

In this study, we introduced a novel deep-learning technique for the automatic segmentation of pelvic fractures. Our methodology comprises two sequential networks. Initially, the anatomical segmentation network sifts through the CT scans to extract the left and right ilia and sacrum. Subsequently, the fracture segmentation network isolates the fragments within each masked bone region. We integrated a distance-weighted loss into a 3D U-net to enhance accuracy near the fracture vicinity. Moreover, we utilized multi-scale deep supervision and employed a smooth transition strategy for effective training.

We curated a dataset featuring 100 CT scans with fractured pelvises, each manually annotated for fractures. Through a five-fold cross-validation experiment, our method surpassed max-flow segmentation and networks devoid of distance weighting. The results exhibited a global Dice of 99.38%, a local Dice of 93.79%, and an Hausdorff distance of 17.12 mm.

In our quest to advance pelvic research, especially in reduction planning, we have made our dataset and corresponding source code accessible to the public.

## Dataset Description

We collated a dataset encompassing 100 preoperative CT scans, representing an array of prevalent pelvic fractures. These scans have been sourced from a pool of 100 patients (ranging between 18-74 years of age, including 41 females) slated for pelvic reduction surgery at Beijing Jishuitan Hospital between the years 2018-2022. All CT scans were captured utilizing a Toshiba Aquilion scanner.

## Dataset Structure and Contents

The dataset is organized in a structured folder system for easy navigation and utilization. Here's an overview:
```
DataSet for Pelvic Fracture Segmentation
|
|-- 001
|   |-- 001_SA.nii.gz
|   |-- 001_SA_mask.nii.gz
|   |-- 001_RI.nii.gz
|   |-- 001_RI_mask.nii.gz
|   |-- 001_LI.nii.gz
|   |-- 001_LI_mask.nii.gz
|
|-- 002
|   |-- 002_SA.nii.gz
|   |-- 002_SA_mask.nii.gz
|   |-- 002_RI.nii.gz
|   |-- 002_RI_mask.nii.gz
|   |-- 002_LI.nii.gz
|   |-- 002_LI_mask.nii.gz
|
|-- ... (other directories and files)
```

**File Descriptions:**
- `XXX_SA.nii.gz`: Represents the segmented anatomical structure of the sacrum area from the CT scan.
- `XXX_SA_mask.nii.gz`: Represents the segmented fracture in the sacrum area.
- `XXX_RI.nii.gz`: Represents the segmented anatomical structure of the right ilium from the CT scan.
- `XXX_RI_mask.nii.gz`: Represents the segmented fracture in the right ilium.
- `XXX_LI.nii.gz`: Represents the segmented anatomical structure of the left ilium from the CT scan.
- `XXX_LI_mask.nii.gz`: Represents the segmented fracture in the left ilium.

`XXX` denotes the patient ID.

## Usage

1. Download the dataset from [DataSet of Pelvic Frcature Segmenation(Google Dirve)](https://drive.google.com/file/d/18xAU3-VJdx1QRP2W2eOqE3NCkbSPP2XG/view?usp=sharing).
2. Implement [fracture segmentation algorithms]().
3. Use our dataset for training/testing purposes, respecting the terms of the license.

## License

This dataset is released under IRB approval (202009-04). Ensure you understand its terms and conditions before using the data.

## Citation

If you find this dataset useful in your research, please consider citing:
```
Liu, Y. et al. (2023). Pelvic Fracture Segmentation Using a Multi-scale Distance-Weighted Neural Network. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14228. Springer, Cham. https://doi.org/10.1007/978-3-031-43996-4_30
```








