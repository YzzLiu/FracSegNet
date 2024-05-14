# DataSet for Pelvic Fracture Segmentation

Pelvic fractures are critical high-energy injuries. Manual delineation of the fracture surface from 3D CT images can be time-consuming and prone to errors. This dataset aims to facilitate the development of automated algorithms, especially for pelvic research and reduction planning.

## Abstract

Pelvic fracture is a severe type of high-energy injury. The segmentation of pelvic fractures from 3D CT images plays a pivotal role in trauma diagnosis, evaluation, and subsequent treatment planning. While manual delineation can be conducted in a slice-by-slice manner, it is inherently slow and often fraught with errors. Automatic segmentation remains a challenge due to the intricate structure of pelvic bones and the vast variability in fracture types and shapes. 

In this study, we introduced a novel deep-learning technique for the automatic segmentation of pelvic fractures. Our methodology comprises two sequential networks. Initially, the anatomical segmentation network sifts through the CT scans to extract the left and right ilia and sacrum. Subsequently, the fracture segmentation network isolates the fragments within each masked bone region. We integrated a distance-weighted loss into a 3D U-net to enhance accuracy near the fracture vicinity. Moreover, we utilized multi-scale deep supervision and employed a smooth transition strategy for effective training.

Tested on a curated dataset of 150 CTs, which we have made publicly available, our method achieves an average Dice coefficient of 0.986 and an average symmetric surface distance of 0.234 mm.

In our quest to advance pelvic research, especially in reduction planning, we have made our dataset and corresponding source code accessible to the public.

## Dataset Description

We collated a dataset encompassing 150 preoperative CT scans, representing an array of prevalent pelvic fractures. These scans have been sourced from a pool of 150 patients (ranging between 16-94 years of age, including 63 females and 87 males) slated for pelvic reduction surgery at Beijing Jishuitan Hospital between the years 2017-2023. 
By sourcing CT scans from a diverse array of machines, including a Toshiba Aquilion scanner, an United Imaging uCT 550 scanner, an United Imaging uCT 780 scanner, and a Philips Brilliance scanner, we have ensured a broad representation of imaging characteristics.

## Dataset Structure and Contents

The dataset is organized in a structured folder system for easy navigation and utilization. Here's an overview:
```
DataSet for Pelvic Fracture Segmentation
|
|-- PENGWIN_CT_train_images
|   |-- 001.mha
|   |-- 002.mha
|   |-- 003.mha
|   |-- ...(other files)
|
|-- PENGWIN_CT_train_labels
|   |-- 001.mha
|   |-- 002.mha
|   |-- 003.mha
|   |-- ...(other files)

**File Descriptions:**
This repository contains the training set of 100 CT scans with pelvic fractures and their ground-truth segmentation labels. The images and labels are stored in mha format. Each bone anatomy (sacrum, left hipbone, right hipbone) has up to 10 fragments. Bone that does not present any fracuture has only one fragment, which is itself. Label assignment: 0 = background, 1-10 = sacrum fragment, 11-20 = left hipbone fragment, 21-30 = right hipbone fragment. 

## Usage

1. Download the dataset from [DataSet of Pelvic Frcature Segmenation(zenodo)](https://zenodo.org/api/records/10927452/files-archive).
2. Implement [fracture segmentation algorithms]().
3. Use our dataset for training/testing purposes, respecting the terms of the license.

## License

This dataset is released under IRB approval (202009-04). Ensure you understand its terms and conditions before using the data.

## Citation

If you find this dataset useful in your research, please consider citing:
```
Liu, Y. et al. (2023). Pelvic Fracture Segmentation Using a Multi-scale Distance-Weighted Neural Network. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14228. Springer, Cham. https://doi.org/10.1007/978-3-031-43996-4_30
```








