# FractureSegModel - Pelvic Fracture Segmentation Model

## Overview

FracSegNet consists of two consecutive neural networks. The first network performs anatomical segmentation, extracting the left and right ilia and sacrum from CT scans. The second network focuses on fracture segmentation, further isolating the fracture fragments within each masked bone region. This two-step approach ensures accurate fracture segmentation despite the complex structure of pelvic bones and the variations in fracture types and shapes.


## Model Description

FractureSegModel is a deep learning model trained on the Pelvic Fracture Dataset for the task of pelvic fracture segmentation. It is designed to automatically segment pelvic fractures from 3D CT images, which is crucial for trauma diagnosis, evaluation, and treatment planning. This model is a part of the FracSegNet project.



## Download and Usage

You can download the AnatomicalSegModel from the following link:

[Download AnatomicalSegModel](https://drive.google.com/file/d/1yLTYsji3XdBKOemmXNe8iD9Puyn_9VhD/view?usp=sharing)

### Usage

1. **Prerequisites:** Make sure you have the necessary Python environment and libraries installed. You can use a virtual environment to manage dependencies.

2. **Loading the Model:** Use your preferred deep learning framework (e.g., TensorFlow, PyTorch) to load the downloaded model checkpoint.

3. **Inference:** You can use the loaded model to perform pelvic fracture segmentation on your own CT images. Provide the input CT scan to the model, and it will produce segmentation results.

4. **Post-processing:** Depending on your specific application, you may need to apply post-processing techniques to refine the segmentation results.

## License

This model is provided under the terms of the license specified in the FracSegNet project repository.

## Acknowledgments

We acknowledge the support and contributions from the Pelvic Fracture Dataset and the research community.
Thanks to Febian, et al. and Pengbo Liu, et al.'s excellent work

For any questions or issues, please feel free to contact us.

---
*This work is a part of the FracSegNet project. For more information, visit [FracSegNet GitHub Repository](https://github.com/YzzLiu/FracSegNet).*


## Citation

If you use AnatomicalSegModel in your research, please cite the following paper:
```
@InProceedings{10.1007/978-3-031-43996-4_30,
author="Liu, Yanzhen and Yibulayimu, Sutuke and Sang, Yudi and Zhu, Gang and Wang, Yu and Zhao, Chunpeng and Wu, Xinbao",
title="Pelvic Fracture Segmentation Using a Multi-scale Distance-Weighted Neural Network",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="312--321",
isbn="978-3-031-43996-4"
}

## References
<sup>1</sup> https://github.com/MIC-DKFZ/nnUNet  
<sup>2</sup> https://github.com/MIRACLE-Center/CTPelvic1K

