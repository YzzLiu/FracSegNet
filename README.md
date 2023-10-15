# FracSegNet(updated in 2023.10.15)
[DataSet for Pelvic Fracture Segmentation]()
# Abstract
Pelvic fracture is a severe type of high-energy injury. Segmentation of pelvic fractures from 3D CT images is important for trauma diagnosis, evaluation, and treatment planning. Manual delineation of the fracture surface can be done in a slice-by-slice fashion but is slow and error-prone. Automatic fracture segmentation is challenged by the complex structure of pelvic bones and the large variations in fracture types and shapes. 

This study proposes a deep-learning method for automatic pelvic fracture segmentation. Our approach consists of two consecutive networks. The anatomical segmentation network extracts left and right ilia and sacrum from CT scans. Then, the fracture segmentation network further isolates the fragments in each masked bone region. 

We design and integrate a distance-weighted loss into a 3D U-net to improve accuracy near the fracture site. In addition, multi-scale deep supervision and a smooth transition strategy are used to facilitate training. We built a dataset containing 100 CT scans with fractured pelvis and manually annotated the fractures. A five-fold cross-validation experiment shows that our method outperformed max-flow segmentation and network without distance weighting, achieving a global Dice of 99.38\%, a local Dice of 93.79\%, and an Hausdorff distance of 17.12 mm.

## 1.Pelvic Fracture DataSets

We will soon release a medical imaging database of pelvic fractures, which will include clinical CT images and expert annotated data from multiple sets of desensitized and ethically reviewed pelvic fractures. We hope it can contribute to the progress and development of community.

## 2.Pelvic Fracture Segmentation Network

We will soon launched a Pelvic Fracture Segmentation Neural Network.


