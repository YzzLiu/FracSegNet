# FracSegNet(updated in 2023.10.15)
2023-10-15: updated [DataSet for Pelvic Fracture Segmentation](https://github.com/YzzLiu/FracSegNet/tree/main/DataSet)

# What is FracSegNet?
Pelvic fracture is a severe type of high-energy injury. Segmentation of pelvic fractures from 3D CT images is important for trauma diagnosis, evaluation, and treatment planning. Manual delineation of the fracture surface can be done in a slice-by-slice fashion but is slow and error-prone. Automatic fracture segmentation is challenged by the complex structure of pelvic bones and the large variations in fracture types and shapes. 

This study proposes a deep-learning method for automatic pelvic fracture segmentation. Our approach consists of two consecutive networks. The anatomical segmentation network extracts left and right ilia and sacrum from CT scans. Then, the fracture segmentation network further isolates the fragments in each masked bone region. 
![Overview](documentation/assets/Overview.png)

## 1.Pelvic Fracture DataSets
We built a dataset containing 100 CT scans with fractured pelvis and manually annotated the fractures. 

The first version of the pelvic fracture segmentation dataset has been updated. In this dataset, we provide detailed annotations of fracture segmentation for 100 patients. For each patient, we offer CT data (with the background masked out) for the left ilium, right ilium, and sacrum, along with their annotations. 

Data Discription: [DataSet for Pelvic Fracture Segmentation](https://github.com/YzzLiu/FracSegNet/tree/main/DataSet)

Paper link: https://link.springer.com/chapter/10.1007/978-3-031-43996-4_30

## 2.Pelvic Fracture Segmentation Network

We will soon launched a Pelvic Fracture Segmentation Neural Network.


