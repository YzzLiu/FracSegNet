# AnatomicalSegModel - Pelvic Fracture Segmentation Model

## Overview

FracSegNet consists of two consecutive neural networks. The first network performs anatomical segmentation, extracting the left and right hip and sacrum from CT scans. The second network focuses on fracture segmentation, further isolating the fracture fragments within each masked bone region. This two-step approach ensures accurate fracture segmentation despite the complex structure of pelvic bones and the variations in fracture types and shapes.
