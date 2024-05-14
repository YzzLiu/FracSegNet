'''

coding by erick liu, 6/16/2023

input: image(.nii.gz) finished by anatomical segment
output: Each label is stored separately as a separate image(.nii.gz) for annotation
'''
import os
from nnunet.basicFunc import *
import SimpleITK as sitk
import pandas as pd


def saveDiffFrac(fileName, labelName):
    # load image data
    splitName = os.path.split(fileName)
    name = splitName[1].split('.nii.gz', 2)

    # ct_scale_img = pelvisOriginData(fileDir, maskName)
    #
    # ct_label_img = sitk.ReadImage(maskName)
    ct_origin_img = sitk.ReadImage(fileName)
    ct_label_img = sitk.ReadImage(labelName)
    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip
    # =============================================================================================
    # ======================extract the single fracture and Rescale Intensity======================
    # =============================================================================================
    frac_sacrum_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 3)

    sitk.WriteImage(frac_sacrum_img, os.path.join(splitName[0], name[0] + '_SA.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_img, os.path.join(splitName[0], name[0] + '_LI.nii.gz'))
    sitk.WriteImage(frac_RightIliac_img, os.path.join(splitName[0], name[0] + '_RI.nii.gz'))

    return 0


if __name__ == "__main__":
    ct_name = "/usr/erickliu/image.nii.gz"
    mask_name = "/usr/erick_data/mask.nii.gz"
    
    saveDiffFrac(ct_name,mask_name)
