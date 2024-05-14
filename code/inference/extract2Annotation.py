'''

coding by erick liu, 6/16/2023

input: image(.nii.gz) finished by anatomical segment
output: Each label is stored separately as a separate image(.nii.gz) for annotation
'''
import os
from nnunet.basicFunc import *
import SimpleITK as sitk
import pandas as pd


def saveDiffFrac(fileName):
    # load image data
    splitName = os.path.split(fileName)
    name = splitName[1].split('.nii.gz', 2)
    labelName = os.path.join(splitName[0], name[0] + '_mask.nii.gz')

    # ct_scale_img = pelvisOriginData(fileDir, maskName)
    #
    # ct_label_img = sitk.ReadImage(maskName)
    ct_origin_img = sitk.ReadImage(fileName)
    ct_label_img = sitk.ReadImage(labelName)
    # label = 1: sacrum / 2: Left iliac / 3:Right iliac
    # =============================================================================================
    # ======================extract the single fracture and Rescale Intensity======================
    # =============================================================================================
    frac_sacrum_img, frac_sacrum_norm = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img, frac_LeftIliac_norm = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img, frac_RightIliac_norm = extractSingleFrac(ct_origin_img, ct_label_img, 3)

    frac_sacrum_norm_scale = sitk.Cast(sitk.RescaleIntensity(frac_sacrum_norm), sitk.sitkFloat32)
    frac_LeftIliac_norm_scale = sitk.Cast(sitk.RescaleIntensity(frac_LeftIliac_norm), sitk.sitkFloat32)
    frac_RightIliac_norm_scale = sitk.Cast(sitk.RescaleIntensity(frac_RightIliac_norm), sitk.sitkFloat32)

    print('------------successful calculate the fracture--------------')
    # =============================================================================================
    # ====================output the nii.gz for manual segmentation in slicer======================
    # =============================================================================================
    sitk.WriteImage(frac_sacrum_img, os.path.join(splitName[0], name[0] + '_SA.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_img, os.path.join(splitName[0], name[0] + '_LI.nii.gz'))
    sitk.WriteImage(frac_RightIliac_img, os.path.join(splitName[0], name[0] + '_RI.nii.gz'))

    sitk.WriteImage(frac_sacrum_norm_scale, os.path.join(splitName[0], name[0] + '_SA_norm.nii.gz'))
    sitk.WriteImage(frac_LeftIliac_norm_scale, os.path.join(splitName[0], name[0] + '_LI_norm.nii.gz'))
    sitk.WriteImage(frac_RightIliac_norm_scale, os.path.join(splitName[0], name[0] + '_RI_norm.nii.gz'))
    print(splitName[0] + name[0] + '_RightIliac.nii.gz')

    return 0


def main_single():
    input_dir = r'N:\Database\1.TaskData\Task616_CT_pelvicFrac150\001_CXM\001_CXM.nii.gz'

    flag = saveDiffFrac(input_dir)

    if (flag == 0):
        print("save successful!")


def main_multi():
    # essential information and preprocessing
    work_dir = r'/usr/erick_data/Research_Database/TMI_data/test'
    work_dir_list = os.listdir(work_dir)
    # filter for dir
    work_dir_list = [name for name in work_dir_list if os.path.isdir(os.path.join(work_dir, name))]
    errorData = []
    dataCount = 0
    work_list_name = pd.DataFrame(data=work_dir_list)
    work_list_name.to_csv(os.path.join(work_dir, os.path.basename(work_dir) + '.csv'))

    for file in work_dir_list:
        print(file)
        dataCount += 1
        print("current data count being processed: ", str(dataCount))
        curr_dir = os.path.join(work_dir, file, file) + '.nii.gz'
        try:
            flag = saveDiffFrac(curr_dir)

        except:
            print('error: ', file)
            errorData.append(file)
        # 失败数据总结
        print('totalDataNum:', len(work_dir_list))
        print('failDataNum:', len(errorData))
        print('failDataName:', errorData)

    return 1


def main():
    main_multi()


if __name__ == "__main__":
    main()
