import SimpleITK as sitk
import numpy as np
from FractureSeg.extract2Annotation import saveDiffFrac
import os


# 3*3*3 kernel
def bool_calculate_kernel(img_cal, length, width, height):
    i = length
    j = width
    k = height
    # zero_kernel = np.zeros((3,3,3))
    # zero_kernel[0,0,0] =
    kernel_data = [img_cal[i - 1, j - 1, k - 1], img_cal[i - 1, j - 1, k], img_cal[i - 1, j - 1, k + 1],
                   img_cal[i - 1, j, k - 1], img_cal[i - 1, j, k], img_cal[i - 1, j, k + 1],
                   img_cal[i - 1, j + 1, k - 1], img_cal[i - 1, j + 1, k], img_cal[i - 1, j + 1, k + 1],

                   img_cal[i, j - 1, k - 1], img_cal[i, j - 1, k], img_cal[i, j - 1, k + 1],
                   img_cal[i, j, k - 1], img_cal[i, j, k + 1],
                   img_cal[i, j + 1, k - 1], img_cal[i, j + 1, k], img_cal[i, j + 1, k + 1],

                   img_cal[i + 1, j - 1, k - 1], img_cal[i + 1, j - 1, k], img_cal[i + 1, j - 1, k + 1],
                   img_cal[i + 1, j, k - 1], img_cal[i + 1, j, k], img_cal[i + 1, j, k + 1],
                   img_cal[i + 1, j + 1, k - 1], img_cal[i + 1, j + 1, k], img_cal[i + 1, j + 1, k + 1]
                   ]
    arr_kernel_data = np.array(kernel_data)
    if (arr_kernel_data > 1).any():
        return 1
    else:
        return 0


# input: seg_label(flieName) --> output: the surface array
def calculate_surface(seg_label):
    itk_img = sitk.ReadImage(seg_label)
    print(
        '-----------------------------Start surface selected-------------------------')
    img_cal = sitk.GetArrayFromImage(itk_img)  # get array from image
    img_shape = img_cal.shape  # calculate the shape of iamge
    surface_label_arr = np.zeros(img_shape)  # the saving array

    img_length = img_shape[0]
    img_width = img_shape[1]
    img_height = img_shape[2]

    print(img_length, img_width, img_height)
    # ergodic the matrix
    for i in range(1, img_length - 1):
        print('\r', "--------" + "Progress of search:" + str(round(i * 100 / img_length, 2)) + "%" + "--------", end='',
              flush=True)
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                # print(i, j, k)
                if img_cal[i, j, k] == 1:
                    if bool_calculate_kernel(img_cal, i, j, k) == 1:  # calculate the surface of label 1
                        # print(i, j, k)
                        surface_label_arr[i, j, k] = 1
    surface_label = sitk.GetImageFromArray(surface_label_arr)
    surface_label.SetDirection(itk_img.GetDirection())
    surface_label.SetOrigin(itk_img.GetOrigin())
    surface_label.SetSpacing(itk_img.GetSpacing())
    print("\n")
    print(
        '-----------------------------Finished surface selected-------------------------')
    return surface_label  # return a surface matrix


# calculate voxels distance
def calculate_voxels_distance(surface_indexes_arr, length, width, height):
    voxel_number = surface_indexes_arr.shape[0]

    voxel_set = np.array(surface_indexes_arr)
    current_set = np.array([length, width, height])
    dis_cur = voxel_set - current_set
    distance_matrix = pow(dis_cur[:, 0], 2) + pow(dis_cur[:, 1], 2) + pow(dis_cur[:, 2], 2)
    # print(distance_matrix)
    distance = np.sqrt(min(distance_matrix))
    # print(distance)
    return distance


# input: seg_label and surface_label --> output: the distance_map
def distance_map(seg_label, surface_label):
    background_param = -10
    seg_label = sitk.ReadImage(seg_label)
    # surface_label = sitk.ReadImage(surface_label)

    seg_label_arr = sitk.GetArrayFromImage(seg_label)  # get array from image
    seg_label_shape = seg_label_arr.shape  # calculate the shape of iamge
    print("seg_label_shape : ", seg_label_shape)

    surface_label_arr = sitk.GetArrayFromImage(surface_label)  # get array from image
    surface_label_shape = surface_label_arr.shape  # calculate the shape of iamge
    print("surface_label_shape : ", surface_label_shape)

    # save the (i,j,k) of 1 in surface_label_arr to a array
    surface_indexes_arr = np.argwhere(surface_label_arr == 1)
    print(surface_indexes_arr.shape)
    # the saving array
    distance_map_arr = background_param * np.ones(surface_label_shape)
    print("distance_map_arr_shape : ", distance_map_arr.shape)

    # ergodic para
    img_length = surface_label_shape[0]
    img_width = surface_label_shape[1]
    img_height = surface_label_shape[2]

    # ergodic the matrix
    for i in range(1, img_length - 1):
        print('\r', "--------" + "Progress of search:" + str(round(i * 100 / img_length, 2)) + "%" + "--------", end='',
              flush=True)
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                # print(i, j, k)
                if seg_label_arr[i, j, k] != 0:
                    distance_map_arr[i, j, k] = calculate_voxels_distance(surface_indexes_arr, i, j, k)
                # elif seg_label_arr[i, j, k] == 0:
                #     distance_map_arr[i, j, k] = -10

    distance_map_matrix = sitk.GetImageFromArray(distance_map_arr)
    distance_map_matrix.SetDirection(seg_label.GetDirection())
    distance_map_matrix.SetOrigin(seg_label.GetOrigin())
    distance_map_matrix.SetSpacing(seg_label.GetSpacing())
    print("\n")
    print('-----------------------------Finished distance calculate-------------------------')
    return distance_map_matrix


def calculate_disMap(label):
    if sitk.GetArrayFromImage(sitk.ReadImage(label)).max() > 1:
        seg_label = label
        surface_label = calculate_surface(seg_label)
        if np.sum(sitk.GetArrayFromImage(surface_label))>10:
           distance_map_matrix = distance_map(seg_label, surface_label)
        else:
            seg_label = sitk.ReadImage(label)
            distance_map_matrix = np.ones_like(seg_label)
            distance_map_matrix = sitk.GetImageFromArray(distance_map_matrix)
            distance_map_matrix.SetDirection(seg_label.GetDirection())
            distance_map_matrix.SetOrigin(seg_label.GetOrigin())
            distance_map_matrix.SetSpacing(seg_label.GetSpacing())
    else:
        seg_label = sitk.ReadImage(label)
        distance_map_matrix = np.ones_like(seg_label)
        distance_map_matrix = sitk.GetImageFromArray(distance_map_matrix)
        distance_map_matrix.SetDirection(seg_label.GetDirection())
        distance_map_matrix.SetOrigin(seg_label.GetOrigin())
        distance_map_matrix.SetSpacing(seg_label.GetSpacing())
    return distance_map_matrix


def CFS_calculate_single(work_dir):
    fileName = os.path.join(work_dir,os.path.basename(work_dir))+'.nii.gz'
    flag = saveDiffFrac(fileName)
    masks = [None]*3
    DFM = [None]*3
    masks[0] = os.path.join(work_dir, os.path.basename(work_dir)) + '_LI_frac.nii.gz'
    masks[1] = os.path.join(work_dir, os.path.basename(work_dir)) + '_RI_frac.nii.gz'
    masks[2] = os.path.join(work_dir, os.path.basename(work_dir)) + '_SA_frac.nii.gz'
    for i in range(3):
        DFM[i] = calculate_disMap(masks[i])
    sitk.WriteImage(DFM[0],os.path.join(work_dir, os.path.basename(work_dir)) + '_LI_frac_DFM.nii.gz')
    sitk.WriteImage(DFM[1],os.path.join(work_dir, os.path.basename(work_dir)) + '_RI_frac_DFM.nii.gz')
    sitk.WriteImage(DFM[2],os.path.join(work_dir, os.path.basename(work_dir)) + '_SA_frac_DFM.nii.gz')

def main():
    work_dir = r'N:\Database\1.TaskData\Task616_CT_pelvicFrac150\001_CXM'
    CFS_calculate_single(work_dir)


if __name__ == '__main__':
    main()
