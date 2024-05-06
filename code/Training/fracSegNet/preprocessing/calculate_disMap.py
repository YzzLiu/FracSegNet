import SimpleITK as sitk
import numpy as np

def bool_calculate_kernel(img_cal, length, width, height):
    i = length
    j = width
    k = height
    # zero_kernel = np.zeros((3,3,3))
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
    # print(len(kernel_data))

# input: seg_label(flieName) --> output: the edge array
def calculate_edge(seg_label):
    # itk_img = sitk.ReadImage(seg_label)
    print(
        '-----------------------------Start edge selected-------------------------')
    img_cal = seg_label
    img_shape = seg_label.shape  # calculate the shape of iamge
    edge_label_arr = np.zeros(img_shape)  # the saving array

    img_length = img_shape[0]
    img_width = img_shape[1]
    img_height = img_shape[2]

    # print(img_length, img_width, img_height)
    # ergodic the matrix
    for i in range(1, img_length - 1):
        for j in range(1, img_width - 1):
            for k in range(1, img_height - 1):
                # print(i, j, k)
                if img_cal[i, j, k] == 1:
                    if bool_calculate_kernel(img_cal, i, j, k) == 1:  # calculate the edge of label 1
                        # print(i, j, k)
                        edge_label_arr[i, j, k] = 1
    edge_label = edge_label_arr
    print("\n")
    print('-----------------------------Finished edge selected-------------------------')
    return edge_label  # return a edge matrix

# input: seg_label and edge_label --> output: the distance_map
# calculate the distance map
def distance_map(seg_label, edge_label):
    background_param = -10
    # edge_label = sitk.ReadImage(edge_label)

    seg_label_arr = seg_label  # get array from image
    seg_label_shape = seg_label_arr.shape  # calculate the shape of iamge
    print("seg_label_shape : ", seg_label_shape)

    edge_label_arr = edge_label  # get array from image
    edge_label_shape = edge_label_arr.shape  # calculate the shape of iamge
    print("edge_label_shape : ", edge_label_shape)
    distance_map_arr = np.ones(seg_label_shape)*background_param
    # save the (i,j,k) of 1 in edge_label_arr to a array
    edge_indexes_arr = np.argwhere(edge_label_arr == 1)
    # print(edge_indexes_arr.shape)
    # ergodic para
    if edge_indexes_arr.shape[0] > 50:
        img_length = edge_label_shape[0]
        img_width = edge_label_shape[1]
        img_height = edge_label_shape[2]

        # ergodic the matrix
        for i in range(1, img_length - 1):
            print('\r', "--------" + "Progress of search:" + str(round(i * 100 / img_length,2)) + "%" + "--------", end='',
                  flush=True)
            for j in range(1, img_width - 1):
                for k in range(1, img_height - 1):
                    # print(i, j, k)
                    if seg_label_arr[i, j, k] != -1:
                        distance_map_arr[i, j, k] = calculate_voxels_distance(edge_indexes_arr, i, j, k)

        distance_map_matrix = distance_map_arr

        print("\n")
        print('-----------------------------Finished distance calculate-------------------------')
    else:
        distance_map_matrix = np.ones_like(seg_label)
    return distance_map_matrix

# calculate the distance between voxels
def calculate_voxels_distance(edge_indexes_arr, length, width, height):
    voxel_number = edge_indexes_arr.shape[0]

    voxel_set = np.array(edge_indexes_arr)
    current_set = np.array([length, width, height])
    dis_cur = voxel_set - current_set
    distance_matrix = pow(dis_cur[:, 0], 2) + pow(dis_cur[:, 1], 2) + pow(dis_cur[:, 2], 2)
    # print(distance_matrix)
    distance = np.sqrt(min(distance_matrix))
    # print(distance)
    return distance

def disMap_weight_relu(distance_map):
    matrix_relu = np.zeros_like(distance_map)
    distance_map[distance_map != -10]/np.max(distance_map)*10
    matrix_relu[distance_map == -10] = 100
    matrix_relu[distance_map != -10] = distance_map[distance_map != -10]/np.max(distance_map)*10
    matrix_relu = np.reciprocal(1+np.exp(matrix_relu-5))*0.8 +0.2
    return matrix_relu


def calculate_disMap(label):
    if np.unique(label).max()>1:
        seg_label = label[0]
        edge_label = calculate_edge(seg_label)
        distance_map_matrix = distance_map(seg_label,edge_label)
        print('distance_map_matrix.shape:',distance_map_matrix.shape)
        # distance_map_matrix_new = [np.newaxis,distance_map_matrix]
        distance_map_matrix = disMap_weight_relu(distance_map_matrix)
        distance_map_matrix_new = distance_map_matrix[None].astype(np.float32)
    else:
        distance_map_matrix_new = np.ones_like(label)*0.2

    return distance_map_matrix_new


if __name__ == '__main__':
    # =========================== dir selected ================================
    fileName = '15_mask.nii.gz'
    write_dir = r'/home/erickliu/fractureSurf_disMap/'

    # =========================== edge selected ================================
    print('-----------------------------edge selected-------------------------')
    edgeLabel = calculate_edge(write_dir + fileName)
    sitk.WriteImage(edgeLabel, write_dir+'edge_label.nii.gz')
    print('-----------------------------Finished writing edge-------------------------')

    # =========================== Distance map calculate ================================
    # edgeLabel = 'edge_label_arr_pre.nii.gz'
    distance_map_nii = distance_map(write_dir + fileName, edgeLabel)
    sitk.WriteImage(distance_map_nii, write_dir + 'distance_map.nii.gz')