from time import time
from fracSegNet.predict_single import predict_single_case
from os.path import isdir
import os
import pandas as pd


def anatomical_seg(input_dicom_or_nii, output_file, model_folder_name, tta=False, clean_up=False, save_stl=0,
                   allow_downsample=False):
    assert output_file.endswith(".nii.gz"), "output_file should end with .nii.gz"
    assert isdir(model_folder_name), "model output folder not found"
    print("using network model in ", model_folder_name)

    time0 = time()
    predict_single_case(model_folder_name, input_dicom_or_nii, output_file, do_tta=tta, clean_up=clean_up,
                        save_stl=save_stl, allow_downsample=allow_downsample)
    time1 = time()
    print("time =", time1 - time0, "s")

    return 1


def get_model_path(model_name):
    # 获取当前文件的绝对路径
    current_file_path = os.path.realpath(__file__)
    # 获取与当前程序所在文件夹相同的另一个文件夹路径
    model_dir = os.path.join(os.path.dirname(current_file_path), 'trained_model')
    model_path = os.path.join(model_dir, model_name)

    return model_path


def main():
    # essential information and preprocessing
    model_name = 'AnatomicalSegModel'
    model_path = get_model_path(model_name)
    parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(parent_directory)
    work_dir = os.path.join(parent_directory, "DataSet", "testData")
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
        input_file = os.path.join(work_dir, file, file) + '.nii.gz'
        output_file = os.path.join(work_dir, file, file) + '_mask.nii.gz'
        try:
            print('--------------------------Start Anatomical Segmentation--------------------------')
            flag = anatomical_seg(input_file, output_file, model_path)
        except:
            print('error: ', file)
            errorData.append(file)
        # 失败数据总结
        print('totalDataNum:', len(work_dir_list))
        print('failDataNum:', len(errorData))
        print('failDataName:', errorData)

    # inference
    #

    return 1

if __name__ == "__main__":
    main()
