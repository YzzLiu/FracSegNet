import shutil
from batchgenerators.utilities.file_and_folder_operations import join, subfolders, maybe_mkdir_p, save_json
import os

base = "/home/datasets/data_ct_fracture_all"
task_id = 600
task_name = "CT_PelvicFrac150"

foldername = "Task%03.0d_%s" % (task_id, task_name)

out_base = join(base, foldername)
imagestr = join(out_base, "imagesTr")
# imagests = join(out_base, "imagesTs")
labelstr = join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
# maybe_mkdir_p(imagests)
maybe_mkdir_p(labelstr)

folder_raw = "/media/yudi/5EE22D1BE22CF943/datasets/data_ct_fracture_all/data_CT_PelvicFracture_100_preprocessed"
all_cases = subfolders(folder_raw, join=True)

n_case = 0

for p in os.listdir(folder_raw):
    p_dir = join(folder_raw, p)
    for f in os.listdir(p_dir):
        if "_RI_frac.nii.gz" in f:
            label_file = join(p_dir, f)
            f_origin = f.replace("_frac.nii.gz", ".nii.gz")
            image_file = join(p_dir, f_origin)
            shutil.copy(image_file, join(imagestr, p + "_RI_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, p + "_RI.nii.gz"))
            n_case = n_case + 1

        if "_LI_frac.nii.gz" in f:
            label_file = join(p_dir, f)
            f_origin = f.replace("_frac.nii.gz", ".nii.gz")
            image_file = join(p_dir, f_origin)
            shutil.copy(image_file, join(imagestr, p + "_LI_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, p + "_LI.nii.gz"))
            n_case = n_case + 1

        if "_SA_frac.nii.gz" in f:
            label_file = join(p_dir, f)
            f_origin = f.replace("_frac.nii.gz", ".nii.gz")
            image_file = join(p_dir, f_origin)
            shutil.copy(image_file, join(imagestr, p + "_SA_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, p + "_SA.nii.gz"))
            n_case = n_case + 1

train_patient_names = os.listdir(labelstr)

json_dict = {}
json_dict['name'] = "CT fracture all 54 res"
json_dict['description'] = "CT fracture all 54 res"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "CT fracture all 54 res"
json_dict['licence'] = ""
json_dict['release'] = "2022.11.03"
json_dict['modality'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "0": "background",
    "1": "main fracture segment",
    "2": "segment 2",
    "3": "segment 3"
}

json_dict['numTraining'] = n_case
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in
                         train_patient_names]
# json_dict['test'] = ["./imagesTs/%s" % i for i in test_patient_names]
json_dict['test'] = []

save_json(json_dict, os.path.join(out_base, "dataset.json"))