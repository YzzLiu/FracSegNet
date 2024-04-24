import os
import numpy as np
import torch
from nnunet.inference.segmentation_export import remove_component_and_save
from nnunet.training.model_restore import load_trained_model
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p


def predict_save_to_queue(preprocess_fn, list_of_lists, output_files, segs_from_prev_stage, classes, transpose_forward):
    errors_in = []

    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            d, _, dct = preprocess_fn(l)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the my_multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            if np.prod(d.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            return output_file, (d, dct)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_single(trainer, list_of_lists, output_files, segs_from_prev_stage=None,allow_downsample=False):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)
    #print("segs_from_prev_stage:",segs_from_prev_stage)
    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    if allow_downsample:
        preprocess_fn = trainer.preprocess_patient_allow_downsample
    else:
        preprocess_fn = trainer.preprocess_patient
    processes = predict_save_to_queue(preprocess_fn,list_of_lists,output_files,
                          segs_from_prev_stage,
                          classes,
                          trainer.plans['transpose_forward'])

    return processes


def predict_single_case(model, input_image_dir, output_filename, do_tta=False, clean_up=True, save_npz=False, save_stl=0, allow_downsample=False):

    # make new folder and change file extension to nii.gz
    dr, f = os.path.split(output_filename)
    if len(dr) > 0:
        maybe_mkdir_p(dr)
    ### if the file format is not niigz, change to it.
    if not f.endswith(".nii.gz"):
        f, _ = os.path.splitext(f)
        f = f + ".nii.gz"
    cleaned_output_file = join(dr, f)


    torch.cuda.empty_cache()
    print("empty cuda cache...")
    # print(torch.cuda.memory_summary())
    trainer, params = load_trained_model(model)   
    print("begain preprocessing...")
    preprocessing = preprocess_single(trainer, [[input_image_dir]], [cleaned_output_file], None, allow_downsample=allow_downsample)
    output_filename, (d, dct) = preprocessing
    if isinstance(d, str):
        data = np.load(d)
        os.remove(d)
        d = data

    print("begain prediction...")
    trainer.load_checkpoint_ram(params, False)
    trainer.data_aug_params['mirror_axes'] = (0, 1, 2)
    softmax=trainer.predict_preprocessed_data_return_softmax(d, do_tta, 1,
                                                              False, 1, trainer.data_aug_params['mirror_axes'],
                                                              True,True, 2 ,trainer.patch_size,True)
    transpose_forward = trainer.plans.get('transpose_forward')
    if transpose_forward is not None:
        transpose_backward = trainer.plans.get('transpose_backward')
        softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

    if save_npz:
        npz_file = output_filename[:-7] + ".npz"
    else:
        npz_file = None

    """There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the my_multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process.
    save_segmentation_nifti_from_softmax can take either filename or np.ndarray and will handle this automatically"""

    # if np.prod(softmax.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be safe
    #     print("This output is too large for python process-process communication. Saving output temporarily to disk. {}".format(
    #         output_filename[:-7] + ".npy"
    #     ))
    #     np.save(output_filename[:-7] + ".npy", softmax)
    #     softmax = output_filename[:-7] + ".npy"
    
    # del trainer
    # torch.cuda.empty_cache()
    # print("empty cuda cache...")
    # print(torch.cuda.memory_summary())
    return remove_component_and_save(softmax, output_filename, dct, 1, None, None, None, npz_file, rm_components=clean_up, save_stl=save_stl)

