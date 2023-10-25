import nnunet
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import importlib
import pkgutil
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
# from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
import os

def recursive_find_trainer(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_trainer([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr


def restore_model(pkl_file, checkpoint=None, train=False):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    
    search_in = join(nnunet.__path__[0], "training", "network_training")
    print(search_in)
    tr = recursive_find_trainer([search_in], name, current_module="nnunet.training.network_training")


    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    # assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
    #                                       "Please make it so!"

    if init[0].endswith('2D.pkl'):
        initnew = list(init)+['2d']
        del init
        init = tuple(initnew)

    #print(len(init))
    #print(*init)
    #print(type(init))
    if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]

    # init[0] is the plans file. This argument needs to be replaced because the original plans file may not exist
    # anymore.
    trainer = tr(*init)
    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    checkpoint = join(folder, "model_best.model")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)

def load_model_and_checkpoint_files(folder, folds=None):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).

    This is best used for inference and test prediction
    :param folder:
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        print('folds: ', folds)
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    #print('we will really load fold[0]: ',folds[0])
    trainer = restore_model(join(folds[0], "model_best.model.pkl"))
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    # trainer.update_fold(None)
    # raise NotImplementedError("I think you should check here... liupengbo-20200621")
    #print('prediction trainer.output_folder: ',trainer.output_folder)
    trainer.initialize(False)
    # raise NotImplementedError("I think you should check here... liupengbo-20200621")
    all_best_model_files = [join(i, "model_best.model") for i in folds]
    #print("!!using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cuda', torch.cuda.current_device())) for i in all_best_model_files]
    return trainer, all_params



def restore_model_simple(pkl_file, checkpoint=None, train=False):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']

    # search_in = join(nnunet.__path__[0], "training", "network_training")
    # tr = recursive_find_trainer([search_in], name, current_module="nnunet.training.network_training")
    if name=="nnUNetTrainer":
        tr=nnUNetTrainer
    elif name=="nnUNetTrainerV2":
        tr=nnUNetTrainerV2
    else:
        tr=nnUNetTrainer
        
    current_module="nnunet.training.network_training"
    m = importlib.import_module(current_module + "." + name)
    tr = getattr(m, name)

    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"

    if init[0].endswith('2D.pkl'):
        initnew = list(init)+['2d']
        del init
        init = tuple(initnew)

    #print(len(init))
    #print(*init)
    #print(type(init))
    if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]

    # init[0] is the plans file. This argument needs to be replaced because the original plans file may not exist
    # anymore.
    trainer = tr(*init)
    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer



def load_trained_model(folder):
    """
    No fold needed. 
    Automatically detect model file in this direct folder.
    """
    pkl_file="model_best.model.pkl"
    for file in os.listdir(folder):
        if ".model.pkl" in file:
            pkl_file=file
            break
    model_file=pkl_file[:-4]
    assert model_file in os.listdir(folder), "Err: Model file not found."
    
    trainer = restore_model_simple(join(folder, pkl_file))
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.initialize(False)
    params = torch.load((join(folder, model_file)), map_location=torch.device('cuda', torch.cuda.current_device()))
    return trainer, params

if __name__ == "__main__":
    pkl = r"E:\TMI-data-v20230926\code2Github\code\FractureSeg\trained_model\FractureSegModel\model_best.model.pkl"
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
