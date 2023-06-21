from skimage.io import imsave
import nibabel as nib
import numpy as np

import os

def brats_preprocessing(root, root_to_save):
    train = os.path.join(root, 'train', 'training_data_v2', 'brain_tumor', 'Training')
    val = os.path.join(root, 'val', 'validation_data_v2', 'brain-tumor', 'Validation')
    test = os.path.join(root, 'test', 'test_QUBIQ', 'brain-tumor', 'Testing')

    for _set in [train, val, test]:
        dirs = os.scandir(_set)
        for subdirs in dirs:
            ssdirs = os.scandir(subdirs)
            for fl in ssdirs:
                pass
