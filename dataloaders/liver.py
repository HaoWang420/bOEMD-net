import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections.abc import Sequence
# from mypath import Path
import nibabel as nib
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F

class SpatialRotation():
    def __init__(self, dimensions: Sequence, k: Sequence = [3], auto_update=True):
        self.dimensions = dimensions
        self.k = k
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = [random.choice(self.k) for dim in self.dimensions]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        for k, dim in zip(self.args, self.dimensions):
            x = torch.rot90(x, k, dim)
        return x

class SpatialFlip():
    def __init__(self, dims: Sequence, auto_update=True) -> None:
        self.dims = dims
        self.args = None
        self.auto_update = auto_update
        self.update()

    def update(self):
        self.args = tuple(random.sample(self.dims, random.choice(range(len(self.dims)))))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_update:
            self.update()
        x = torch.flip(x, self.args)
        return x

class Liver_p_id(Dataset):
    def __init__(self, dataset_location = "/data/sdb/qingqiao/workspaces/datasets/liver_datasets/processed", mode="fix", data_mode="train"):
        """_summary_

        Args:
            dataset_location (_type_, optional): _description_. Defaults to Path.getPath('liver').
            transform (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to "fix".
            data_mode (str, optional): _description_. Defaults to "train".
        """
        TRAIN = 'train'
        VAL = 'val'
        TEST = "test"
        transformations = [
            transforms.Normalize(mean = (0.0255), std = (0.2244))
            
        ]
        
        self.transform = None
        assert mode in ("fix", "random", "choice", "choice_p")
        self.mode = mode
        self.data_mode = data_mode

        assert data_mode in (TRAIN, VAL, TEST)
        # if data_mode == TRAIN:
        #     transformations += [
        #         SpatialRotation([(1,2)], [*[0]*3,1,2,3], auto_update=False),
        #         SpatialFlip(dims=(1,2), auto_update=False),
        #     ]
            # transformations += self.updated_transform
        
        self.transform = transforms.Compose(transformations)
        
        self.images = []
        self.labels = []
        patient_index = -1
        if data_mode == TRAIN:
            patient_index = (0, 8)  
        elif data_mode == VAL:
            patient_index = (9, 11)
        else:
            patient_index = (12, 14)
        
        for i, patient in enumerate( os.listdir(dataset_location) ):
            if i >= patient_index[0] and i <= patient_index[1]:
                folder = os.path.join(dataset_location, patient)
                imgs =  [x for x in os.listdir(folder) if not x.startswith("image")]
                for image in imgs:
                    self.images.append(nib.load(os.path.join(folder, image)).get_fdata())
                    temp_label = []
                    image_index = image.split(".")[0].split("_")[-1]
                    for expert_index in range(1, 3):
                        temp_label.append(nib.load(os.path.join(folder, 
                                                                "expert{expert_index}_{image_index}.nii.gz".format(
                                                                    expert_index = expert_index + 1, image_index = image_index))).get_fdata())
                    self.labels.append(temp_label)
                    
        assert(len(self.images) == len(self.labels))

        if mode == "random":
            self.perm_mat = np.zeros((len(self.images), 2), dtype=np.int)
            for ii in range(len(self.images)):
                self.perm_mat[ii] = np.random.permutation(2)
        
    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)
        # print(self.labels[index])
        # print(self.labels[index][0].shape)
        labels = np.stack(self.labels[index], axis = 0)
        # swap 2 by 1
        # print(np.unique(labels))
        labels[labels == 2] = 1
      
        if self.mode == "random":
            labels = labels[self.perm_mat[index]]
        
        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        labels = torch.from_numpy(labels)
        
        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        
        # pad the image and label to dimenstion 512*512
        
        #Normalize inputs
        self.transform(image)
        out = {}
        if self.mode == 'choice' and self.data_mode == "train":
            out['label'] = labels[np.random.randint(len(labels))][None, ...]
        elif self.mode == "choice_p":
            image = F.pad(image, (56, 56, 56, 56))
            labels = F.pad(labels, (56, 56, 56, 56))
            out['label'] = labels[np.random.randint(len(labels))]
        else:
            out['label'] = labels
        out['image'] = image
            
        return out

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    liver_set = Liver_p_id(mode = 'choice_p')
    loader = DataLoader(liver_set, batch_size = 4)
    for input in loader:
        print(input['image'].shape)
        print(input['label'].shape)
        break