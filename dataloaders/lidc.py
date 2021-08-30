# This code is based on: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Modifications: Marc Gantenbein
# This software is licensed under the Apache License 2.0
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mypath import Path


def load_data_into_loader(sys_config, name, batch_size, transform=None):
    location = os.path.join(sys_config.data_root, name)
    dataset = LIDC_IDRI(dataset_location=location, transform=transform)


    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices, val_indices = indices[2*split:], indices[2*split:3*split], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=12, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
    validation_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
    print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

    return train_loader, test_loader, validation_loader


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []
    MEAN = 0.22223
    STD = 0.1843

    def __init__(self, dataset_location=Path.getPath('lidc'), transform=None, mode='ged'):
        """
        mode = choices(['ged', 'qubiq'])
        """
        self.transform = transform
        self.mode = mode
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        label = self.labels[index][np.random.randint(4)][None, ...]
        labels = np.stack(self.labels[index], axis=0)

        # Convert image and label to torch tensors
        image = (torch.from_numpy(image) - self.MEAN) / self.STD
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        # Normalise inputs
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        if self.mode == 'ged':
            return {'image': image, 'label': label, 'labels': labels}
        elif self.mode == 'qubiq':
            return {'image': image, 'label': labels}
        else:
            raise NotImplementedError

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

    def mean(self, ):
        return np.mean(self.images)

    def std(self, ):
        return np.std(self.images)
