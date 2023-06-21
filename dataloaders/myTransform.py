import torch
import numpy as np
import random
import torch.nn.functional as F

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        x, y = sample
        assert type(x) is np.ndarray and type(y) is np.ndarray
        if random.random() < 0.5:
            x = x[:, :, -1::-1]
            y = y[:, :, -1::-1]

        return (x, y)
        
class RandomVerticalFlip(object):

    def __call__(self, sample):
        x, y = sample
        assert type(x) is np.ndarray and type(y) is np.ndarray
        if random.random() < 0.5:
            x = x[:, -1::-1, :]
            y = y[:, -1::-1, :]

        return (x, y)

class RandomResizedCrop:
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        

    def __call__(self, sample):
        x = sample['image']
        y = sample['lable']

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))


        return {
                'image': x,
                'label': y
                }

