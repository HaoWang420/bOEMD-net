import os
import torch
import numpy as np
from torch import masked_fill
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory, dataset='uncertain-brats'):
        self.directory = directory
        self.dataset = dataset

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    
    def visualize(self, writer, image, target, pred, global_step):
        image_grid = make_grid(image[-1:, -1:].clone().cpu().data, normalize=False)
        writer.add_image('Image', image_grid, global_step)
        
        target_grid = make_grid(
                        target[-1:, -1:].detach().cpu().data, 
                        normalize=False, 
                        )
        writer.add_image('Ground Truth label', target_grid, global_step)


        # sigmoid activation to produce probability map
        pred_grid = make_grid(
                        self.__sigmoid(pred[-1:, -1:].detach().cpu().data), 
                        normalize=False,
                        )
        writer.add_image('Predicted label', pred_grid, global_step)

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))


