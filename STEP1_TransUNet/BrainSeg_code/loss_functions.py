import torch
import torch.nn as nn
import torch.nn.functional as F

import copy 


# dice loss reference: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html#DiceLoss
# dice loss reference: https://minimin2.tistory.com/179
# dice loss reference: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py

"""
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc, dsc
"""

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5


    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                    y_pred.sum() + y_true.sum() + self.smooth
                )

        return 1. - dsc



