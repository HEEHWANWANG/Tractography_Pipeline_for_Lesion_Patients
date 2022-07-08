import torch
import torch.nn as nn
import torch.nn.functional as F

import monai

import copy 


# dice loss reference 1: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html#DiceLoss
# dice loss reference 2: https://minimin2.tistory.com/179
# dice loss reference 3: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
# dice loss, tversky loss, focal loss, tversky focal loss reference: https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
# tversky loss for multiple sclerosis segmentation: https://arxiv.org/pdf/1706.05721.pdf
 
"""
class DiceLoss(nn.Module):

    def __init__(self, mode = None):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
"""

class DiceLoss(nn.Module):
    def __init__(self, mode = 'train'):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5
        self.mode = mode

    def _calculate_weights(self, y_true):
        weight0 = y_true[:,1,:,:].sum() / (y_true[:,0,:,:].sum() + y_true[:,1,:,:].sum())
        weight1 = 1 - weight0
        return torch.tensor([weight0, weight1]) 

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        n_classes = y_true.size()[1]
        
        if self.mode == 'train': 
            weights = self._calculate_weights(y_true)
            ds_coef = None
            loss = 0.0
            for c in range(n_classes): 
                y_pred_channel = y_pred[:, c, :, :]
                y_true_channel = y_true[:, c, :, :]
                intersection = (y_pred_channel * y_true_channel).sum()
                y_pred_channel_sum = (y_pred_channel * y_pred_channel).sum()
                y_true_channel_sum = (y_true_channel * y_true_channel).sum()
                dsc = (2. * intersection + self.smooth) / (y_pred_channel_sum + y_true_channel_sum + self.smooth)
                loss += weights[c] * (1 - dsc)
            
        else:
            weights = torch.tensor([1/n_classes] * n_classes)
            ds_coef = []
            loss = 0.0
            for c in range(n_classes): 
                y_pred_channel = y_pred[:, c, :, :]
                y_true_channel = y_true[:, c, :, :]
                intersection = (y_pred_channel * y_true_channel).sum()
                y_pred_channel_sum = (y_pred_channel * y_pred_channel).sum()
                y_true_channel_sum = (y_true_channel * y_true_channel).sum()
                dsc = (2. * intersection + self.smooth) / (y_pred_channel_sum + y_true_channel_sum + self.smooth)
                ds_coef.append(dsc.item())
                loss += weights[c] * (1 - dsc)
        return loss, ds_coef



class weighted_BCELoss(nn.Module): 
    def __init__(self, mode = 'train'):
        super(weighted_BCELoss, self).__init__()
        self.mode = mode

    def _calculate_weights(self, y_true):
        weight0 = y_true[:,1,:,:].sum() / (y_true[:,0,:,:].sum() + y_true[:,1,:,:].sum())
        weight1 = 1 - weight0
        return torch.tensor([weight0, weight1]) 

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        if self.mode == 'train': 
            weights = self._calculate_weights(y_true)
            batch_size, height, width = y_true.size()[0], y_true.size()[2], y_true.size()[3]
            weight0 = weights[0] * torch.ones((batch_size, height, width)).unsqueeze(1)
            weight1 = weights[1] * torch.ones((batch_size, height, width)).unsqueeze(1)
            pix_weights = torch.cat([weight0, weight1], dim = 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, weight=pix_weights)
        
        else: 
            weights = torch.tensor([0.5,0.5])      
            batch_size, height, width, depth = y_true.size()[0], y_true.size()[2], y_true.size()[3], y_true.size()[4]
            weight0 = weights[0] * torch.ones((batch_size, height, width, depth)).unsqueeze(1)
            weight1 = weights[1] * torch.ones((batch_size, height, width, depth)).unsqueeze(1)
            pix_weights = torch.cat([weight0, weight1], dim = 1)
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true, weight=pix_weights)
        
        return loss 


class FocalLoss(nn.Module):
    def __init__(self, gamma = 5):
        super(FocalLoss, self).__init__()
        self.loss_fn = monai.losses.FocalLoss(gamma=gamma)

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        loss = self.loss_fn(y_pred, y_true)
        return loss 


class TverskyLoss(nn.Module): 
    def __init__(self, alpha = 0.2, smooth = 1e-5):
        super(TverskyLoss, self).__init__()
        self.loss_fn = monai.losses.TverskyLoss(alpha = alpha, smooth_nr = smooth, smooth_dr = smooth) 

    def forward(self, y_pred, y_true): 
        assert y_pred.size() == y_true.size()
        loss = self.loss_fn(y_pred, y_true)
        return loss 


class FocalTverskyLoss(nn.Module):
    def __init__(self, gamma = 5, alpha = 0.1 , smooth = 1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.loss_fn = monai.losses.TverskyLoss(alpha = alpha, smooth_nr = smooth, smooth_dr = smooth)

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        loss = self.loss_fn(y_pred, y_true)
        return torch.pow((1. - loss), self.gamma)


        


