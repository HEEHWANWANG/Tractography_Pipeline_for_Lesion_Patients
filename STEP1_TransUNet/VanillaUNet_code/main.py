
import argparse
import logging
import os
import random
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn

from networks.UNet import UNet
from dataloader import *
from experiments_slice import *

import time





parser = argparse.ArgumentParser()

## data parameters
parser.add_argument('--root_dir', type=str,
#                    default='/scratch/connectome/dhkdgmlghks/lesion_tract_pipeline/lesion_tract/TransUNet_training/TransUNet_code', help='root dir for data')
                    default='/home/ubuntu/dhkdgmlghks/TransUNet/VanillaUNet_code', help='root dir for data') 
parser.add_argument("--val_size",type=float, 
                    default=0.1, required=False,help='The ratio of data used for validation')
parser.add_argument("--test_size",type=float, 
                    default=0.1, required=False,help='The ratio of data used for test')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')

## model parameters 
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network. For lesion binary mask, this should be set as 2')

## training parameters 
parser.add_argument("--cross_entropy_weights",type=float,
                    default=[0.1, 0.9], nargs='*', required=False,help='weights for weighted cross entropy. latter number should be large because the number of lesion voxels is smaller than the number of non-lesion voxels.')
parser.add_argument("--epoch",type=int,
                    required=True,help='the number of epoch for training')
parser.add_argument("--optimizer",type=str,
                    required=True,help='', choices=['Adam','SGD'])
parser.add_argument("--lr", type=float, 
                    default=0.01,required=False,help='learing rate')
parser.add_argument("--weight_decay",type=float,
                    default=1e-4, required=False,help='weight decay of optimizer')
parser.add_argument("--train_batch_size",type=int, 
                    default=96, required=False,help='the size of batch for training')
parser.add_argument("--val_batch_size",type=int, 
                    default=96, required=False,help='the size of batch for validation')
parser.add_argument("--test_batch_size",type=int, 
                    default=96, required=False,help='the size of batch for test')


## other parameters 
parser.add_argument("--exp_name",type=str,
                    required=True,help='')
parser.add_argument("--gpus", type=int,
                    nargs='*', required=False, help='gpu device ids used for training')
parser.add_argument("--sbatch", type=str, 
                    required=False, choices=['True', 'False'], help='If script is running via slurm, set this argument as "True"')
parser.add_argument("--checkpoint_dir", type=str, 
                    default=None,required=False)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

args = parser.parse_args()


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # loading data and partitioning dataset 
    T1_images = loading_images_training(work_dir = args.root_dir, bids_dir = os.path.join(*[args.root_dir, '../', 'data', 'Data_Training_655']))
    mask_images = loading_images_training(work_dir = args.root_dir, bids_dir = os.path.join(*[args.root_dir, '../', 'data', 'Data_Training_655','mask']))
    T1_images = np.sort(T1_images)
    mask_images = np.sort(mask_images)
    dataset_partition, test_subjects = partitioning_dataset(T1_images, mask_images, args)
    print("===== DONE LOADING DATA =====")

    

    # setting UNet 
    net = UNet()

    # setting optimizer & scheduler
    if args.optimizer == 'SGD': 
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0)
    
    # attach network to cuda and do experiments
    if args.sbatch == 'True':
        net = nn.DataParallel(net)
    else: 
        net = nn.DataParallel(net, device_ids=args.gpus)
    net.to(f'cuda:{net.device_ids[0]}')

    # training network
    print("===== START TRAINING =====")
    result = {} 
    result['train_losses'] = []
    result['train_dice_losses'] = []
    result['train_focal_losses'] = []
    result['val_losses'] = []
    result['val_dice_lossses'] = []
    result['val_mask_dscs'] = []
    result['val_focal_losses'] = []
    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        train_loss, train_dice_loss, train_focal_loss = train(net, dataset_partition, optimizer, epoch, args)
        val_loss, val_dice_loss, val_mask_dsc,val_focal_loss = validation(net, dataset_partition, scheduler, args)
        te = time.time()
        torch.cuda.empty_cache()
        print('Epoch {}. Training Loss: {:2.2f}. Training Dice Loss {:2.2f}. Training FOCAL Loss {:2.2f}. Validation Loss: {:2.2f}. Validation Dice Loss {:2.2f}. Validation Focal Loss: {:2.2f}. Current learning rate {}. Took {:2.2f} sec'.format(epoch, train_loss, train_dice_loss, train_focal_loss, val_loss, val_dice_loss, val_focal_loss, optimizer.param_groups[0]['lr'], te-ts))
        print("DICE COEFFICIENT OF MASK IS {}".format(val_mask_dsc))
        
        # save result 
        result['train_losses'].append(train_loss)
        result['train_dice_losses'].append(train_dice_loss)
        result['train_focal_losses'].append(train_focal_loss)
        result['val_losses'].append(val_loss)
        result['val_dice_lossses'].append(val_dice_loss)
        result['val_mask_dscs'].append(val_mask_dsc)
        result['val_focal_losses'].append(val_focal_loss)
        
        # save checkpoint
        if epoch > 0:
            if val_mask_dsc >= max(result['val_mask_dscs'][:-1]):
                checkpoint_save(net, optimizer, scheduler, epoch, args)



        
        
    # inference
    print("===== START INFERENCE =====")
    test_loss, test_dice_loss, test_mask_dsc, test_focal_loss = test(net, dataset_partition, test_subjects, args)
    print('Test Loss: {:2.2f}. Test Dice Loss {:2.2f}. Test Focal Loss: {:2.2f}.'.format(test_loss, test_dice_loss, test_focal_loss))
    print("DICE COEFFICIENT OF MASK IS {}".format(test_mask_dsc))





