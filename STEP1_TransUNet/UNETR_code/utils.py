import os
import pandas as pd
import numpy as np
import torch

import nibabel as nib
import hashlib
import json
import argparse 
import copy
from copy import deepcopy

def save_image(images, test_subjects, subj_idx, args=None):
    mkdir_nifti(args)        
    images = check_numpy(images)

    start = copy.copy(subj_idx)
    end = start + images.shape[0]

    for i, subject_T1 in enumerate(test_subjects[start:end]):
        head, tail = os.path.split(subject_T1)
        subject_name = tail.replace('.nii.gz','')
        
        image = images[i]
        image = image.reshape(image.shape[-3:])

        mask = npy2nifti(subject_T1, image)
        nib.save(mask, os.path.join(*[args.root_dir, 'inference_nifti', '%s_T1w_label-lesion_roi.nii.gz' % subject_name]))

    subj_idx = end 
    return subj_idx

def npy2nifti(nitfi_img_path, npy_img):
    nifti_img = nib.load(nitfi_img_path)
    header = nifti_img.header
    affine = nifti_img.affine
    nifti_mask = nib.Nifti1Image(npy_img, header=header, affine=affine)
    
    return nifti_mask

def check_numpy(images):
    if type(images) == torch.Tensor:
        images = images.numpy()
    return images 

def mkdir_nifti(args):   
    if os.path.isdir(os.path.join(*[args.root_dir, 'inference_nifti'])) == False:
        os.mkdir(os.path.join(*[args.root_dir, 'inference_nifti']))

def make_savedir(args):
    if os.path.isdir(os.path.join(*[args.root_dir, 'result'])) == False:
        os.mkdir(os.path.join(*[args.root_dir, 'result']))

def checkpoint_save(net, optimizer, scheduler, epoch, args):
    make_savedir(args)
    if os.path.isdir(os.path.join(*[args.root_dir, 'result', 'model'])) == False:
        os.mkdir(os.path.join(*[args.root_dir, 'result', 'model']))
    checkpoint_dir = os.path.join(*[args.root_dir, 'result', 'model', '%s.pth' % args.exp_name])

    torch.save({'net':net.module.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'scheduler': scheduler.state_dict(),
                    'epoch':epoch}, checkpoint_dir)
    print("Checkpoint is saved")
   
