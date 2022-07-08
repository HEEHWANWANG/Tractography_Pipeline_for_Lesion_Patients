import os 
import glob
import numpy as np 
import nibabel as nib
import torch  

import monai
from monai.transforms import AddChannel, Compose, ScaleIntensity, NormalizeIntensity, ToTensor, ToNumpy
from monai.data import ImageDataset



def loading_images(work_dir, bids_dir):
    images_path = []

    os.chdir(bids_dir)
    subject_list = glob.glob('*')

    for subject in subject_list:
        subject_dir = os.path.abspath(subject)
        subject_file = os.path.join(subject_dir,'%s_T1w.nii.gz' % subject) 
        images_path.append(subject_file)

    os.chdir(work_dir)

    return images_path



def loading_images_training(work_dir, bids_dir):
    images_path = []
    
    os.chdir(bids_dir)

    images = glob.glob('*.nii.gz')
    for subj_image in images:
        images_path.append(os.path.join(os.path.abspath(bids_dir), subj_image))
    
    os.chdir(work_dir)

    return images_path



def partitioning_dataset(T1_images, mask_images, args = None):

    transform = Compose([ScaleIntensity(),
                         AddChannel(),
                         ToTensor()])

    seg_transform = Compose([AddChannel(),
                             ToTensor()])

    num_total = len(T1_images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)    

    # image and label information of train
    images_train = T1_images[:num_train]
    masks_train = mask_images[:num_train]

    # image and label information of valid
    images_val = T1_images[num_train:num_train+num_val]
    masks_val = mask_images[num_train:num_train+num_val]

    # image and label information of test
    images_test = T1_images[num_train+num_val:]
    masks_test = mask_images[num_train+num_val:]

    train_set = ImageDataset(image_files=images_train, seg_files=masks_train, transform=transform, seg_transform=one_hot_encoding(seg_transform))
    val_set = ImageDataset(image_files=images_val, seg_files=masks_val, transform=transform, seg_transform=one_hot_encoding(seg_transform))
    test_set = ImageDataset(image_files=images_test, seg_files=masks_test, transform=transform, seg_transform=one_hot_encoding(seg_transform))
    #train_set = ImageDataset(image_files=images_train, seg_files=masks_train, transform=transform, seg_transform=seg_transform)
    #val_set = ImageDataset(image_files=images_val, seg_files=masks_val, transform=transform, seg_transform=seg_transform)
    #test_set = ImageDataset(image_files=images_test, seg_files=masks_test, transform=transform, seg_transform=seg_transform)

    print(len(images_train))
    print(len(images_val))
    print(len(images_test))

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set 

    return partition, images_test


class one_hot_encoding(object):
    def __init__(self, transform):
        self.transform = transform 

    def __call__(self, mask):
        mask = self.transform(mask)
        mask0 = torch.ones(mask.size()) - mask 
        mask1 = mask
        new_mask = torch.cat([mask0,mask1],dim=0)
        return new_mask 



