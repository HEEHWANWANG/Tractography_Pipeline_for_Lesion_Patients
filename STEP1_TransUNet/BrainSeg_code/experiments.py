"""이미지 슬라이스는 train loader가 불러온 이후에, 모델로 feeding하기 이전에"""
import numpy as np 
import torch

from loss_functions import *
from utils import *



def train(net, partition, optimizer, args):
    train_loader = torch.utils.data.DataLoader(partition['train'],
                                                batch_size=args.train_batch_size,
                                                shuffle=False,
                                                num_workers=16)

    net.train()
    
    result = {}
    result['loss'] = []
    result['dice_coeff'] = []

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        image, mask = data

        x, y, z = image.size()[-3], image.size()[-2], image.size()[-1]
        losses = 0
        dices = 0

        new_sagittal_mask = torch.tensor([])
        new_horizontal_mask = torch.tensor([])
        new_coronal_mask = torch.tensor([])

        # sagittal slicing
        x_check = 0 
        for x_i in range(x):
            sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
            predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))
  
            loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask)
            #loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=args.cross_entropy_weights)
            loss.backward()
            losses += loss
            dices += dice
            x_check += 1


        # horizontal slicing 
        y_check = 0
        for y_i in range(y):
            horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
            predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
            
            loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask)
            #loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask, weight=args.cross_entropy_weights)
            loss.backward()
            losses += loss
            dices += dice
            y_check += 1
            

        # coronal slicing
        z_check = 0  
        for z_i in range(z):
            coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
            predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
            
            loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask)
            #loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=args.cross_entropy_weights)
            loss.backward()
            losses += loss
            dices += dice
            z_check += 1


        optimizer.step()

        losses = losses / (x_check + y_check + z_check)
        dices = dices / (x_check + y_check + z_check)

        result['loss'].append(losses.item())
        result['dice_coeff'].append(dices.item())
        
    return np.mean(result['loss']), np.mean(result['dice_coeff'])


def validation(net, partition, scheduler, args):
    val_loader = torch.utils.data.DataLoader(partition['val'],
                                                batch_size=args.val_batch_size,
                                                shuffle=False,
                                                num_workers=16)

    net.eval()

    result = {}
    result['loss'] = []
    result['dice_coeff'] = []



    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask = data

            x, y, z = image.size()[-3], image.size()[-2], image.size()[-1]
            losses = 0
            dices = 0

            new_sagittal_mask = torch.tensor([])
            new_horizontal_mask = torch.tensor([])
            new_coronal_mask = torch.tensor([])

            # sagittal slicing 
            for x_i in range(x):
                sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
                predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=[1,1])
                losses += loss
                dices += dice


            # horizontal slicing 
            for y_i in range(y):
                horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
                predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask, weight=[1,1])
                losses += loss
                dices += dice


            # coronal slicing 
            for z_i in range(z):
                coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
                predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=[1,1])
                losses += loss
                dices += dice


            losses = losses / (x + y + z)
            dices = dices / (x + y + z)

            result['loss'].append(losses.item())
            result['dice_coeff'].append(dices.item())
            
    scheduler.step()

    return np.mean(result['loss']), np.mean(result['dice_coeff'])

def test(net, partition, test_subjects,args):
    test_loader = torch.utils.data.DataLoader(partition['test'],
                                                batch_size=args.test_batch_size,
                                                shuffle=False,
                                                num_workers=16)

    net.eval()

    result = {}
    result['loss'] = []
    result['dice_coeff'] = []
    result['sagittal_mask'] = torch.tensor([])
    result['horizontal_mask'] = torch.tensor([])
    result['coronal_mask'] = torch.tensor([])

    with torch.no_grad():
        for data in test_loader:
            image, mask = data

            x, y, z = image.size()[-3], image.size()[-2], image.size()[-1]
            losses = 0
            dices = 0
            
            # new lesion mask per mini-batches
            new_sagittal_mask = torch.tensor([])
            new_horizontal_mask = torch.tensor([])
            new_coronal_mask = torch.tensor([])
            
            # sagittal slicing 
            for x_i in range(x):
                sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
                predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=[1,1])
                losses += loss
                dices += dice
            
                # accumalting slices to reconstruct 3D images
                pred_sagittal_mask = torch.argmax(predicted.cpu(), dim=1, keepdim=True)
                new_sagittal_mask = torch.cat([new_sagittal_mask, pred_sagittal_mask.unsqueeze(2)],dim=2)  


            # horizontal slicing 
            for y_i in range(y):
                horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
                predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask, weight=[1,1])
                losses += loss
                dices += dice

                # accumalting slices to reconstruct 3D images
                pred_horizontal_mask = torch.argmax(predicted.cpu(), dim=1, keepdim=True)
                new_horizontal_mask = torch.cat([new_horizontal_mask, pred_horizontal_mask.unsqueeze(3)],dim=3)  


            # coronal slicing 
            for z_i in range(z):
                coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
                predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=[1,1])
                losses += loss
                dices += dice

                # accumalting slices to reconstruct 3D images
                pred_coronal_mask = torch.argmax(predicted.cpu(), dim=1, keepdim=True)
                new_coronal_mask = torch.cat([new_coronal_mask, pred_coronal_mask.unsqueeze(4)],dim=4)  


            losses = losses / (x + y + z)
            dices = dices / (x + y + z)

            result['loss'].append(losses.item())
            result['dice_coeff'].append(dices.item())
            result['sagittal_mask'] = torch.cat([result['sagittal_mask'], new_sagittal_mask])
            result['horizontal_mask'] = torch.cat([result['horizontal_mask'], new_horizontal_mask])
            result['coronal_mask'] = torch.cat([result['coronal_mask'], new_coronal_mask])

    save_image(result['sagittal_mask'], test_subjects, view='sagittal', args=args)
    save_image(result['horizontal_mask'], test_subjects, view='horizontal', args=args) 
    save_image(result['coronal_mask'], test_subjects, view='coronal', args=args)     

    return np.mean(result['loss']), np.mean(result['dice_coeff'])
