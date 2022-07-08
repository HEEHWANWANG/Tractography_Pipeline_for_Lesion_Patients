"""이미지 슬라이스는 train loader가 불러온 이후에, 모델로 feeding하기 이전에"""
import numpy as np 
import torch

from loss_functions import *
from utils import *

from tqdm import tqdm
import gc


def train(net, partition, optimizer, epoch, args):
    """
    training strategy: 
    1. image slices including lesion -> whole brain image slices -> image slices including lesion...
    2. skip image slices with only background pixel value (loss calculating and backprogating are done only when the number of unique pixel values are over 1)
    """
    train_loader = torch.utils.data.DataLoader(partition['train'],
                                                batch_size=args.train_batch_size,
                                                shuffle=True,
                                                num_workers=16)

    net.train()
    
    result = {}
    result['total_loss'] = []
    result['dice_loss'] = []
    result['focal_loss'] = []

    for i, data in enumerate(train_loader):
        

        image, mask = data

        x, y, z = image.size()[-3], image.size()[-2], image.size()[-1]
        total_losses = 0
        focal_losses = 0
        dice_losses = 0

        new_horizontal_mask = torch.tensor([])
        ground_truth_mask = torch.tensor([])
        """
        # sagittal slicing
        x_check = 0 
        for x_i in range(x):
            if x_i == 128:
                sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
                predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))

  
                loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask)
                #loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=args.cross_entropy_weights)
                loss.backward()
                losses += loss
                dices += dice
                x_check += 1
                
                pred_mask = torch.argmax(predicted.cpu(), dim=1)

        if i == 1:
            np.save(os.path.join(*[args.root_dir,'../','train_sagittal_mask_{}.npy'.format(i)]), pred_mask.numpy()) 
        """
        dice_loss_fn = DiceLoss()
        focal_loss_fn = FocalTverskyLoss()

        # horizontal slicing
        y_check = 0  
        for y_i in range(y):
            optimizer.zero_grad()
            horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
            if epoch % 4 != 0:
                if len(torch.unique(horizontal_mask[:,1,:,:])) > 1:
                    predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                    predicted = torch.softmax(predicted, dim=1)
                    dice_loss, _ = dice_loss_fn(predicted.cpu(), horizontal_mask)
                    focal_loss = focal_loss_fn(predicted.cpu(), horizontal_mask)
                    loss = 0.5 * dice_loss + 0.5 * focal_loss
                
                    loss.backward()
                    optimizer.step()
                    
                    total_losses += loss.item()
                    dice_losses += dice_loss.item()
                    focal_losses += focal_loss.item()

                    y_check += 1
            else:
                if len(torch.unique(horizontal_image[:,0,:,:])) > 1:
                    predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                    predicted = torch.softmax(predicted, dim=1)
                    dice_loss, _ = dice_loss_fn(predicted.cpu(), horizontal_mask)
                    focal_loss = focal_loss_fn(predicted.cpu(), horizontal_mask)
                    loss = 0.5 * dice_loss + 0.5 * focal_loss
                    
                    loss.backward()
                    optimizer.step()
                        
                    total_losses += loss.item()
                    dice_losses += dice_loss.item()
                    focal_losses += focal_loss.item()

                    y_check += 1            
             
                    
        """
        # coronal slicing
        z_check = 0  
        for z_i in range(z):
            coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
            predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
            
            if len(torch.unique(coronal_mask)) > 1:
                loss, dice = loss_and_dice(predicted.cpu(), horizontal_mask)
                #loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=args.cross_entropy_weights)
                loss.backward()
                losses += loss
                dices += dice
                z_check += 1

        if i == 1:
            np.save(os.path.join(*[args.root_dir,'../','train_coronal_mask_{}.npy'.format(i)]), new_coronal_mask.numpy()) 
       
        


        losses = losses / (x_check + y_check + z_check)
        dices = dices / (x_check + y_check + z_check)
        """
        
        total_losses = total_losses / y_check
        dice_losses = dice_losses / y_check
        focal_losses = focal_losses / y_check

        print('TOTAL LOSS {}. FOCAL LOSS {}. DICE LOSS {}'.format(total_losses,focal_losses,dice_losses))

        result['total_loss'].append(total_losses)
        result['dice_loss'].append(dice_losses)
        result['focal_loss'].append(focal_losses)

    return np.mean(result['total_loss']), np.mean(result['dice_loss']), np.mean(result['focal_loss'])


def validation(net, partition, scheduler, args):
    """
    validation strategy: 
    1. slice-wise predicting -> concatenating -> evaluation
    2. image slices with only background pixel value are concatenating directly (this slices are fed into neural network)
    """
    val_loader = torch.utils.data.DataLoader(partition['val'],
                                                batch_size=args.val_batch_size,
                                                shuffle=False,
                                                num_workers=16)

    net.eval()

    result = {}
    result['total_loss'] = []
    result['dice_loss'] = []
    result['focal_loss'] = []



    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask = data

            x, y, z = image.size()[-3], image.size()[-2], image.size()[-1]


            new_horizontal_mask = torch.tensor([])
            ground_truth_mask = torch.tensor([])
            """
            # sagittal slicing 
            for x_i in range(x):
                if x_i == 128: 
                    sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
                    predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))
                    loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=[1,1])
                    losses += loss
                    dices += dice

                    pred_mask = torch.argmax(predicted.cpu(), dim=1)

            if i == 1:
                np.save(os.path.join(*[args.root_dir,'../','val_sagittal_mask_{}.npy'.format(i)]), pred_mask.numpy()) 
            """
            
            dice_loss_fn = DiceLoss(mode='eval')
            focal_loss_fn = FocalTverskyLoss()

            # horizontal slicing 
            
            for y_i in range(y):
                horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
                if len(torch.unique(horizontal_image[:,0,:,:])) > 1:
                    predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                    predicted = torch.softmax(predicted, dim=1)
                    new_horizontal_mask = torch.cat([new_horizontal_mask, predicted.unsqueeze(3).cpu()], dim=3)
                else:
                    new_horizontal_mask = torch.cat([new_horizontal_mask, horizontal_mask.unsqueeze(3).cpu()], dim=3)
                ground_truth_mask = torch.cat([ground_truth_mask, horizontal_mask.unsqueeze(3)], dim=3)
            
            dice_loss, ds_coef = dice_loss_fn(new_horizontal_mask, ground_truth_mask)
            focal_loss = focal_loss_fn(new_horizontal_mask, ground_truth_mask)
            loss = 0.5 * dice_loss + 0.5 * focal_loss
            

            result['total_loss'].append(loss.item())
            result['dice_loss'].append(dice_loss.item())
            result['focal_loss'].append(focal_loss.item())


            if i == 0:
                np.save(os.path.join(*[args.root_dir,'images','val_horizontal_mask_{}.npy'.format(i)]), new_horizontal_mask.detach().numpy()) 
                np.save(os.path.join(*[args.root_dir,'images','val_horizontal_GTmask_{}.npy'.format(i)]), ground_truth_mask.detach().numpy())
                del new_horizontal_mask
                del ground_truth_mask
                gc.collect()
                print('SAVE IMAGE DONE')




            """
    
            # coronal slicing 
            for z_i in range(z):
                coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
                predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=[1,1])
                losses += loss
                dices += dice

            if i == 1:
                np.save(os.path.join(*[args.root_dir,'../','val_coronal_mask_{}.npy'.format(i)]), new_coronal_mask.numpy()) 

            """

    scheduler.step()
        
    return np.mean(result['total_loss']), np.mean(result['dice_loss']), ds_coef[1], np.mean(result['focal_loss'])


def test(net, partition,  test_subjects, args):
    test_loader = torch.utils.data.DataLoader(partition['test'],
                                                batch_size=args.test_batch_size,
                                                shuffle=False,
                                                num_workers=16)

    net.eval()

    result = {}
    result['total_loss'] = []
    result['dice_loss'] = []
    result['focal_loss'] = []


    subj_idx = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # new lesion mask per mini-batches
            new_horizontal_mask = torch.tensor([])
            ground_truth_mask = torch.tensor([])
            
            
            image, mask = data


            """
            # sagittal slicing 
            for x_i in range(x):
                if x_i == 128: 
                    sagittal_image, sagittal_mask = image[:, :, x_i, :, :], mask[:, :, x_i, :, :]
                    predicted = net(sagittal_image.to(f'cuda:{net.device_ids[0]}'))
                    loss, dice = loss_and_dice(predicted.cpu(), sagittal_mask, weight=[1,1])
                    losses += loss
                    dices += dice

                    pred_mask = torch.argmax(predicted.cpu(), dim=1)

            if i == 1:
                np.save(os.path.join(*[args.root_dir,'../','val_sagittal_mask_{}.npy'.format(i)]), pred_mask.numpy()) 
            """
            
            dice_loss_fn = DiceLoss(mode='eval')
            focal_loss_fn = FocalTverskyLoss()

            # horizontal slicing 
            y = image.size()[-2]
            for y_i in range(y):
                horizontal_image, horizontal_mask = image[:, :, :, y_i, :], mask[:, :, :, y_i, :]
                if len(torch.unique(horizontal_image[:,0,:,:])) > 1:
                    predicted = net(horizontal_image.to(f'cuda:{net.device_ids[0]}'))
                    predicted = torch.softmax(predicted, dim=1)
                    new_horizontal_mask = torch.cat([new_horizontal_mask, predicted.unsqueeze(3).cpu()], dim=3)
                else:
                    new_horizontal_mask = torch.cat([new_horizontal_mask, horizontal_mask.unsqueeze(3).cpu()], dim=3)
                ground_truth_mask = torch.cat([ground_truth_mask, horizontal_mask.unsqueeze(3)], dim=3)
            
            dice_loss, ds_coef = dice_loss_fn(new_horizontal_mask, ground_truth_mask)
            focal_loss = focal_loss_fn(new_horizontal_mask, ground_truth_mask)
            loss = 0.5 * dice_loss + 0.5 * focal_loss
            

            """
            # coronal slicing 
            for z_i in range(z):
                coronal_image, coronal_mask = image[:, :, :, :, z_i], mask[:, :, :, :, z_i]
                predicted = net(coronal_image.to(f'cuda:{net.device_ids[0]}'))
                loss, dice = loss_and_dice(predicted.cpu(), coronal_mask, weight=[1,1])
                losses += loss
                dices += dice

                # accumalting slices to reconstruct 3D images
                pred_coronal_mask = torch.argmax(predicted.cpu(), dim=1, keepdim=True)
                new_coronal_mask = torch.cat([new_coronal_mask, pred_coronal_mask.unsqueeze(2)],dim=2)  
            

            losses = losses / (x + y + z)
            dices = dices / (x + y + z)
            
            result['loss'].append(losses.item())
            result['dice_coeff'].append(dices.item())
            #result['sagittal_mask'] = torch.cat([result['sagittal_mask'], new_sagittal_mask])
            result['horizontal_mask'] = torch.cat([result['horizontal_mask'], new_horizontal_mask])
           #result['coronal_mask'] = torch.cat([result['coronal_mask'], new_coronal_mask])

    """
            result['total_loss'].append(loss.item())
            result['dice_loss'].append(dice_loss.item())
            result['focal_loss'].append(focal_loss.item())


            new_horizontal_mask = torch.argmax(new_horizontal_mask, dim = 1)
            subj_idx = save_image(new_horizontal_mask.detach().numpy(), test_subjects, subj_idx, args)
            del new_horizontal_mask
            del ground_truth_mask
            gc.collect()


    return np.mean(result['total_loss']), np.mean(result['dice_loss']), ds_coef[1], np.mean(result['focal_loss'])

