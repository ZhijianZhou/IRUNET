import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scripts.data import UPairLoader
# from scripts.model import gUNet
from scripts.New_model import UNet
import torch.cuda.amp as amp
import torch.optim as optim
import random
import torch.multiprocessing as multiprocessing
from tensorboardX import SummaryWriter
from scripts.ssmi import MS_SSIM_L1_LOSS
from scripts.utils import AverageMeter

multiprocessing.set_sharing_strategy('file_system')
seed = 3407
random.seed(seed)
torch.manual_seed(seed)

## 超参 ##

dataset_dir = "./data/IRDehaze"
setting = {}
setting["log_dir"] = "/root/The_Project_of_Zhou/CV/Project/experience/unet_ir_v_rgb_3/tensorboards_log"
setting['save_dir'] ="/root/The_Project_of_Zhou/CV/Project/experience/unet_ir_v_rgb_3/models"
setting['device'] = "cuda:0"
setting['patch_size'] = 256
setting['edge_decay'] = 0
setting['only_h_flip'] = False
setting['batch_size'] = 16
setting['valid_mode'] = "test"
setting["no_autocast"] = False
setting['num_epochs'] = 500
setting['save_freq'] = 5
setting['print_freq'] = 5
setting["lr"] = 1e-4
setting["weight_decay"] = 0 

if not os.path.isdir(setting["log_dir"]):
    os.makedirs(setting["log_dir"])
if not os.path.isdir(setting['save_dir']):
    os.makedirs(setting['save_dir'])
def train(dataset_dir):
## 损失函数
    writer = SummaryWriter(log_dir=setting["log_dir"])
    criterion = nn.L1Loss()
    network = UNet(n_channels=3,n_classes=3)
    optimizer = optim.Adam(network.parameters(), lr=setting['lr'], weight_decay=setting['weight_decay'])
    
    # Set the device to use for training
    device = torch.device(setting['device'] )
    network.to(device)
    
    # Define the training and validation datasets and data loaders
    dataset_dir = dataset_dir
    train_dataset = UPairLoader(dataset_dir, 'train', 'train', 
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = UPairLoader(dataset_dir, 'test', setting['valid_mode'], 
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=0,
                            pin_memory=True)
    
    # Training loop
    scaler = amp.GradScaler()
    decay_rate = 0.90
    decay_steps = 10
    for epoch in range(setting['num_epochs']):
        # Set the network to train mode
        lr = setting["lr"] * (decay_rate ** (epoch / decay_steps))
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=setting['weight_decay'])
        network.train()

        # Loop over the training set
        for batch_idx, batch in enumerate(train_loader):
            # Load the source and target images from the batch
            source_img = batch['source'].to(device)
            target_img = batch['target'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            with autocast(setting["no_autocast"]):
                output = network(source_img[:,0:3,:,:],source_img[:,3:5,:,:])

                # Calculate the loss
                loss = criterion(output, target_img)
                
            # Backward pass
           
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training Loss', loss.item(), global_step=global_step)
            # Print some information about the training progress
            if batch_idx % setting['print_freq'] == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\t'
                      f'Loss: {loss.item():.6f}')
            
                
        
        # Evaluation loop
        network.eval()
        val_loss = 0
        with torch.no_grad():
            PSNR = AverageMeter()
            for batch_idx, batch in enumerate(val_loader):
                source_img = batch['source'].to(device)
                target_img = batch['target'].to(device)
                
                output = network(source_img[:,0:3,:,:],source_img[:,3:5,:,:])
                val_loss += criterion(output, target_img).item()
                mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
                psnr = 10 * torch.log10(1 / mse_loss).mean()
                PSNR.update(psnr.item(), source_img.size(0))
        avg_psnr = PSNR.avg
        
        writer.add_scalar('valid_psnr', avg_psnr, epoch)
        writer.add_scalar('lr', lr, global_step=epoch)
        val_loss /= len(val_loader)
        writer.add_scalar('Val Loss', val_loss, global_step=epoch)
        print(f'Validation Loss: {val_loss:.6f}')
        print('valid_psnr', avg_psnr, epoch)
        print("learning_rate:",lr)
        
        # Save the model checkpoint
        if (epoch+1) % setting['save_freq'] == 0:
            checkpoint_path = os.path.join(setting['save_dir'], f'model_{epoch+1}.pth')
            torch.save(network.state_dict(), checkpoint_path)
if __name__ == "__main__":
    train(dataset_dir)
