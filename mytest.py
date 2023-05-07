import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scripts.data import UPairLoader
from scripts.data import write_img,chw_to_hwc
# from scripts.model import gUNet
from scripts.New_model import UNet
import torch.cuda.amp as amp
import torch.optim as optim
import random
import torch.multiprocessing as multiprocessing
multiprocessing.set_sharing_strategy('file_system')
seed = 3407
random.seed(seed)
torch.manual_seed(seed)

## 超参 ##

dataset_dir = "./data/IRDehaze"
setting = {}
setting['patch_size'] = 256
setting['edge_decay'] = 0
setting['only_h_flip'] = False
setting['batch_size'] = 1
setting['valid_mode'] = "test"
setting["no_autocast"] = False
setting['num_epochs'] = 10
setting['save_freq'] = 5
setting['print_freq'] = 5
setting['save_dir'] ="/root/The_Project_of_Zhou/CV/Project/experience/unet_ir_v_rgb3/"
setting["lr"] = 1e-5
setting["weight_decay"] = 0 
setting["result_dir"] = "/root/The_Project_of_Zhou/CV/Project/experience/unet_ir_v_rgb_3/output"
setting["load_model_path"] = "/root/The_Project_of_Zhou/CV/Project/experience/unet_ir_v_rgb_2/models/model_490.pth"
if not os.path.isdir(os.path.join(setting["result_dir"], 'imgs')):
    os.makedirs(os.path.join(setting["result_dir"], 'imgs'))
def test(dataset_dir):
    criterion = nn.L1Loss()
    # Load the saved model
    checkpoint_path = setting["load_model_path"]
    network = UNet(n_channels=3, n_classes=3)
    network.load_state_dict(torch.load(checkpoint_path))

    # Set the device to use for testing
    device = torch.device("cuda:6")
    network.to(device)

    # Define the test dataset and data loader
    test_dataset = UPairLoader(dataset_dir, 'test', 'test',
                              setting['patch_size'])
    test_loader = DataLoader(test_dataset,
                             batch_size=setting['batch_size'],
                             num_workers=0,
                             pin_memory=True)

    # Set the network to evaluation mode
    network.eval()

    # Loop over the test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            filename = batch['filename'][0]
            # Load the source and target images from the batch
            source_img = batch['source'].to(device)
            target_img = batch['target'].to(device)
            
            # Forward pass
            output = network(source_img[:,0:3,:,:],source_img[:,3:5,:,:])
            
            # Calculate the loss
            loss = criterion(output, target_img)
            output = output * 0.5 + 0.5
            
            # Print some information about the test progress
            if batch_idx % setting['print_freq'] == 0:
                print(f'Test Batch: {batch_idx}/{len(test_loader)}\t'
                      f'Loss: {loss.item():.6f}')

    # Calculate the overall test loss
            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(setting["result_dir"], 'imgs', filename), out_img)
    # test_loss /= len(test_loader)
    # print(f'Test Loss: {test_loss:.6f}')
if __name__ == "__main__":
    test(dataset_dir)