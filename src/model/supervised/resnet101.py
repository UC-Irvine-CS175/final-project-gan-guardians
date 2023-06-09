"""
This file is based on the following tutorial: 
https://learnopencv.com/t-sne-for-feature-visualization/
"""
import os
import random
import numpy as np
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))

import pandas as pd
import cv2
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from dataclasses import dataclass
from torchvision.models import resnet
from torchvision.models.resnet import Bottleneck
from torch.hub import load_state_dict_from_url
from mpl_toolkits.mplot3d import Axes3D
import wandb


from src.dataset.bps_datamodule import BPSDataModule
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)
from src.vis_utils import(
    show_label_batch
)

@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        acc_devices: Number of devices to use for training. Defaults to 1.
        
        device: Type of device used for training, checks for 'cuda', otherwise defaults to 'cpu'

        num_workers: Number of cpu cores dedicated to processing the data in the dataloader

        dm_stage: Set the partition of data depending to either 'train', 'val', or 'test'
                    However, our test images are not yet available.


    """
    data_dir:           str = root / 'data' / 'processed'
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_vis_dir:       str = root / 'models' / 'dummy_vis'
    save_models_dir:    str = root / 'models' / 'baselines'
    batch_size:         int = 4
    max_epochs:         int = 20
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 2
    dm_stage:           str = 'train'
    


class ResNet101(resnet.ResNet):
    """
    We will use the Resnet101 pretrained model for feature extraction and use with TSNE
    The original Resnet101 had 1000 output classes, which it used for classification.
    We will remove the final fully connected layer, and use the output of the final
    convolutional layer to investigate the features that the model has learned on the
    BPS dataset.
    """
    def __init__(self, num_classes=1000, pretrained=True, 
                 pre_weights=None, 
                 **kwargs):
        
        # Start with the standard resnet101
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            if pre_weights:
                self.load_state_dict(torch.load(os.path.join(root, 'src', 'model', 'supervised', 'weights', 'resnet_weights.pth')))
            else:
                state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                                                      progress=True)
                self.load_state_dict(state_dict)
 
    def _forward_impl(self, x):
        # Resnet101 originally takes 3 channel images for classification
        # of imagenet classes. To use Resnet101 for our 1 channel images,
        # we will need to convert our images to 3 channels using repeat()
        # This will copy the image on all 3 channels as a grayscale
        x = x.repeat(1, 3, 1, 1)

        # From here we implement a standard forward for ResNet101
        # pass through first convolutional layer
        x = self.conv1(x)
    
        # pass through batch norm
        x = self.bn1(x)
 
        # pass through relu activation function
        x = self.relu(x)

        # pass through max pooling layer
        x = self.maxpool(x)

        # pass through fully connected layer1
        x = self.layer1(x)
        
        # pass through fully connected layer2
        x = self.layer2(x)

        # pass through fully connected layer3
        x = self.layer3(x)

        # pass through fully connected layer4
        x = self.layer4(x)
 
        # Notice there is no forward pass through the original classifier.
        # Pass through average pooling layer
        x = self.avgpool(x)

        # flatten the output
        x = torch.flatten(x, 1)

        # linear transform of tensor (TO MATCH ORIGINAL RESNET FUNCTION)
        x = self.fc(x)

        return x
    
def main():
    wandb.login()

    # Initialize a BPSConfig object
    config = BPSConfig()
    
    # Fix random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(224, 224),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage="train")
    bps_datamodule.setup(stage="validate")

    ### PRETRAINED WEIGHTS ###
    pretrained_weights_path = None

    if pretrained_weights_path:
        model = ResNet101(num_classes=2, pretrained=True)
    # Initialize pretrained ResNet101 model
    else:
        model = ResNet101(pretrained=True)

    # we need to freeze parameters so that pretrained layers aren't modified
    for param in model.parameters():
        param.requires_grad = False


    num_classes = 2  # re-initialize the number of classes for the final transform 
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # define loss function and optimizer for training
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Move the model to GPU
    model.to(config.device)

    train_dataloader = bps_datamodule.train_dataloader()
    val_dataloader = bps_datamodule.val_dataloader()

    model.eval()

    
################################# WANDB IS COOL #############################################
    
    wandb.init(project="BPSResNet101",
               dir=config.save_vis_dir,
               config={
                   "architecture": "ResNet101",
                   "dataset": "BPS Microscopy Mouse Dataset",
                   "learning rate": "0.001",
                   "loss function": "Cross Entropy Loss",
                   "optimizer": "Adam"})
    


    num_epochs = config.max_epochs # CHANGE CONFIG


    if pretrained_weights_path == None:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data[0].to(config.device), data[1].to(config.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            wandb.log({"epoch": epoch, "loss": running_loss / len(train_dataloader)})

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            images, labels = data[0].to(config.device), data[1].to(config.device)
            outputs = model(images)
            predicted = (outputs == torch.max(outputs, dim=1, keepdim=True)[0]).float()
            result = []
            for r1, r2 in zip(predicted, labels):
                result.append(int(torch.equal(r1, r2)))
            correct += sum(result)
            total += config.batch_size
    wandb.log({"accuracy": correct / total})

    # Save the model
    save = False
    # CHANGE THE PATH WHEN SAVING NEW MODEL
    if save:
        path = os.path.join(root, 'src', 'model', 'supervised', 'weights', 'resnet_weights2.pth')
        torch.save(model.state_dict(), path)
    
    wandb.finish()
    
#############################################################################
# The ResNet101 network expects input images with 3 channels, so images with a different number of channels will need to be converted before they can be input to the network.

if __name__ == '__main__':
    main()