import torch
from torch import nn
import pytorch_lightning as pl


class Discriminator(nn.Module):
    """
    GAN discriminator for BPS classification
    """
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        """
        Discriminator architecture.
        """
        # Normal
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.LeakyRelu = nn.LeakyReLU(0.2)
        
        # Downsample
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

        self.Linear1 = nn.Linear(4 * 4 * 256, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):
        """
        Layers for determining the validity of an image.
        """
        # Normal
        x = self.conv1(x)
        x = self.LeakyRelu(x)
        
        # Downsample
        x = self.conv2(x)
        x = self.LeakyRelu(x)
        x = self.conv3(x)
        x = self.LeakyRelu(x)
        x = self.conv4(x)
        x = self.LeakyRelu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.Linear1(x)
        x = self.sigmoid(x)
        return x