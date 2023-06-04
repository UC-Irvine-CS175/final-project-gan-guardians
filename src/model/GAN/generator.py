import torch
from torch import nn
import pytorch_lightning as pl

class Generator(nn.Module):
    """
    GAN generator for BPS classification
    """
    def __init__(self, latent_dim):
        super().__init__()
        """
        Generator architecture.
        """
        self.latent_dim = latent_dim
        self.Linear1 = nn.Linear(latent_dim, 4 * 4 * 256)
        self.LeakyRelu = nn.LeakyReLU(0.2)
        self.reshape = nn.Unflatten(4 * 4 * 256, (256,4,4))
        self.upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x: torch.Tensor):
        """
        Layers for generating an image.
        """
        x = self.Linear1(x)
        x = self.LeakyRelu(x)
        x = self.reshape(x)
        x = self.upsample1(x)
        x = self.LeakyRelu(x)
        x = self.upsample2(x)
        x = self.LeakyRelu(x)
        x = self.upsample3(x)
        x = self.LeakyRelu(x)
        x = self.conv1(x)
        x = self.tanh(x)
        return x