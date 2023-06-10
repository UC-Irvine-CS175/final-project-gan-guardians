import os
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from dataclasses import dataclass
import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import wandb
from src.dataset.bps_datamodule import BPSDataModule


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
    data_dir:           str = os.path.join(root,'data','processed')
    gen_image_save_dir: str = os.path.join(root, 'data', 'generated')
    
    fe_img_save_dir:    str = os.path.join(root, 'data', 'generated', 'Fe')
    xray_img_save_dir:  str = os.path.join(root, 'data', 'generated', 'Xray')
    
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    fe_train_meta:      str = 'meta_dose_hi_hr_4_post_exposure_train_Fe.csv'
    xray_train_meta:    str = 'meta_dose_hi_hr_4_post_exposure_train_X_ray.csv'

    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    fe_test_meta:      str = 'meta_dose_hi_hr_4_post_exposure_test_Fe.csv'
    xray_test_meta:    str = 'meta_dose_hi_hr_4_post_exposure_test_X_ray.csv'
    
    save_vis_dir:       str = os.path.join(root, 'models', 'dummy_vis')
    save_models_dir:    str = os.path.join(root, 'models', 'baselines')
    batch_size:         int = 64
    max_epochs:         int = 220
    accelerator:        str = 'auto'
    acc_devices:        int = 1
    device:             str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers:        int = 12
    dm_stage:           str = 'train'


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape


        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )


    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        gen_image_save_dir,
        train_image_save: bool = False,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 256, 
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, 
                                   img_shape=data_shape)

        self.discriminator = Discriminator(img_shape=data_shape)
        
        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        # Save directory of the generated images after each epoch.
        self.gen_image_save_dir = gen_image_save_dir
        self.train_image_save = train_image_save

        # Save height and width
        self.height = height
        self.width = width
    
    
    def forward(self, z):
        return self.generator(z)


    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        if torch.cuda.is_available(): self.toggle_optimizer(optimizer=optimizer_g, optimizer_idx=0)
        else: self.toggle_optimizer(optimizer=optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        
        # Convert the grid to a numpy array
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        # Log the image using wandb
        wandb.log({"generated_images": [wandb.Image(grid_np, caption="Generated Images")]})

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        wandb.log({"g_loss": g_loss})
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        if torch.cuda.is_available(): self.untoggle_optimizer(optimizer_idx=0)
        else: self.untoggle_optimizer(optimizer=optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        if torch.cuda.is_available(): self.toggle_optimizer(optimizer=optimizer_d, optimizer_idx=1)
        else: self.toggle_optimizer(optimizer=optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        wandb.log({"d_loss": d_loss})
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        if torch.cuda.is_available(): self.untoggle_optimizer(optimizer_idx=1)
        else: self.untoggle_optimizer(optimizer=optimizer_d)


    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return opt_g, opt_d


    def on_train_epoch_end(self):
        # # Save the image
        if self.train_image_save:
            z = self.validation_z.type_as(self.generator.model[0].weight)
            sample_imgs = self(z)

            img_transform = transforms.ToPILImage()
            img = img_transform(sample_imgs[0])
            img.save(os.path.join(self.gen_image_save_dir, 
                                f'generated_bps_epoch_{self.current_epoch}.jpeg'))
            return super().on_train_epoch_end()


    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


    def create_images(self, num_images, particle_type):
        """
        Produce num_images amount of images.
        """
        
        for i in range(num_images):
            z = torch.randn(128, self.hparams.latent_dim)
            generated_imgs = self(z)
            sample_img = generated_imgs[0]
        
            # Convert the grid to a numpy array
            sample_img_np = sample_img.permute(1, 2, 0).cpu().detach().numpy()
            plt.figure()
            plt.imshow(sample_img_np)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'gen_{particle_type}_data_{i}.jpeg', bbox_inches='tight', pad_inches=0)
            #cv2.imwrite(f'gen_{particle_type}_data_{i}.jpeg', sample_img_np)
            
            
def define_gan(
        config: BPSConfig, 
        particle_type: str, 
        checkpoint_path: str = None
    ) -> tuple[GAN, BPSDataModule]:
    """
    Train a GAN to generate Fe radiated cell images
    Args:
        config: BPSConfig containing necessary information like epoch and batch.
        particle_type: Fe or Xray (depending on what the GAN should produce).
        checkpoint_path: path to checkpoint file to continue training previous GAN.

    Returns:
        model: the GAN model class.
        bps_datamodule: the bps_datamodule class.
    """
    # Instantiate BPSDataModule with Fe images.
    if particle_type == "Fe":
        bps_datamodule = BPSDataModule(train_csv_file=config.fe_train_meta,
                                       train_dir=config.data_dir,
                                       val_csv_file=config.fe_test_meta,
                                       val_dir=config.data_dir,
                                       resize_dims=(128, 128),
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers)
    # Instantiate BPSDataModule with Xray images.
    elif particle_type == "Xray":
        bps_datamodule = BPSDataModule(train_csv_file=config.xray_train_meta,
                                       train_dir=config.data_dir,
                                       val_csv_file=config.xray_test_meta,
                                       val_dir=config.data_dir,
                                       resize_dims=(128, 128),
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers)
    # Instantiate BPSDataModule with both Fe and Xray images.
    else:
        bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                       train_dir=config.data_dir,
                                       val_csv_file=config.val_meta_fname,
                                       val_dir=config.data_dir,
                                       resize_dims=(128, 128),
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage=config.dm_stage)

    # Create a GAN model.
    model = GAN(1, bps_datamodule.resize_dims[0],
                bps_datamodule.resize_dims[1],
                batch_size=config.batch_size,
                gen_image_save_dir=config.gen_image_save_dir)
    
    # Load checkpoint if one was given.
    if checkpoint_path != None:
        model = model.load_from_checkpoint(checkpoint_path)

    # Return the defined GAN model.
    return model, bps_datamodule
    
    
def main():
    config = BPSConfig()
    wandb.init(project="MNIST-GAN",
               dir=config.save_vis_dir,
               #mode="disabled",
               name="Fe-train-800-epochs",
               config=
               {
                   "architecture": "MNIST GAN",
                   "dataset": "BPS Microscopy"
               })
    
    # Checkpoint for continuing training.
    checkpoint = os.path.join(root, 'models', 'weights', 'epoch=480-step=11200.ckpt')
    
    # Uncomment this line to train a GAN to generate Fe radiated images.
    model, bps_datamodule = define_gan(config, "Fe", checkpoint)

    # Uncomment this line to train a GAN to generate Xray radiated images.
    # model, bps_datamodule = define_gan(config, "Xray", checkpoint)
    
    # Create a PyTorch Lightning trainer.
    trainer = L.Trainer(
        accelerator=config.accelerator,
        devices=config.acc_devices,
        max_epochs=config.max_epochs,
    )
    # Train the model.
    trainer.fit(model, bps_datamodule.train_dataloader())
    wandb.finish()


if __name__ == '__main__':
    main()
