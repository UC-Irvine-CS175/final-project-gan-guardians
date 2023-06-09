# nmist_gan.py

## 1. BPSConfig 
This is a data class that holds all the configuration options for the BPS Microscopy 
dataset. It defines various directory and file names as paths, batch sizes as the numbers 
of images per batch, max number of epochs, type of accelerators (to use for training), and
other parameters related to hardware training settings, as well as the dm_state which sets
the partition of data into either train, val, or test. Right now the number of epochs is 
100 but it might benefit from another 100 epochs.


## 2. Generator 
The Generator class is a subclass of nn.Module and represents the generator network of the 
GAN. It takes two arguments, a latent dimension and an image shape as inputs, initializes 
it, and builds a sequential model using linear layers, batch normalization, leaky ReLU 
activation, and a final Tanh activation.


## 3. Discriminator 
The Discriminator class is also a subclass of nn.Module. It represents the discriminator 
network of the GAN. It takes img_shape as an argument for image shape as the input, and 
then it builds a sequential model using linear layers, leaky ReLU activation, dropout, 
and a final sigmoid activation.


## 4. GAN
The GAN class is a subclass of L.LightningModule from PyTorch Lightning. It defined the GAN model by combining the generator and discriminator networks. The class includes methods for the forward pass, adversarial loss calculation, training step, optimizer configuration, and hooks for epoch-end events.

## 5. create_images
This function takes a GAN model, the number of images to generate, and the particle type as input. It generates the specified number of images by sampling noise from the latent space and passing it through the generator network. The generated images are saved to disk.

## 6. main
This is the main function that orchestrates the training and generation process. It instantiates a BPSConfig object to configure the dataset and model parameters. Then, it sets up the BPSDataModule for the Fe particle type, initializes WandB for logging, creates a GAN model for the Fe particle type, and trains the model using a PyTorch Lightning trainer. After training, it calls the create_images function to generate a single Fe particle image. Finally, it finishes logging with WandB.

