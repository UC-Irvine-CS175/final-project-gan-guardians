# CS175 Final Project
## Gan Guardians
- Ye Aung
- Arthur Hsu
- Warren Wei Leu
- Tyler Fermanian
- Tariq Brown

# Getting Started
## Setting Up the Environment
Go to `setup/environments` and follow the directions to setup your conda environment based on whether your machine is gpu or cpu enabled using Miniconda, a lightweight form of Anaconda.

## Downloading Data From AWS
Run the `main()` function from bps_gan.py to download the images from AWS as .tiff files.
Call the `separate_particle_type()` function to separate different particle types into their own .csv files.

## Training the GAN model
- Ensure that the BPSConfig has all the right paths and values(paths to the metadata .csv files are stored, number of epochs, number of batches).  
- In main, comment/uncomment lines to define the type of GAN to train (Fe, X-ray, or both).  
- Set the width and height of the images, or use the default (128x128).  
- Optionally set a path to a checkpoint file (.ckpt) to resume training of an already started (but paused) GAN.  
- Uncomment line 392 `prepare_data()` function before the first run of the file (make sure this line only runs once, or images will be downloaded every file run).  
- Run `bps_gan.py` to train the GAN.  
