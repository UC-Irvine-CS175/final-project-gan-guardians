# bps_datamodule.py

The bps_datamodule.py file contains a PyTorch Lightning DataModule called BPSDataModule 
that is used to organize and provide access to the training, validation, and testing 
data for a BPS microscopy dataset.

## BPSDataModule class -
This class inherits from pl.LightningDataModule, which is a PyTorch Lightning class for 
organizing and providing data to the model. The constructor __init__ takes various 
arguments for configuring the data module such as file paths, directories, data resizing 
dimensions, batch size, number of workers, etc. The class defines methods to download and 
prepare the data (prepare_data) and set up the datasets (setup). It also includes methods 
for creating dataloaders for training, validation, and testing data (train_dataloader, 
val_dataloader, test_dataloader).

## prepare_data(self) function - 
This function is responsible for downloading the data if needed. It is called only once on 
a single CPU. It uses the save_tiffs_local_from_s3 function from src.data_utils module to 
download TIFF files from S3 to the local directory specified by train_dir, val_dir, and 
test_dir.

## setup(self, stage: str) function - 
This method is responsible for setting up the datasets for training, validation, and 
testing. It takes in stage which is a string as an argument (which can be either "train", 
"validate", or "test"). Depending on the stage argument, it creates instances of the 
BPSMouseDataset class (from src.dataset.bps_dataset module) for the respective data 
splits. The BPSMouseDataset instances are initialized with the corresponding CSV 
files, directories, transformations, and other necessary arguments.

## train_dataloader - 
returns TRAIN_DATALOADERS as the training dataloader. The dataloaders are created by 
passing the corresponding BPSMouseDataset instances and batch size to the DataLoader 
constructor.

## val_dataloader - 
Returns the validation dataloader. The dataloaders are created by passing the 
corresponding BPSMouseDataset instances and batch size to the DataLoader constructor.

## test_dataloader - 
Returns the test dataloader but in this case, we will only use the val_dataloader. The 
dataloaders are created by passing the corresponding BPSMouseDataset instances and batch 
size to the DataLoader constructor.

## main - 
The main function serves as an example usage of the BPSDataModule class. It sets up the 
necessary configurations, such as bucket name, S3 path, S3 client, and file paths for the 
BPS dataset. Then, it creates an instance of BPSDataModule, passing the required arguments.

The prepare_data method is called to download the data from S3 and the setup method is 
called to set up the datasets. Finally, an example loop is provided to demonstrate the 
usage of the train dataloader, printing the batch index, image shape, and label.



# bps_dataset.py - 

The file bps_dataset.py contains a PyTorch dataset class called BPSMouseDataset.

## BPSMouseDataset class - 
BPSMouseDataset class is a subclass of torch.utils.data.Dataset and represents a custom PyTorch dataset for BPS microscopy data. It takes the following arguments:

meta_csv_file, which is the name of the metadata CSV file.

meta_root_dir, which is the path to the metadata CSV file.

s3_client, which is a boto3.client object for interacting with AWS S3, it is optional.

bucket_name, which is the name of the S3 bucket from the AWS open source registry, it is 
optional.

transform, which is an optional transformation to be applied to the samples, it is 
optional.

file_on_prem, which is a boolean flag indicating whether the data is on the local file 
system or S3, it is also optional.

The class also includes an __init__ method for initializing the dataset and a __len__ 
method to return the number of images in the dataset.

## getitem method - 
The __getitem__ method fetches the image and corresponding label for a given index. It 
reads the metadata at the specified index, retrieves the image file name, and constructs 
the image path. If the data is on S3, it fetches the image from the S3 bucket using the 
get_bytesio_from_s3 function, converts it to a NumPy array, and applies the transformation 
if available. If the data is on the local file system, it loads the image using OpenCV 
(cv2) and applies the transformation if available. Finally, it fetches the one-hot encoded 
labels for the particle types, converts them to tensors, and returns the image and label.

## main function - 
The main function is used to test the BPSMouseDataset class. It sets the S3 bucket name 
and paths, and then creates an instance of BPSMouseDataset with various parameters, such 
as the metadata CSV file, the local root directory, and transformations. It also prints 
the length of the dataset and an example sample obtained using the __getitem__ method.

