# resnet101.py


## BPSConfig - 
The BPSConfig class is a dataclass that holds configuration options for the BPS Microscopy dataset. 
It includes various parameters such as the data directory, file names for training and validation 
CSV files, save directories for visuals and model weights, batch size, maximum epochs, accelerator 
type, device, number of workers, and data module stage. 


## ResNet101 - 
This class is a subclass of torchvision.models.resnet.ResNet and represents the ResNet-101 model 
with transfer learning for image classification. It overrides the constructor to customize the 
model's behavior. It loads the pre-trained weights if specified in the configuration or uses 
the default pre-trained weights. It also modifies the last fully connected layer to output 
the desired # of classes.


## main function - 
The main function contains the training and evaluation logic.

Wandb Login: This function logs in to the Weights & Biases platform.

Initializing BPSConfig: An instance of BPSConfig is created to store the configuration 
options for the BPS Microscopy dataset.

Fixing Random Seed: The random seed is set to ensure reproducibility of results.

Instantiating BPSDataModule: The BPSDataModule class is instantiated with the necessary 
parameters. This class is responsible for loading and preparing the data for training and 
validation.

Setting Up DataModule: The setup method of BPSDataModule is called to prepare the data for 
the training and validation stages.

Initializing ResNet101 Model: An instance of the ResNet101 class is created, initializing 
the model with the pre-trained weights if config.load_weights is set to True. The fully 
connected layer (fc) is modified to have 2 output classes.

Freezing Pretrained Layers: The parameters of the pre-trained layers in the model are set 
to requires_grad=False to freeze them and prevent them from being modified during training.

Defining Loss Function and Optimizer: The loss function is defined as torch.nn.CrossEntropyLoss(), 
which is commonly used for multi-class classification tasks. The optimizer is defined as 
torch.optim.Adam() with a learning rate of 0.001.

Moving Model to GPU: If a GPU is available (config.device is set to 'cuda'), the model is 
moved to the GPU.

Getting Train and Validation Dataloaders: The dataloaders for the training and validation 
datasets are obtained from the BPSDataModule object.

Setting Model to Evaluation Mode: The model is set to evaluation mode using model.eval().

Initializing Weights & Biases: Weights & Biases is initialized with the project name and 
configuration details.

Training the Model: If config.load_weights is False, the model is trained for the specified 
number of epochs. The training loop iterates over the training dataloader, performs forward 
and backward passes, and updates the model parameters.

Evaluating the Model: The trained model is evaluated on the validation dataset. The accuracy 
is computed by comparing the predicted labels with the ground truth labels.

Logging Metrics to Weights & Biases: The training loss and validation accuracy are logged to 
Weights & Biases for visualization.

Saving the Model: If config.save_weights is True, the trained model's state_dict is saved to 
the specified save directory.

Ending Weights & Biases Run: The Weights & Biases run is finished.


## if __name__ == '__main__': - 
Finally, the main function is called to start the execution of the code.


