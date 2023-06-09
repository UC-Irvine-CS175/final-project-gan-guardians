# pca_tsne.py
This file can load and preprocess image data using BPSDatamodule and perform 
dimensionality reduction on the data by using PCA (Principal Component Analsis) and 
applying t-SNE (t-distributed Stochastic Neighbor Embedding) on the image data.

# utility funtions
Here are the readmes for the utility functions within the file.

## 1. 
## preprocess_images
This is a function that takes in a Dataloader object and preprocesses 
it by flatten images into a 1-dimensional representation. It returns a numpy array 
of flattened images, as well as a list of labels that corresponds to each flattened 
images.

## 2. 
## perform_pca
This is a function that performs PCA (Principal Component Analysis) on the flattened 
images in order to reduce their dimensions. What it does is that the function takes in 
a numpy array of flattened images and an integer representing the numbers of components 
to keep when represented in a lower dimension as the two parameters. The function then 
initializes an PCA object which specified the numbers of dimensions/components to keep, 
then in the end it will return a tuple that is a PCA object, along with the compressed 
image data as a tuple.

## 3.
## perform_tsne
This is a function that applies the t-SNE (t-distributed Stochastic Neighbor Embedding)
on image data that had previously underwent transformations under the PCA and had their 
dimensions reduced. The function returns lower-dimensional t-SNE components.

## 4. 
## create_tsne_cp_df



This function creates a pandas DataFrame that contains the lower-dimensional t-SNE 
components and labels for each image. It takes in a numpy array of the lower dimensional 
t-SNE components as well as a list of one hot encoded labels corresponding to each 
flattened image, plus a third argument as an integer showing the numbers of point to plot.
The function create_tsne_cp_df returns a dataframe that contains the lower dimensional 
t-SNE components and the labels for each image.
