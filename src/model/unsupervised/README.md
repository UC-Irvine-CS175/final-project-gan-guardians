# pca_tsne.py
This file can load and preprocess image data using BPSDatamodule and perform 
dimensionality reduction on the data by using PCA (Principal Component Analsis) and 
applying t-SNE (t-distributed Stochastic Neighbor Embedding) on the image data.

# utility funtions
1. 
preprocess_images is a function that takes in a Dataloader object and preprocesses 
it by flatten images into a 1-dimensional representation. It returns a numpy array 
of flattened images, as well as a list of labels that corresponds to each flattened 
images.

2. 
