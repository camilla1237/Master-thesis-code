This is the code for my master thesis which implements and trains a neural network on the ADNI dataset

# Overview of files and how to run them

ReadData
A file that implements that class used to load in the data using the dataloader mecanisms from python

Neuralnetsimple1 and NeuralnetBasaia
The classes that implements the architecture of the two networks

TrainBasaia
Train the model inspired by Basaia et al. with this file. Simply run it like "python3 Train Basaia"

TrainSimple
Same as above, but for the simple network


forwardBasaia
Import a saved model that was trained with the Basaia network and test it.

forwardSimple
The same but for the simple model


t-SNE2.py and Tsnenet
For implementing the T-sne 
