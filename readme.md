This is the code for my master thesis which implements and trains a neural network on the ADNI dataset

# Overview of important files and how to run them

ReadData
A file that implements that class used to load in the data using the dataloader mecanisms from python

Neuralnetsimple1 and NeuralnetBasaia
The classes that implements the architecture of the two networks

TrainBasaia
Train the model inspired by Basaia et al. with this file. Simply run it like "python3 file" or use sub.sh on the cluster

TrainSimple
Same as above, but for the simple network


forwardBasaia
Import a saved model that was trained with the Basaia network and test it. use submitforwardgpu.sh for the cluster

forwardSimple
The same but for the simple model


t-SNE2.py and Tsnenet
For implementing the T-sne. run using python3 T-sne2.py 

dataset_with_fieldstrengths_AIBL
The most updated dataset for AIBL including field strength for each image

dataset_With_filenames_ItalianADNI
Same, but for I-ADNI

dataset_with_filenames_allADNI
Same but for ADNI - be aware of duplicate subjects


ng2*.csv
original dataset I-ADNI

*mixed2.csv
data for the model trained on both field strenghts

aibl_pdx*.csv and baselineAIBL
original datasets for AIBL
