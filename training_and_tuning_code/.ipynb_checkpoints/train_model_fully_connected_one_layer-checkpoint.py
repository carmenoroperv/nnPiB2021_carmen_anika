import numpy as np
import pandas as pd
import torch
import os
import sys
import math
import matplotlib.pyplot as plt
from math import sqrt
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adam

#import custom written classes and functions
from utils.dataset_class import custom_Dataset
from nn_models.fully_connected_one_layer import NeuralNet 
from utils.model_utils import nnUtils

############
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
############

subfolder = snakemake.params.subfolder
torch.set_num_threads(snakemake.threads)
    
############## PREP AND LOAD DATA ###################

features_file_train = snakemake.input.features_train
target_file_train = snakemake.input.target_train
features_file_val = snakemake.input.features_val
target_file_val = snakemake.input.target_val

traindata = custom_Dataset(features_file_train, target_file_train)
valdata = custom_Dataset(features_file_val, target_file_val)

print("\n\n\n\nTrain data and validation data loaded.")
print(f"The number of rows in the train data is {len(traindata)} and the number of rows in validation data is {len(valdata)}")
print(f"There are {traindata.shape()[1]} features in train data.")

#####################################################

####################### LOAD MODEL PARAMETERS AND MODEL ########################

input_size = traindata.shape()[1]

h1_size = snakemake.params.h1_size

torch.manual_seed(1)
neural_net = NeuralNet(input_len = input_size, 
                       h1_size = h1_size)

################################################################################

############### LOAD BATCHES AND INITIALIZE MODEL UTILS CLASS #################

torch.manual_seed(1)
bs = int(snakemake.params.batch_size)
print_interval = snakemake.params.print_interval
learning_rate = float(snakemake.params.learning_rate)
weight_decay = snakemake.params.weight_decay

train_batches = DataLoader(traindata, batch_size=bs, shuffle=True)
val_batches = DataLoader(valdata, batch_size=81258)

print(f"\nTrain batches loaded. The batch size is {bs} and there are {len(traindata)/bs} batches in one epoch.")

criterion = nn.MSELoss()
if weight_decay == "None": 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate)
else: 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate, weight_decay = weight_decay)
model_utils = nnUtils(neural_net, criterion, optimizer, train_batches, val_batches, subfolder, print_interval, ray_tune = False)

print(f"nnUtils class initialized. The model is on {model_utils.device} device.")
sys.stdout.flush()
################################################################################

max_epochs = snakemake.params.max_epochs

model_utils.train_model(max_epochs)

print("\nTraining done!")
print(f"Best validation error seen: {model_utils.best_val_loss}")
sys.stdout.flush()