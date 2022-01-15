import numpy as np
import pandas as pd
import torch
import os
import sys
import math
import matplotlib.pyplot as plt
import importlib
from math import sqrt
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adam

#import custom written classes and functions
from utils.dataset_class import custom_Dataset
from nn_models.conv_net_basset_for_chr22 import ConvNet ###### This does not work dynamically right now, and needs to be changed if we want to import a different model from a different file
                                                        ###### I (Carmen) am sure that there is a way to do this dynamically, so we do not have to change it here, but I have to look more into that
from utils.model_utils import nnUtils

############
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
############

subfolder = snakemake.params.subfolder

#os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(snakemake.threads)

from nn_models.conv_net_basset import ConvNet

#from tuning_configs import config_linear_one_layer_TR1
config_file_name = snakemake.params.config_file_name
config_dict_name = snakemake.params.config_dict_name

config_dicts = importlib.import_module(config_file_name)
config_dict = getattr(config_dicts, config_dict_name)



    
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
input_size = traindata.shape()[1]
num_filters_c1 = config_dict["num_filters_c1"]
num_filters_c2 = config_dict["num_filters_c2"]
conv_filter_size_c1 = config_dict["conv_filter_size_c1"]
conv_filter_size_c2 = config_dict["conv_filter_size_c2"]
conv_filter_stride = config_dict["conv_filter_stride"]
pooling_kernel_size_c1 = config_dict["pooling_kernel_size_c1"]
pooling_kernel_size_c2 = config_dict["pooling_kernel_size_c2"]
h1_size = config_dict["h1_size"]
#h2_size = config_dict["h2_size"]
dropout_p = config_dict["dropout_p"]
    
weight_decay = config_dict["weight_decay"]
learning_rate = config_dict["learning_rate"]
    
print_interval = snakemake.params.print_interval
max_epochs = snakemake.params.max_epochs

torch.manual_seed(1)
neural_net = ConvNet(input_len = input_size, 
                     num_channels_c1 = num_filters_c1, 
                     num_channels_c2 = num_filters_c2,
                     conv_kernel_size_nts_c1 = conv_filter_size_c1, 
                     conv_kernel_size_nts_c2 = conv_filter_size_c2,
                     conv_kernel_stride = conv_filter_stride,
                     pool_kernel_size_c1 = pooling_kernel_size_c1, 
                     pool_kernel_size_c2 = pooling_kernel_size_c2,
                     h1_size = h1_size, 
                    #h2_size = h2_size, 
                     dropout_p = dropout_p)

################################################################################

############### LOAD BATCHES AND INITIALIZE MODEL UTILS CLASS #################

torch.manual_seed(1)
bs = config_dict["batch_size"]


train_batches = DataLoader(traindata, batch_size=bs, shuffle=True)
val_batches = DataLoader(valdata, batch_size=81258)

print(f"\nTrain batches loaded. The batch size is {bs} and there are {len(traindata)/bs} batches in one epoch.")

criterion = nn.MSELoss()
if weight_decay == None: 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate)
else: 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate, weight_decay = weight_decay)
model_utils = nnUtils(neural_net, criterion, optimizer, train_batches, val_batches, subfolder = subfolder, print_interval = print_interval, ray_tune = False)

print(f"nnUtils class initialized. The model is on {model_utils.device} device.")
sys.stdout.flush()
################################################################################


model_utils.train_model(max_epochs)

print("\nTraining done!")
print(f"Best validation error seen: {model_utils.best_val_loss}")
with open(snakemake.output.output, 'w') as f:
    f.write(str(model_utils.best_val_loss))
sys.stdout.flush()