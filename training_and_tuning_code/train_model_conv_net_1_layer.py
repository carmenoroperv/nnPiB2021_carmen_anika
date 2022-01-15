import numpy as np
import pandas as pd
import torch
import os
import sys
import math
import matplotlib.pyplot as plt
from math import sqrt
import random
import importlib

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adam

#import custom written classes and functions
from utils.dataset_class import custom_Dataset
from utils.model_utils import nnUtils

#from nn_models.fully_connected_one_layer_reg import NeuralNet_bn_do 
model_file_name = snakemake.params.model_file_name
model_name = snakemake.params.model_name

models = importlib.import_module(model_file_name)
NeuralNet = getattr(models, model_name)

subfolder = snakemake.params.subfolder
torch.set_num_threads(snakemake.threads)

resume_training_nfilters_dict = {"n_filters_25" : "model_nn_epoch_300.pt", 
                                 "n_filters_50" : "model_nn_epoch_240.pt",
                                 "n_filters_75" : "model_nn_epoch_195.pt",
                                 "n_filters_100" : "model_nn_epoch_150.pt",
                                 "n_filters_150" : "model_nn_epoch_105.pt",
                                 "n_filters_200" : "model_nn_epoch_53.pt"}

resume_training_filtersize_dict = {"filtersize_10" : "model_nn_epoch_300.pt", 
                                   "filtersize_11" : "model_nn_epoch_300.pt",
                                   "filtersize_12" : "model_nn_epoch_300.pt",
                                   "filtersize_13" : "model_nn_epoch_270.pt",
                                   "filtersize_14" : "model_nn_epoch_300.pt",
                                   "filtersize_15" : "model_nn_epoch_300.pt",
                                   "filtersize_16" : "model_nn_epoch_300.pt",
                                   "filtersize_17" : "model_nn_epoch_270.pt",
                                   "filtersize_18" : "model_nn_epoch_270.pt",
                                   "filtersize_19" : "model_nn_epoch_300.pt",
                                   "filtersize_20" : "model_nn_epoch_300.pt",
                                   "filtersize_21" : "model_nn_epoch_300.pt",
                                   "filtersize_22" : "model_nn_epoch_300.pt",
                                   "filtersize_23" : "model_nn_epoch_300.pt",
                                   "filtersize_24" : "model_nn_epoch_300.pt",
                                   "filtersize_25" : "model_nn_epoch_300.pt",
                                   "filtersize_26" : "model_nn_epoch_300.pt",
                                   "filtersize_27" : "model_nn_epoch_300.pt",
                                   "filtersize_28" : "model_nn_epoch_300.pt",
                                   "filtersize_29" : "model_nn_epoch_282.pt",
                                   "filtersize_30" : "model_nn_epoch_300.pt"}

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

h1_size = int(snakemake.params.h1_size)
num_channels = int(snakemake.params.num_channels)
conv_kernel_size_nts = int(snakemake.params.conv_kernel_size_nts)
conv_kernel_stride = int(snakemake.params.conv_kernel_stride)
pool_kernel_size = int(snakemake.params.pool_kernel_size)

torch.manual_seed(int(snakemake.params.seed))
neural_net = NeuralNet(input_len = input_size, 
                       num_channels = num_channels, 
                       conv_kernel_size_nts = conv_kernel_size_nts, 
                       conv_kernel_stride = conv_kernel_stride, 
                       pool_kernel_size = pool_kernel_size, 
                       h1_size = h1_size)

################################################################################

############### LOAD BATCHES AND INITIALIZE MODEL UTILS CLASS #################


bs = int(snakemake.params.batch_size)
print_interval = snakemake.params.print_interval
learning_rate = float(snakemake.params.learning_rate)
weight_decay = float(snakemake.params.weight_decay)

torch.manual_seed(int(snakemake.params.seed))
train_batches = DataLoader(traindata, batch_size=bs, shuffle=True, num_workers = 4, pin_memory = True)
val_batches = DataLoader(valdata, batch_size=81258, num_workers = 4, pin_memory = True)

print(f"\nTrain batches loaded. The batch size is {bs} and there are {len(traindata)/bs} batches in one epoch.")

criterion = nn.MSELoss()
if weight_decay == "None": 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate)
else: 
    optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate, weight_decay = weight_decay)
model_utils = nnUtils(neural_net, criterion, optimizer, train_batches, val_batches, subfolder, print_interval, ray_tune = False)

print(f"nnUtils class initialized. The model is on {model_utils.device} device.")
sys.stdout.flush()

####################### RESUME TRAINING ########################################
if str(snakemake.params.resuming_training_model) == "yes":
    model_utils.load_checkpoint("tuning_results/conv_net_1_layer/TR1_n_filters/ConvNet_n_filters_" + str(num_channels) + "/" + str(resume_training_nfilters_dict["n_filters_" + str(num_channels)]))
    print("Model training resumed from the given model")
    sys.stdout.flush()
################################################################################

max_epochs = snakemake.params.max_epochs

model_utils.train_model(max_epochs, seed = int(snakemake.params.seed))

print("\nTraining done!")
print(f"Best validation error seen: {model_utils.best_val_loss}")
with open(snakemake.output.output, 'w') as f:
    f.write(str(model_utils.best_val_loss))

sys.stdout.flush()