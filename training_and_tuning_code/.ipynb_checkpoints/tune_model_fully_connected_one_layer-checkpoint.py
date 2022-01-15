import numpy as np
import pandas as pd
import torch
import os
import sys
import json
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

import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


#import custom written classes and functions
from utils.dataset_class import custom_Dataset
from utils.model_utils import nnUtils

# IMPORT MODEL TO BE TUNED AND CONFIG FOR TUNING
#from nn_models.fully_connected_one_layer import NeuralNet
model_file_name = snakemake.params.model_file_name
model_name = snakemake.params.model_name

models = importlib.import_module(model_file_name)
NeuralNet = getattr(models, model_name)

#from tuning_configs import config_linear_one_layer_TR1
config_file_name = snakemake.params.config_file_name
config_dict_name = snakemake.params.config_dict_name

config_dicts = importlib.import_module(config_file_name)
config_dict = getattr(config_dicts, config_dict_name)

#####################################################

subfolder = snakemake.params.subfolder

#os.environ["OMP_NUM_THREADS"]="1"
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
sys.stdout.flush()



#####################################################

def tuning_func(config, checkpoint_dir=None, data_train_val = None):
    print("Starting!")
    
    ################## LOAD TRAIN AND VALIDATION DATA BATCHES ######################
    traindata = data_train_val[0]
    valdata = data_train_val[1]
    
    bs = config["batch_size"]
    train_batches = DataLoader(traindata, batch_size=bs, shuffle=True)
    val_batches = DataLoader(valdata, batch_size=81258)
    print(f"\nTrain batches loaded. The batch size is {bs} and there are {len(traindata)/bs} batches in one epoch.")
    sys.stdout.flush()
    
    ####################### LOAD MODEL PARAMETERS AND MODEL ########################
    input_size = traindata.shape()[1]
    h1_size = config["h1_size"]
    weight_decay = config["weight_decay"]
    learning_rate = config["learning_rate"]
    print_interval = snakemake.params.print_interval
    max_epochs = snakemake.params.max_epochs

    torch.manual_seed(1)
    neural_net = NeuralNet(input_len = input_size, h1_size = h1_size)
    criterion = nn.MSELoss()
    if weight_decay == None: 
        optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate)
    else: 
        optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    model_utils = nnUtils(neural_net, criterion, optimizer, train_batches, val_batches, subfolder, print_interval, checkpoint_dir = checkpoint_dir, ray_tune = True)
    print(f"nnUtils class initialized. The model is on {model_utils.device} device.")
    sys.stdout.flush()
    
    
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        neural_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    ################################################################################

    model_utils.train_model(max_epochs)

    print("\nTraining done!")
    print(f"Best validation error seen: {model_utils.best_val_loss}")
    sys.stdout.flush()

ray.init(include_dashboard=False)

asha_scheduler = ASHAScheduler(time_attr='training_iteration',
                               metric='loss',
                               mode='min',
                               reduction_factor = 2, 
                               max_t = snakemake.params.max_epochs, 
                               grace_period=20)

result = tune.run(tune.with_parameters(tuning_func, data_train_val = [traindata, valdata]),
                                       resources_per_trial={"cpu": 4, "gpu": 0},
                                       config = config_dict,
                                       scheduler=asha_scheduler, 
                                       log_to_file = True,
                                       local_dir="logs/raytune_output", name=snakemake.params.raytune_output_folder)


print("Best config: ", result.get_best_config(metric="loss", mode="min"))


df = result.results_df
print(df)
sys.stdout.flush()
