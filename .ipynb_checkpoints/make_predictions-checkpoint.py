import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import importlib

from utils.dataset_class import custom_Dataset
from utils.model_utils import nnUtils


torch.set_num_threads(snakemake.threads)

# read in model class
model_file_name = snakemake.params.model_file_name
model_name = snakemake.params.model_name

models = importlib.import_module(model_file_name)
NeuralNet = getattr(models, model_name)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def predict(model, test_data, device, criterion, observed):
    model.eval() 

    predictions = model(test_data)

    model.train()
    
    loss = criterion(predictions, observed)

    return predictions.detach().cpu().numpy(), loss

# initialize model: 
input_size = 804

h1_size = int(snakemake.params.h1_size)
num_channels_c1 = int(snakemake.params.num_channels_c1)
num_channels_c2 = int(snakemake.params.num_channels_c2)
conv_kernel_size_nts_c1 = int(snakemake.params.conv_kernel_size_nts_c1)
conv_kernel_size_nts_c2 = int(snakemake.params.conv_kernel_size_nts_c2)
conv_kernel_stride = int(snakemake.params.conv_kernel_stride)
pool_kernel_size_c1 = int(snakemake.params.pool_kernel_size_c1)
pool_kernel_size_c2 = int(snakemake.params.pool_kernel_size_c2)

torch.manual_seed(int(snakemake.params.seed))
neural_net = NeuralNet(input_len = input_size, 
                       num_channels_c1 = num_channels_c1, 
                       num_channels_c2 = num_channels_c2,
                       conv_kernel_size_nts_c1 = conv_kernel_size_nts_c1, 
                       conv_kernel_size_nts_c2 = conv_kernel_size_nts_c2,
                       conv_kernel_stride = conv_kernel_stride, 
                       pool_kernel_size_c1 = pool_kernel_size_c1,
                       pool_kernel_size_c2 = pool_kernel_size_c2,
                       h1_size = h1_size, 
                       dropout_p = 0.5)

# read in checkpoint
criterion = nn.MSELoss()
checkpoint = torch.load(snakemake.input.model_path, map_location=torch.device('cpu'))

# Restore model state
neural_net.load_state_dict(checkpoint['model_state_dict'])
  
# read in test data
test_data = torch.load(snakemake.input.test_data_tensor_file)
test_data_split = torch.split(test_data, 41398)

print("Test data split length")
print(len(test_data_split))

# read in test observations
test_data_obs = torch.load(snakemake.input.test_data_observed_tensor)
test_data_obs_split = torch.split(test_data_obs, 41398)


# make predictions
with torch.no_grad():
    predictions_1, test_loss_1 = predict(neural_net, test_data_split[0], device, criterion, test_data_obs_split[0])
    predictions_2, test_loss_2 = predict(neural_net, test_data_split[1], device, criterion, test_data_obs_split[1])

    
test_loss = (test_loss_1 + test_loss_2)
predictions = np.concatenate((predictions_1, predictions_2), axis=0)
with torch.no_grad():
    for i in range(2, len(test_data_split)):
        predictions_chunk, test_loss_chunk = predict(neural_net, test_data_split[i], device, criterion, test_data_obs_split[i])
        predictions = np.concatenate((predictions, predictions_chunk), axis=0)
        test_loss = (test_loss + test_loss_chunk)

test_loss = test_loss/len(test_data_split)
pred_df = pd.DataFrame(predictions)
pred_df = pred_df.rename(columns={0:"C02M02"})
print("Head of predictions dataframe")
print(pred_df.head())

print("Shape of predictions dataframe")
print(pred_df.shape)

#pred_df.to_csv("predictions/linear_one_layer_model/test_data_predictions.tsv", sep='\t', index = False)


# save predictions with positions
positions = pd.read_csv(snakemake.input.test_data_tsv_file_with_positions, sep='\t', header=0)
positions = positions.iloc[:, 0:3]

print("Head of positions")
print(positions.head())

pos_pred_df = pd.concat([positions.reset_index(drop=True), pred_df], axis=1)
print("Head of predictions with positions")
print(pos_pred_df.head())
pos_pred_df.to_csv(snakemake.output.predictions_with_positions, index = False, sep='\t')


print(f"MSE on test data: {test_loss}")