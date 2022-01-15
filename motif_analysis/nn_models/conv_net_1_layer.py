import torch
from torch import nn, optim
import torch.nn.functional as F
import math

def calc_conv_layer_output_size(n, k, stride):
    return int((n - k*4 + stride)/stride)

def calc_pool_output_size(n, padding, kernel_size, stride):
    return math.ceil((n + padding - (kernel_size -1) - 1)/stride) + 1

def calc_flat_output_size(second_dim, third_dim):
    return second_dim * third_dim


class ConvNet(torch.nn.Module):
    
    def __init__(self, 
                 input_len, 
                 num_channels, 
                 conv_kernel_size_nts, 
                 conv_kernel_stride, 
                 pool_kernel_size, 
                 h1_size):

        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=conv_kernel_size_nts * 4, stride = conv_kernel_stride)
        self.bn_c1 = nn.BatchNorm1d(num_channels)
        self.pool_c1 = torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        self.dropout_c1 = nn.Dropout(0.5)
        
        conv_output_size_c1 = calc_conv_layer_output_size(n = input_len, k = conv_kernel_size_nts, stride = conv_kernel_stride)
        pool_c1_output_size = calc_pool_output_size(n = conv_output_size_c1, padding = 0, kernel_size = pool_kernel_size, stride = pool_kernel_size)
        flat_size = calc_flat_output_size(second_dim = num_channels, third_dim = pool_c1_output_size)
        
        self.h1 = torch.nn.Linear(flat_size, h1_size)
        self.bn_h1 = nn.BatchNorm1d(h1_size)
        self.dropout_h1 = nn.Dropout(0.5) 
        self.out = torch.nn.Linear(h1_size, 1)

    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn_c1(x)
        x = self.pool_c1(x)
        x = self.dropout_c1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.h1(x))
        x = self.bn_h1(x)               
        x = self.dropout_h1(x)
        x = self.out(x)
        return x