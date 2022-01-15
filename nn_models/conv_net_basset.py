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

    def __init__(self, input_len, 
                 num_channels_c1, 
                 num_channels_c2, 
                 conv_kernel_size_nts_c1, 
                 conv_kernel_size_nts_c2,
                 conv_kernel_stride, 
                 pool_kernel_size_c1, 
                 pool_kernel_size_c2,
                 h1_size, 
                 #h2_size, 
                 dropout_p):

        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=num_channels_c1, kernel_size=conv_kernel_size_nts_c1 * 4, stride = conv_kernel_stride)
        self.bn_c1 = nn.BatchNorm1d(num_channels_c1)
        self.pool_c1 = torch.nn.MaxPool1d(kernel_size=pool_kernel_size_c1, stride=pool_kernel_size_c1, ceil_mode=True)
        
        conv_output_size_c1 = calc_conv_layer_output_size(n = input_len, 
                                                          k = conv_kernel_size_nts_c1, 
                                                          stride = conv_kernel_stride)
        pool_c1_output_size = calc_pool_output_size(n = conv_output_size_c1, 
                                                    padding = 0, 
                                                    kernel_size = pool_kernel_size_c1, 
                                                    stride = pool_kernel_size_c1)
        
        self.conv2 = torch.nn.Conv1d(in_channels=num_channels_c1, out_channels=num_channels_c2, kernel_size=conv_kernel_size_nts_c2 * 4, stride = conv_kernel_stride)
        self.bn_c2 = nn.BatchNorm1d(num_channels_c2)
        self.pool_c2 = torch.nn.MaxPool1d(kernel_size=pool_kernel_size_c2, stride=pool_kernel_size_c2, ceil_mode=True)
        
        pool_c2_output_size = calc_pool_output_size(n = calc_conv_layer_output_size(n = pool_c1_output_size, 
                                                                                    k = conv_kernel_size_nts_c2, 
                                                                                    stride = conv_kernel_stride), 
                                                    padding = 0, 
                                                    kernel_size = pool_kernel_size_c2, 
                                                    stride = pool_kernel_size_c2)
        
        #self.conv3 = torch.nn.Conv1d(in_channels=num_channels_c2, out_channels=num_channels_c3, kernel_size=conv_kernel_size_nts_c3 * 4, stride = conv_kernel_stride)
        #self.bn_c3 = nn.BatchNorm1d(num_channels_c3)
        #self.pool_c3 = torch.nn.MaxPool1d(kernel_size=pool_kernel_size_c3, stride=pool_kernel_size_c3, ceil_mode=True)
        
        #pool_c3_output_size = calc_pool_output_size(n = calc_conv_layer_output_size(n = pool_c2_output_size, 
        #                                                                            k = conv_kernel_size_nts_c3, 
        #                                                                            stride = conv_kernel_stride), 
        #                                            padding = 0, 
        #                                            kernel_size = pool_kernel_size_c3, 
        #                                            stride = pool_kernel_size_c3)

        flat_size = calc_flat_output_size(second_dim = num_channels_c2, third_dim = pool_c2_output_size)
        
        self.h1 = torch.nn.Linear(flat_size, h1_size)
        self.dropout_h1 = nn.Dropout(dropout_p) 
        
        #self.h2 = torch.nn.Linear(h1_size, h2_size)
        #self.dropout_h2 = nn.Dropout(dropout_p) 
        
        #self.out = torch.nn.Linear(h2_size, 1)
        self.out = torch.nn.Linear(h1_size, 1)

    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(self.bn_c1(x))
        x = self.pool_c1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn_c2(x))
        x = self.pool_c2(x)
        
        #x = self.conv3(x)
        #x = F.relu(self.bn_c3(x))
        #x = self.pool_c3(x)
     
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.h1(x))              
        x = self.dropout_h1(x)
        
        #x = F.relu(self.h2(x))              
        #x = self.dropout_h2(x)
        x = self.out(x)
        return x