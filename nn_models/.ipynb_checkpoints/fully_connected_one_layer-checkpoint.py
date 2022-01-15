import torch
from torch import nn, optim
import torch.nn.functional as F
import math


class NeuralNet(torch.nn.Module):

    def __init__(self, input_len, h1_size):

        super(NeuralNet, self).__init__()
        self.h1 = torch.nn.Linear(input_len, h1_size) #Hidden layer 1   
        self.output_layer = torch.nn.Linear(h1_size, 1)
    
    def forward(self, x):

        x = torch.nn.functional.relu(self.h1(x)) #Activation of hidden layer 1
        x = self.output_layer(x)#Activation of the output layer

        return x
