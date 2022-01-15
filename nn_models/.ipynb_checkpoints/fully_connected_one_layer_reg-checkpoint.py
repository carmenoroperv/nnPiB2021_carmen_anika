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
        x = self.output_layer(x) #Activation of the output layer

        return x


    
class NeuralNet_bn(torch.nn.Module):

    def __init__(self, input_len, h1_size):

        super(NeuralNet_bn, self).__init__()
        self.h1 = torch.nn.Linear(input_len, h1_size) #Hidden layer 1   
        self.bn = torch.nn.BatchNorm1d(h1_size) #NEW
        self.output_layer = torch.nn.Linear(h1_size, 1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.h1(x)) #Activation of hidden layer 1
        x = self.bn(x)
        x = self.output_layer(x) #Activation of the output layer

        return x

    
    
class NeuralNet_do(torch.nn.Module):

    def __init__(self, input_len, h1_size):

        super(NeuralNet_do, self).__init__()
        self.h1 = torch.nn.Linear(input_len, h1_size) #Hidden layer 1   
        self.dropout = torch.nn.Dropout(0.5) # Implementation of the dropout. The percentage should be adjusted. 
        self.output_layer = torch.nn.Linear(h1_size, 1)
                
    def forward(self, x):
        x = torch.nn.functional.relu(self.h1(x)) #Activation of hidden layer 1
        x = self.dropout(x)
        x = self.output_layer(x) #Activation of the output layer

        return x
    
    
class NeuralNet_bn_do(torch.nn.Module):

    def __init__(self, input_len, h1_size):

        super(NeuralNet_bn_do, self).__init__()
        self.h1 = torch.nn.Linear(input_len, h1_size) #Hidden layer 1   
        self.bn = torch.nn.BatchNorm1d(h1_size) #NEW
        self.dropout = torch.nn.Dropout(0.5) # Implementation of the dropout. The percentage should be adjusted. 
        self.output_layer = torch.nn.Linear(h1_size, 1)
                
    def forward(self, x):
        x = torch.nn.functional.relu(self.h1(x)) #Activation of hidden layer 1
        x = self.bn(x)
        x = self.dropout(x)
        x = self.output_layer(x) #Activation of the output layer

        return x