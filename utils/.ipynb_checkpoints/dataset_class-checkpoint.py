from torch.utils.data import Dataset
import torch

class custom_Dataset(Dataset):
    def __init__(self, features_file, target_file):
        'Initialization'
        features = torch.load(features_file)
        target = torch.load(target_file)
        features = features.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        self.features = features
        self.target = target
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.features.shape[0]
    
    def shape(self):
        'Denotes the total number of samples and features'
        return self.features.shape
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index][:]
        y = self.target[index][:]
        
        return X, y