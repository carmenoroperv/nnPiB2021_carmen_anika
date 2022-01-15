from sklearn.linear_model import LassoCV
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
import random
import sys
import torch
import seaborn as sns
torch.set_default_dtype(torch.float)

###Dataset class
class Dataset(torch.utils.data.Dataset):
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
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index][:]
        y = self.target[index][:]
        
        return X, y
    
traindata = Dataset(snakemake.input.trainset_x, snakemake.input.trainset_y)
valdata = Dataset(snakemake.input.valset_x, snakemake.input.valset_y)
testdata = Dataset(snakemake.input.testset_x, snakemake.input.testset_y)

random.seed(1)

loading_training = torch.utils.data.DataLoader(traindata, batch_size= len(traindata), num_workers = 1)
for i, batch in enumerate(loading_training): 
        train_features, train_labels = batch


loading_validation = torch.utils.data.DataLoader(valdata, batch_size= len(valdata), num_workers= 1)

for i, batch in enumerate(loading_validation): 
        val_features, val_labels = batch
        
loading_test = torch.utils.data.DataLoader(testdata, batch_size= len(valdata), num_workers= 1)

for i, batch in enumerate(loading_test): 
        test_features, test_labels = batch
        
val_features = val_features.numpy()
val_features = val_features.astype(int)

val_labels = val_labels.numpy()
val_labels = val_labels.reshape(-1,1)
val_labels = np.concatenate(val_labels)

test_features = test_features.numpy()
test_features = test_features.astype(int)

test_labels = test_labels.numpy()
test_labels = test_labels.reshape(-1,1)
test_labels = np.concatenate(test_labels)

train_features = train_features.numpy()
train_features = train_features.astype(int)

train_labels = train_labels.numpy()
train_labels = train_labels.flatten()
        
Linreg = LinearRegression().fit(train_features, train_labels)

Linpredictions = Linreg.predict(val_features)
test_pred = Linreg.predict(test_features)



print("RSME of linear regression - VAL")
print(np.sqrt(mean_squared_error(val_labels, Linpredictions)))

print("RSME of linear regression -TEST")
print(np.sqrt(mean_squared_error(test_labels, test_pred)))

print("MSE of linear regression -VAL")
print(mean_squared_error(val_labels, Linpredictions))

print("MSE of linear regression -TEST")
print(mean_squared_error(test_labels, test_pred))

print("Standard deviation in validation set")
np.std(val_labels)

print("Standard deviation in test set")
np.std(test_labels)

predictions_lr = pd.DataFrame(data = test_pred, columns = ["predicted"])
predictions_lr["observed"] = test_labels
predictions_lr.head()

predictions_lr.to_csv("../predictions/predictions_baseline_linear.csv", index=False)


g = sns.jointplot(data=predictions_lr, x="observed", y="predicted", kind="reg", line_kws={"color": "gold"})
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=10)

g.savefig("Linear_reg.png")


print("Correlation of the true values and the predictions for linear regression")
print(np.corrcoef(test_labels, test_pred))

from joblib import dump, load
dump(Linreg, 'Linear_regression.joblib') 

sys.stdout.flush()
