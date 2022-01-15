import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("ggplot")
import time

############
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
############

class nnUtils(object):
    def __init__(self, model, mse_loss, optimizer, train_batches, val_batches, subfolder, print_interval, checkpoint_dir = None, ray_tune = False):
        self.model = model
        self.mse_loss = mse_loss
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_batches = train_batches
        self.val_batches = val_batches
        
        self.train_loss = []
        self.val_loss = []
        self.total_epochs = 0
        self.best_val_loss = 100
        
        self.subfolder = subfolder
        self.model_save_interval = None
        self.print_interval = print_interval
        self.checkpoint_dir = checkpoint_dir
        self.ray_tune = ray_tune
        
        os.makedirs(f"{self.subfolder}", exist_ok=True)
        os.makedirs(f"{self.subfolder}Plots/", exist_ok=True)
        
        
    def train_batch(self):
        running_loss = 0
        for train_features_batch, train_target_batch in self.train_batches:
        
            train_features_batch = train_features_batch.to(self.device)
            train_target_batch = train_target_batch.to(self.device)

            self.model.train()
            pred_target = self.model(train_features_batch)
            loss = self.mse_loss(pred_target, train_target_batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = running_loss/len(self.train_batches)
        return epoch_loss
        
    def val_batch(self):
        running_loss = 0
        self.model.eval()
        for val_features_batch, val_target_batch in self.val_batches:
            
            val_features_batch = val_features_batch.to(self.device)
            val_target_batch = val_target_batch.to(self.device)
            
            pred_val = self.model(val_features_batch)
            loss = self.mse_loss(pred_val, val_target_batch)
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(self.val_batches)
        return epoch_loss
            
    def set_seeds(self, seed = 0):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train_model(self, max_epochs, seed = 0):
        self.set_seeds(seed)
        self.model_save_interval = round(max_epochs / 20)
        
        epoch_start = self.total_epochs
        for epoch in range(epoch_start, max_epochs):
            start_time = time.time()
            self.total_epochs += 1
            train_loss_epoch = self.train_batch()
            self.train_loss.append(train_loss_epoch)
            
            with torch.no_grad():
                val_loss_epoch = self.val_batch()
                self.val_loss.append(val_loss_epoch)
                
            if (self.total_epochs) % self.model_save_interval == 0 or val_loss_epoch < self.best_val_loss:
                self.save_model()
                if val_loss_epoch < self.best_val_loss: 
                    self.best_val_loss = val_loss_epoch
            
            if (self.total_epochs) % self.print_interval == 0 or val_loss_epoch == self.best_val_loss:
                print(f"\nEPOCH {epoch +1} DONE.")
                print("Running time of this epoch: %s seconds" % (time.time() - start_time))
                print("Train MSE:", train_loss_epoch)
                print("Val MSE:", val_loss_epoch)
                print("Best val MSE seen so far:", self.best_val_loss)
                sys.stdout.flush()
                
                self.save_plot()
                self.save_losses()
            
            if self.ray_tune == True:
                with tune.checkpoint_dir(step=self.total_epochs) as self.checkpoint_dir:
                    path = os.path.join(self.checkpoint_dir, "checkpoint")
                    torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
                tune.report(loss = val_loss_epoch)
            
        # save final model, for resuming training if needed
        self.save_model()
                
    def save_model(self):
        state = {'epoch': self.total_epochs,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'train_mse':self.train_loss, 
                 'val_mse': self.val_loss,
                 'best_val_mse': self.best_val_loss
                 }
        filepath = f"{self.subfolder}model_nn_epoch_{self.total_epochs}.pt"
        torch.save(state, filepath)
        return state

    def save_losses(self):
        filepath = f"{self.subfolder}train_val_mse.csv"
        if self.total_epochs == 1: 
            df = pd.DataFrame(columns=["train_loss","val_loss","epoch"])
            df.to_csv(filepath)
        
        df = pd.DataFrame({'train_loss': [self.train_loss[-1]], 
                           'val_loss': [self.val_loss[-1]], 
                           'epoch': [self.total_epochs]})
        df.to_csv(filepath, mode='a', header=False)
   

    def save_plot(self):
        filepath = f'{self.subfolder}Plots/plot_losses_epoch_{self.total_epochs}.png'
        plt.title("Training Curve, with mse")
        plt.plot(list(range(1, self.total_epochs + 1)), self.train_loss, label="Train")
        plt.plot(list(range(1, self.total_epochs + 1)), self.val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend(loc='best')
        plt.savefig(filepath)
        plt.show()
        plt.close()

    def load_checkpoint(self, filename):
        # Load file with model, optimizer and losses
        checkpoint = torch.load(filename)

        # Restore 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.train_loss = checkpoint['train_mse']
        self.val_loss = checkpoint['val_mse']
        self.best_val_loss = min(self.val_loss)
        self.model.train()   

    def predict(self, test_data):
        self.model.eval() 
        
        # Takes aNumpy input and make it a float tensor
        # Send input to device and uses model for prediction
        predictions = self.model(test_data.to(self.device))
        
        # Set it back to train mode
        self.model.train()
        
        # Detaches it, brings it to CPU and back to Numpy
        return predictions.detach().cpu().numpy()

        
        

        
        
    
        
        