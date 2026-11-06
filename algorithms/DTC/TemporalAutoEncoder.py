'''
Implementation of the Temperal AutoEncoder class for the Deep Temporal Clustering model

based on the model implemented in the work at this link https://github.com/FlorentF9/DeepTemporalClustering
referring to the papaer https://arxiv.org/abs/1802.01059

The original work differs under sone aspects but the main structure is the same
'''


import os
import csv
import argparse
from time import time

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
import importlib
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.cluster import AgglomerativeClustering, KMeans



class BiLSTM(nn.Module):
    
    def __init__(self, 
                 previous_output, 
                 n_units, 
                 use_charge=False
                 ):
        
        super().__init__()
        
        self.previous_output = previous_output
        if use_charge:
            channels = 2
        else:
            channels = 1
        self.channels = channels
        self.n_units = n_units
        
        self.biLSTM = nn.LSTM(
            input_size = previous_output, 
            hidden_size = n_units, 
            bidirectional = True, 
            batch_first = True # this is needed to ensure that the first dimension is the batch size
        )
        
    
    def forward(self, x):
        
        x_flipped = torch.permute(x, (0, 2, 1))
        
        bidirectional_output, _ = self.biLSTM(x_flipped)

        forward = bidirectional_output[:, :, :self.n_units]
        backward = bidirectional_output[:, :, self.n_units:]
                
        total = forward + backward
        
        total_flipped = torch.permute(total, (0, 2, 1)) # total.permute(0, 2, 1) is even lighter
        
        return total_flipped
        
        


class TemporalAutoEncoder(nn.Module):
    
    def __init__(self, 
                 input_length=200, # this is how many points in time our signals have
                 n_filters=50, kernel_size=10, strides=1, # this is needed in the convolutional layer part
                 pool_size=10, # this is used in the MaxPooling part and it is needed that input_length % pool_size == 0
                 n_units=[50, 1], # these are the units for the BiLSTM class --> corresponds to the channels of the inputs and the number of filters of the convolutional layers
                 use_charge=False # default False boolean variable, if False --> channels = 1 else channels = 2
                 ):
        
        super().__init__()
        
        # All of these initialisations are needed because othewise these variables can be used only inside the init function
        self.input_length = input_length
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.use_charge = use_charge
        
        if self.use_charge:
            channels = 2
        else:
            channels = 1
        self.channels = channels
        
        # ----- NOISE MODEL (for the weighted NLL) -----
        # generic starting values: overwrite them with the baseline noise of the training waveforms
        # (data['current_noise'] / data['charge_noise'] from create_loaders), e.g.
        #     with torch.no_grad(): tae.logvar_obs.fill_(float(np.log(noise ** 2)))
        current_noise = 0.01
        charge_noise  = 0.01
        
        if self.channels == 2:
            sigma_obs = torch.tensor([current_noise, charge_noise])
        else:
            sigma_obs = torch.tensor([current_noise])
        
        # defined as a parameter in order to update it in the pre-training
        # during the DTC training this will not be affected by the gradietn --> tae.logvar_obs.requires_grad_(False)
        self.logvar_obs = nn.Parameter(torch.log(sigma_obs ** 2))
            
        
        assert(input_length % pool_size == 0)
        
        # ----- ENCODER -----
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = self.channels, 
                      out_channels = self.n_filters, 
                      kernel_size = self.kernel_size, 
                      stride = self.strides, 
                      padding = 'same', # padding = 'same' in Torch can be used only with strides = 1 that it's ok in this case
                      ), 
            # at this point the dimensions are (batch x channels x time) = (batch x channels x 200)
            nn.LeakyReLU(0.1), 
            
            nn.MaxPool1d(kernel_size = self.pool_size), # at this point the dimensions are (batch x channels x time) = (batch x channels x 20) --> but in Torch LSTM require (batch x time x channels)
            
            # 1^ BiLSTM
            BiLSTM(previous_output=self.n_filters, n_units=self.n_units[0]),
            nn.LeakyReLU(0.1),
            
            # 2^ BiLSTM
            BiLSTM(previous_output=self.n_units[0], n_units=self.n_units[1]),
            nn.LeakyReLU(0.1) # This is the latent space
        )
        
        # ----- DECODER -----
        self.decoder = nn.Sequential(
            nn.Conv1d(self.n_units[1], self.n_filters, kernel_size=self.kernel_size, stride=self.strides, padding='same'),
            
            nn.LeakyReLU(0.1),
            
            nn.Upsample(scale_factor=self.pool_size, mode='linear', align_corners=False),
    
            nn.Conv1d(self.n_filters, self.channels, kernel_size=self.kernel_size, stride=self.strides, padding='same')
)
        
        
    def forward(self, x):
        
        latent = self.encoder(x)
        
        reconstructed = self.decoder(latent)
        
        return latent, reconstructed
    
    
# i think this is totally useless     
def mse_loss(reconstructed, original):
    
    loss_function = nn.MSELoss()
    loss = loss_function(reconstructed, original)
    
    return loss



def nll_loss(reconstructed, original, logvar_obs):
    
    batch_size = original.size(0) # dimension of the batch
    wave_shape = original.size(2)
        
    #-----------------------------------------------------
    # ---------------- RECONSTRUCTION TERM ----------------
    squared_errors =  (reconstructed - original) ** 2
    squared_errors_per_channel = squared_errors.sum(dim=(0, 2))
    weighted_errors = squared_errors_per_channel / torch.exp(logvar_obs)
    
    log_term = (batch_size * wave_shape) * logvar_obs
    
    negative_loglikelihood = 0.5 * torch.sum(weighted_errors + log_term)
    
    loss = negative_loglikelihood / batch_size
    
    return loss


