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

import tSdistances as tdist
importlib.reload(tdist)


class TimeSeriesClustering(nn.Module):
    
    def __init__(self, 
                 n_clusters=2,       # number of clusters we want to individuate
                 input_length=20,    # time dimension of the latent (after pooling)
                 n_features=1,       # feature dimension of the latent (n_units[-1] of the TAE)
                 alpha=1.0,          # alpha parameter of the t-Student distribution
                 distance_metric='euclidean',
                 ):
        
        super().__init__()
        
        self.n_clusters = n_clusters
        self.input_length = input_length
        self.n_features = n_features
        self.alpha = alpha
        self.distance_metric = distance_metric
        
        # Cluster centroids: same shape as a single latent sequence, one per cluster.
        # Randomly initialised for now; will be overwritten with real KMeans/hierarchical
        # centroids later, in DTC.py.
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, input_length, n_features))
        nn.init.xavier_uniform_(self.clusters)
        
        
    def forward(self, x):
        # x: (batch, timesteps, n_features) -- latent sequences from the TAE encoder
        # (already permuted to this convention before being passed in)
        
        x_expanded = x.unsqueeze(1)  # (batch, 1, timesteps, n_features)
        
        if self.distance_metric == 'euclidean':
            diff = x_expanded - self.clusters                     # (batch, n_clusters, timesteps, n_features)
            distance = torch.sqrt(torch.sum(diff ** 2, dim=2))    # sum over time -> (batch, n_clusters, n_features)
            distance = torch.sum(distance, dim=-1)                # sum over features -> (batch, n_clusters)
        
        elif self.distance_metric == 'cid':
            # Complexity estimate per sequence: sqrt(sum of squared consecutive diffs) along time
            ce_x = torch.sqrt(torch.sum((x[:, 1:, :] - x[:, :-1, :]) ** 2, dim=1) + 1e-9)  # (batch, n_features)
            ce_w = torch.sqrt(torch.sum((self.clusters[:, 1:, :] - self.clusters[:, :-1, :]) ** 2, dim=1) + 1e-9)  # (n_clusters, n_features)
            
            ce_x_expanded = ce_x.unsqueeze(1)  # (batch, 1, n_features)
            ce = torch.maximum(ce_x_expanded, ce_w) / torch.minimum(ce_x_expanded, ce_w)  # (batch, n_clusters, n_features)
            
            diff = x_expanded - self.clusters                          # (batch, n_clusters, timesteps, n_features)
            ed = torch.sqrt(torch.sum(diff ** 2, dim=2))                # per-feature euclidean, no sum yet -> (batch, n_clusters, n_features)
            
            distance = torch.sum(ed * ce, dim=-1)                      # weighted sum over features -> (batch, n_clusters)
        
        elif self.distance_metric == 'cor':
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True, unbiased=False)
            x_norm = (x - x_mean) / (x_std + 1e-9)                      # (batch, timesteps, n_features)
            
            c_mean = self.clusters.mean(dim=1, keepdim=True)
            c_std = self.clusters.std(dim=1, keepdim=True, unbiased=False)
            c_norm = (self.clusters - c_mean) / (c_std + 1e-9)          # (n_clusters, timesteps, n_features)
            
            pcc = torch.mean(x_norm.unsqueeze(1) * c_norm, dim=2)       # Pearson correlation per feature -> (batch, n_clusters, n_features)
            
            distance = torch.sum(torch.sqrt(2.0 * (1.0 - pcc) + 1e-9), dim=-1)  # correlation -> distance, sum over features
        
        elif self.distance_metric == 'acf':
            def compute_acf(ts):
                
                T = ts.shape[-2]
                ts_mean = ts.mean(dim=2, keepdim=True) # compute the mean value along the time axis
                ts_c = ts - ts_mean # centers the time signal around zero
                var = (ts_c ** 2).sum(dim=-2, keepdim=True) + 1e-9 # compute the normalisation factor
                
                acf_lags = []
                
                for lag in range(T):
                    '''
                    This function comout autocorrelation for every temporal shift
                    '''
                    if lag == 0:
                        cov = (ts_c * ts_c).sum(dim=-2, keepdim=True)
                    else:
                        cov = (ts_c[..., :-lag, :] * ts_c[..., lag:, :]).sum(dim=-2, keepdim=True)
                        
                    # this normalisation is done in order to have correlation coefficients between -1 and 1
                    acf_lags.append(cov / var) 
                    
                return torch.cat(acf_lags, dim=-2)
            
            # then we compute the acf coefficients for both the signal AND the centroids
            x_acf = compute_acf(x_expanded)                                 
            c_acf = compute_acf(self.clusters.unsqueeze(0))
            
            T = x_expanded.shape[2]
            
            # now we need the \Omega matrix that is a weigthind decaying matrix (diagonal)
            # Lags next to xzero weights 1.0, the ones far away weights 0.0
            weights = torch.linspace(1.0, 0.0, T, device=x.device).view(1, 1, T, 1)
            
            diff = x_acf - c_acf
            distance = torch.sqrt(torch.sum(weights * (diff ** 2), dim=2) + 1e-9) 
            distance = torch.sum(distance, dim=-1)
            
        else:
            raise ValueError('Available distances are euclidean, cid, cor and acf!')
        
        # Student's t-distribution kernel
        num = 1.0 / (1.0 + (distance **2) / self.alpha)
        num = num ** ((self.alpha + 1) / 2)
        
        # Row-normalisation: each sample's cluster probabilities sum to 1
        den = torch.sum(num, dim=1, keepdim=True)
        q = num / den
        
        return q
    
    
def target_distribution(q):
    
    f = q.sum(dim=0)                        # (n_clusters,) dimensione di ogni cluster
    
    num = q ** 2 / f                        # (n_eventi, n_clusters)
    den = num.sum(dim=1, keepdim=True)      # (n_eventi, 1)
    
    p = num / den
    
    return p.detach()                       # p è un bersaglio fisso, niente gradiente


def kl_loss(p, q, epsilon=1e-8):
    batch_size = p.size(0)
    
    log = torch.log((p + epsilon) / (q + epsilon))
    argument = p * log
    
    clusters_sum = argument.sum(dim=1)
    loss = clusters_sum.sum() / batch_size
    # loss = clusters_sum.mean()
    
    return loss