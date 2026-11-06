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

import metrics
importlib.reload(metrics)

import tSdistances as tdist # not useful, in the end I implemented all of the distances directly in the TSC class
importlib.reload(tdist)

import TemporalAutoEncoder as tae
importlib.reload(tae)

import TimeSeriesClustering as tsc
importlib.reload(tsc)

import l200_utils as ut          # common/: all the plotting functions live here (NOT reloaded: keeps ut.TAG / ut.PREFIX)


class DTC(nn.Module):
    '''
    DEEP TEMPORAL CLUSTERING class
    '''
    
    def __init__(self,
                 input_length=200, # this is how many points in time our signals have
                 n_filters=50, kernel_size=10, strides=1, # this is needed in the convolutional layer part
                 pool_size=10, # this is used in the MaxPooling part and it is needed that input_length % pool_size == 0
                 n_units=[50, 1], # these are the units for the BiLSTM class --> corresponds to the channels of the inputs and the number of filters of the convolutional layers
                 n_clusters = 2, # number of clusters we want to individuate (2 SSE/MSE in our case)
                 alpha=1.0, # alpha parameter of the t-Student distribution
                 distance_metric='euclidean',
                 use_charge=False # default False boolean variable, if False --> channels = 1 else channels = 2
                 ):
        
        super().__init__()
        
        assert(input_length % pool_size == 0)
        
        self.input_length = input_length
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.distance_metric = distance_metric
        
        if use_charge:
            channels = 2
        else: 
            channels = 1
        self.channels = channels
        
        self.tae = tae.TemporalAutoEncoder(input_length = self.input_length, 
                                           n_filters = self.n_filters, 
                                           kernel_size = self.kernel_size, 
                                           strides = self.strides, 
                                           pool_size = self.pool_size, 
                                           n_units = self.n_units, 
                                           use_charge=use_charge
                                           )
        
        self.tsc = tsc.TimeSeriesClustering(n_clusters = self.n_clusters, 
                                            input_length = self.input_length // self.pool_size,
                                            n_features = self.n_units[-1],
                                            alpha = self.alpha, 
                                            distance_metric = self.distance_metric                                            
                                            )
        
        
    def forward(self, x):
        
        latent, reconstructed = self.tae(x)
        
        latent_permuted = latent.permute(0, 2, 1)
        q = self.tsc(latent_permuted)
        
        return reconstructed, q
    
    
# ----- ACCURACY and PURITY COMPUTATION from METRICS file -----
def accuracy_purity(model, loader, device=None):
    
    model.eval()
    
    all_predictions, all_true = [], []
    
    with torch.no_grad():
        for waves, labels in loader:
            waves = waves.to(device)
            
            _, q = model(waves)
            
            all_predictions.append(q.argmax(dim=1).cpu())
            all_true.append(labels)
            
    y_predictions = torch.cat(all_predictions).numpy()
    y_true = torch.cat(all_true).numpy()
    
    accuracy = metrics.cluster_accuracy(y_true, y_predictions)
    purity = metrics.cluster_purity(y_true, y_predictions)
    
    return accuracy, purity                
# ---------------------------------------------------------------
    
    
# (plot_reconstruction_grid moved to common/l200_utils.py)


# ---------------------------------------- PRETRAINING - only TEMPORAL AUTOENCODER -----------------------------------------
def pretrain(model, train_loader, validation_loader, epochs=10, lr=1e-3, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Pretraining on the device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, validation_losses = [], []
    
    # ------------ PER IL PLOT ------------
    fixed_batch_idx = 1
    fixed_sample_idx = 5
    reconstructed_validation_per_epoch = []
    original_sample_fixed = None
    # -------------------------------------
    
    for epoch in range(epochs):
        
        # ---------------- TRAINING ----------------
        model.train()
        epoch_loss = 0.0
        
        for batch_waves, _ in train_loader:
            batch_waves = batch_waves.to(device)
            
            optimizer.zero_grad()
            latent, reconstructed = model(batch_waves)
            loss = tae.nll_loss(reconstructed, batch_waves, model.logvar_obs)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        validation_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (batch_waves, _) in enumerate(validation_loader):
                batch_waves = batch_waves.to(device)
                
                latent, reconstructed = model(batch_waves)
                loss = tae.nll_loss(reconstructed, batch_waves, model.logvar_obs)
                validation_loss += loss.item()
                
                if batch_idx == fixed_batch_idx:
                    # shape (channels, length): niente squeeze, cosi' funziona sia con 1 che 2 canali
                    if original_sample_fixed is None:
                        original_sample_fixed = batch_waves[fixed_sample_idx].cpu().numpy()
                    reconstructed_sample_epoch = reconstructed[fixed_sample_idx].cpu().numpy()
        
        avg_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(avg_validation_loss)
        
        print(f"[pretrain] epoch {epoch+1}/{epochs} - train: {avg_train_loss:.4f}  val: {avg_validation_loss:.4f}")
        
        reconstructed_validation_per_epoch.append(reconstructed_sample_epoch)
    
    # ----- PLOT dell'evoluzione della ricostruzione nel tempo -----
    epochs_to_plot = sorted(set([0, epochs // 2 - 1 if epochs > 1 else 0, epochs - 1]))
    
    fig, axes, handles, labels = ut.plot_reconstruction_grid(
        original_sample_fixed,
        [reconstructed_validation_per_epoch[e] for e in epochs_to_plot],
        [f'Epoch #{e + 1}' for e in epochs_to_plot]
    )
    
    fig.suptitle('Original vs Reconstructed waveform (TAE pretraining) - Validation set',
                 x=0.5, y=0.99, fontsize=16, ha='center')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=len(labels), frameon=False, fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ut.save(fig, 'pretrain_reconstruction')
    plt.show()
    
    return train_losses, validation_losses #model
# ---------------------------------------------------------------------------------------------------------------------------


# ------------------------------------- INITIALISATION of CLUSTERS WEIGHTS via K-MEANS -------------------------------------
def initialization_of_clusters_weights(model, train_loader, device=None, seed=0):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Clusters initialisation on the device: {device}")
    
    print('Centroids initialisation via KMeans')
    
    all_latents = []
    
    with torch.no_grad():
        for waves, _ in train_loader:
            waves = waves.to(device)
            
            latent, _ = model.tae(waves)
            latent = latent.permute(0, 2, 1)
            all_latents.append(latent.cpu())
    
    features_np = torch.cat(all_latents, dim=0).numpy()
    
    flat_features = features_np.reshape(features_np.shape[0], -1)
    km = KMeans(n_clusters=model.n_clusters, n_init=10, random_state=seed).fit(flat_features)
    
    cluster_centers = km.cluster_centers_.reshape(
        model.n_clusters, features_np.shape[1], features_np.shape[2]
    )
    
    with torch.no_grad():
        model.tsc.clusters.copy_(torch.tensor(cluster_centers, dtype=torch.float32))
    
    print("Clusters initialised")
# ------------------------------------------------------------------------------------------------------------------------------ 


# -------------- CLUSTERS ASSIGNEMENT -------------- 
def cluster_assignments(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for waves, _ in loader:
            _, q = model(waves.to(device)) 
            # for an input q =[prob_cluster_1, prob_cluster_2, ...,  prob_cluster_k]
            # argmax returns the position in which the value is higher 
            # for example if q = [0.15, 0.20, 0.65] q.argmax = 2 --> cluster 2 assigned
            preds.append(q.argmax(dim=1).cpu())
    # the returns of this function is a numpy array with the information of which cluster the sample is assigned to        
    return torch.cat(preds).numpy() # cat stands for concatenate

                
# -------------------------------------------------- TOTAL TRAINING tae + tsc -------------------------------------------------- 
def training(model, train_loader, validation_loader, epochs=10, lr=1e-3, gamma=1.0, device=None): 
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on the device: {device}")
    
    # safety check: logvar_obs must be frozen before training
    assert not model.tae.logvar_obs.requires_grad, "Freeze logvar_obs before training"
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    n_clusters = model.n_clusters
    train_losses, validation_losses = [], []
    nll_train, kl_train, nll_val, kl_val = [], [], [], []
    fractions, changes = [], []

    # ------------ PER IL PLOT ------------
    fixed_batch_idx = 1
    fixed_sample_idx = 5
    reconstructed_validation_per_epoch = []
    original_sample_fixed = None
    # -------------------------------------
    
    # ---------------- CLUSTER ASSIGNMENTS BEFORE TRAINING ----------------
    y_last = cluster_assignments(model, validation_loader, device)
    fraction_start = np.bincount(y_last, minlength=n_clusters) / len(y_last)
    print(f"Initial (after KMeans) - events per cluster: {fraction_start.round(3)}")
    # ---------------------------------------------------------------------
    
    for epoch in range(epochs):
        
        # ---------------- TRAINING ----------------
        model.train()
        epoch_loss, epoch_tae, epoch_tsc = 0.0, 0.0, 0.0
                
        for batch_waves, _ in train_loader:
            batch_waves = batch_waves.to(device)
            
            optimizer.zero_grad()
            reconstructed, q = model(batch_waves)
            p = tsc.target_distribution(q)
            
            tae_loss = tae.nll_loss(reconstructed, batch_waves, logvar_obs = model.tae.logvar_obs)
            tsc_loss = tsc.kl_loss(p, q, epsilon=1e-8)
            total_loss = tae_loss + gamma * tsc_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_tae += tae_loss.item()
            epoch_tsc += tsc_loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_tae = epoch_tae / len(train_loader)
        avg_tsc = epoch_tsc / len(train_loader)
        train_losses.append(avg_train_loss)
        nll_train.append(avg_tae)
        kl_train.append(avg_tsc)
        
        # ---------------- VALIDATION ----------------
        model.eval()
        validation_loss, val_tae, val_tsc = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_idx, (batch_waves, _) in enumerate(validation_loader):
                batch_waves = batch_waves.to(device)
                
                reconstructed, q = model(batch_waves)
                p = tsc.target_distribution(q)

                tae_loss = tae.nll_loss(reconstructed, batch_waves, logvar_obs = model.tae.logvar_obs)
                tsc_loss = tsc.kl_loss(p, q, epsilon=1e-8)
                total_loss = tae_loss + gamma * tsc_loss
                
                validation_loss += total_loss.item()
                val_tae += tae_loss.item()
                val_tsc += tsc_loss.item()
                
                if batch_idx == fixed_batch_idx:
                    if original_sample_fixed is None:
                        original_sample_fixed = batch_waves[fixed_sample_idx].cpu().numpy()
                    reconstructed_sample_epoch = reconstructed[fixed_sample_idx].cpu().numpy()
        
        avg_validation_loss = validation_loss / len(validation_loader)
        validation_losses.append(avg_validation_loss)
        nll_val.append(val_tae / len(validation_loader))
        kl_val.append(val_tsc / len(validation_loader))
                
        # ---------------- CLUSTER ASSIGNMENTS (no labels needed) ----------------
        y_pred = cluster_assignments(model, validation_loader, device)
        fraction = np.bincount(y_pred, minlength=n_clusters) / len(y_pred)
        change = (y_pred != y_last).mean()
        y_last = y_pred
        fractions.append(fraction)
        changes.append(change)
        
        print(f"[train] epoch {epoch+1}/{epochs} - train: {avg_train_loss:.4f}  val: {avg_validation_loss:.4f}  "
              f"events per cluster: {fraction.round(3)}  changes: {change:.4f}")
        print(f"        NLL: {avg_tae:.2f}   KL: {avg_tsc:.4f}")
        
        reconstructed_validation_per_epoch.append(reconstructed_sample_epoch)
    
    # ----- PLOT dell'evoluzione della ricostruzione nel tempo -----
    epochs_to_plot = sorted(set([0, epochs // 2 - 1 if epochs > 1 else 0, epochs - 1]))
    
    fig, axes, handles, labels = ut.plot_reconstruction_grid(
        original_sample_fixed,
        [reconstructed_validation_per_epoch[e] for e in epochs_to_plot],
        [f'Epoch #{e + 1}' for e in epochs_to_plot]
    )
    
    fig.suptitle('Original vs Reconstructed waveform - Validation set',
                    x=0.5, y=0.99, fontsize=16, ha='center')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
                ncol=len(labels), frameon=False, fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ut.save(fig, f'{model.n_clusters}cl_training_reconstruction')
    plt.show()

    # storia del training: la leggi con model.history (i valori di ritorno restano quelli di prima)
    model.history = dict(nll_train=nll_train, kl_train=kl_train, nll_val=nll_val, kl_val=kl_val,
                         changes=changes, fractions=fractions)

    return model, train_losses, validation_losses, fractions, changes
# ---------------------------------------------------------------------------------------------------------------------------------------                 

        