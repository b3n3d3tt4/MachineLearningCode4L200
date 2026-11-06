"""
beta-VAE with learned observation noise + classifier on the latent space.

This module only TRAINS and EVALUATES: it returns numbers and arrays, it never plots.
All the plots are in common/l200_utils.py (section 3b).
"""

import copy
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import l200_utils as ut          # common/: loaders


def _get_device(device):
    # If a device is not specified and CUDA is available then CUDA is chosen
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class VariationalAutoEncoder(nn.Module):
    
    def __init__(self, input_length=200, latent_dim=8, channels=1, noise_init=None):
        super().__init__()

        self.input_length = input_length
        self.latent_dim = latent_dim
        self.channels = channels
        
        # -----------------------------------
        # ------------- ENCODER -------------
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, stride=2, padding=2),   # 200 -> 100
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),        # 100 -> 50
            nn.LeakyReLU(0.1),
        )
        
        # ------------------------------------------------------
        # ----------- FLATTENING of the FINAL VECTOR -----------
        # NOTE: the sizes below are fixed for input_length = 200
        self.enc_out_channels = 32
        self.enc_out_length = 50
        flat_dim = self.enc_out_channels * self.enc_out_length   # 1600
        
        self.to_hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.LeakyReLU(0.1),
        )
        
        # -------------------------------------------
        # -------------- LATENT SPACE ---------------
        # 1 for the means, 1 for the logarithm of the variances 
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim) # this appears in the regularisation term of the loss and it is precisely the uncertainty we have in autoencoder representation
        
        # -------------------------------------------------------------------------      
        # ------------- INFORMATION ACQUISITION from the LATENT SPACE ------------- 
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, flat_dim),
            nn.LeakyReLU(0.1),
        )
        
        # ----------------------------------------------------
        # --------------- EXPLICIT UPSAMPLING ----------------
        # --- Upsample module is just a linear interpolation ----
        self.up1 = nn.Upsample(size=100, mode='linear', align_corners=False) # this makes the 50 points going to 100 
        self.dec_conv1 = nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2)   # still 100 points
        
        self.up2 = nn.Upsample(size=input_length, mode='linear', align_corners=False)
        self.dec_conv2 = nn.Conv1d(16, channels, kernel_size=5, stride=1, padding=2)  # still 200 points (or input_length)
        
        # ---------------------------------------------------------
        # --------------- LEARNED OBSERVATION NOISE ---------------
        # ONE value per channel, in the SAME order used by ut.multichannel:
        # channel 0 = current, channel 1 = charge.
        # It is initialised near the observed noise level (baseline std, computed in
        # preliminary.prepare_data); if noise_init is not given, a generic value is used.
        if noise_init is None:
            noise_init = [0.01] * channels
        assert len(noise_init) == channels, \
            f"noise_init has {len(noise_init)} values but the model has {channels} channels"
        sigma_obs = torch.tensor(noise_init, dtype=torch.float32)
        self.logvar_obs = nn.Parameter(torch.log(sigma_obs ** 2))        
        
        
    def reparametrize(self, mu, logvar):
        '''
        The reparametrisation trick is needed for backpropagation
        Sampling over a distribution is NOT a differentiable procedure
        '''
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        
        return z
    
    
    def encode(self, x):
        h = self.encoder(x)
        hidden = self.to_hidden(h)
        return hidden, self.mu(hidden), self.logvar(hidden)
    
    
    def decode(self, z):
        d = self.from_latent(z)
        d = d.view(-1, self.enc_out_channels, self.enc_out_length) # this is an operation on the tensor, not a network layer
        d = self.up1(d)
        d = self.dec_conv1(d)
        d = torch.nn.functional.leaky_relu(d, 0.1)
        d = self.up2(d) 
        return self.dec_conv2(d)
        
        
    def forward(self, x):
        '''
        Forward pass (encoder -> sampling of z -> decoder), used for the loss.
        Always call model(x), never model.forward(x): __call__ runs the PyTorch hooks too.
        '''
        hidden, mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        reconstructed = self.decode(z)
        
        return hidden, mu, logvar, reconstructed
    
    
    @torch.no_grad()
    def reconstruct(self, x):
        '''
        DETERMINISTIC reconstruction: decodes mu instead of a random sample of z.
        This is the one to look at in the plots (and the one to use for denoising):
        the sampled reconstruction of forward() changes every time it is called.
        '''
        _, mu, _ = self.encode(x)
        return self.decode(mu)

    
def betaloss(reconstructed, original, mu, logvar, logvar_obs, beta=1.0, free_bits=0.0):
    '''
    Negative ELBO with learned Gaussian observation noise (one sigma per channel):
    L = NLL(reconstruction) + beta * sum_j max(KL_j, free_bits)
    Returns (total loss, NLL, KL without free bits).
    '''
    
    batch_size = original.size(0)
    wave_shape = original.size(2)
    
    #-----------------------------------------------------
    # ---------------- RECONSTRUCTION TERM ----------------
    squared_errors =  (reconstructed - original) **2
    squared_errors_per_channel = squared_errors.sum(dim=(0, 2))
    weighted_errors = squared_errors_per_channel / torch.exp(logvar_obs)
    
    log_term = (batch_size * wave_shape) * logvar_obs
    
    negative_loglikelihood = 0.5 * torch.sum(weighted_errors + log_term)
    
    reconstruction = negative_loglikelihood / batch_size
    
    # -----------------------------------------------------
    # ---------------- REGULARISATION TERM ----------------
    KL_per_dimension = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()), dim=0)
     
    KLregularisation = KL_per_dimension.sum()
    
    if free_bits > 0:
        KL_loss = torch.clamp(KL_per_dimension, min=free_bits).sum() 
    else:
        KL_loss = KLregularisation
    
    # ---------------- TOTAL LOSS ----------------
    loss = reconstruction + beta * KL_loss
    
    return loss, reconstruction, KLregularisation


def training(model, waves_train, labels_train, waves_val, labels_val, epochs, beta=1.0,
             warmup_epochs=10, free_bits=0.5, channels=1, batch_size=64, lr=1e-3, clip_norm=5.0,
             fixed_val_idx=69, device=None):
    '''
    Trains the VAE. Returns
        model
        history   -> dict of per-epoch lists: train_loss/rec/kl, val_loss/rec/kl
        snapshots -> dict: 'original' (one fixed validation waveform, (channels, 200)) and
                     'reconstructions' (its deterministic reconstruction after every epoch)
    '''
    device = _get_device(device)
    model.to(device)
    print(f"Training on the device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # The validation loader is built ONCE: it is deterministic (no shuffle),
    # so every epoch is evaluated in the very same way
    val_loader = ut.test_loader(waves_val, labels_val, channels=channels, batch_size=batch_size)
    
    history = {k: [] for k in ('train_loss', 'train_rec', 'train_kl', 'val_loss', 'val_rec', 'val_kl')}
    
    # fixed validation waveform followed epoch by epoch (default 69 = sample #5 of batch #1, as before)
    fixed_val_idx = min(fixed_val_idx, len(waves_val) - 1)
    x_fixed = torch.as_tensor(waves_val[fixed_val_idx:fixed_val_idx + 1], dtype=torch.float32, device=device)
    snapshots = dict(original=np.array(waves_val[fixed_val_idx]), reconstructions=[], index=fixed_val_idx)
    
    for epoch in range(epochs):
        # KL warm-up: beta grows linearly from ~0 to its target value in the first warmup_epochs epochs
        current_beta = beta * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        
        # ---------------- TRAINING ----------------
        model.train()
        loader = ut.epoch_loader(waves_train, labels_train, channels=channels, batch_size=batch_size)
        
        epoch_loss, epoch_rec, epoch_kl = 0.0, 0.0, 0.0
        
        for batch_waves, _ in loader:
            batch_waves = batch_waves.to(device)
            
            optimizer.zero_grad()
            hidden, mu, logvar, reconstructed = model(batch_waves)
            
            loss, reconstruction, kl = betaloss(reconstructed, batch_waves, mu, logvar, model.logvar_obs,
                                                 beta=current_beta, free_bits=free_bits)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rec += reconstruction.item()
            epoch_kl += kl.item()
        
        history['train_loss'].append(epoch_loss / len(loader))
        history['train_rec'].append(epoch_rec / len(loader))
        history['train_kl'].append(epoch_kl / len(loader))
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_rec, val_kl = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_waves, _ in val_loader:
                batch_waves = batch_waves.to(device)
                
                hidden, mu, logvar, reconstructed = model(batch_waves)
                loss, reconstruction, kl = betaloss(reconstructed, batch_waves, mu, logvar, model.logvar_obs,
                                                     beta=current_beta, free_bits=free_bits)
                val_loss += loss.item()
                val_rec += reconstruction.item()
                val_kl += kl.item()
            
            snapshots['reconstructions'].append(model.reconstruct(x_fixed)[0].cpu().numpy())
        
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_rec'].append(val_rec / len(val_loader))
        history['val_kl'].append(val_kl / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{epochs}] beta={current_beta:.3f} | "
              f"train Loss:{history['train_loss'][-1]:.4f} Rec:{history['train_rec'][-1]:.4f} KL:{history['train_kl'][-1]:.4f} | "
              f"val Loss:{history['val_loss'][-1]:.4f} Rec:{history['val_rec'][-1]:.4f} KL:{history['val_kl'][-1]:.4f}")
    
    return model, history, snapshots
    
    
def testing(model, waves, labels, beta=1.0, free_bits=0.5, channels=1, batch_size=64, example_seed=None, device=None):
    '''
    Evaluates the VAE on a set. Use the SAME beta and free_bits of the training, otherwise
    the test loss is not comparable with the validation loss.
    Returns a dict with loss, rec, kl and 'example' (one random waveform and its
    deterministic reconstruction, for the plot).
    '''
    device = _get_device(device)
    model.to(device)
    model.eval()
    print(f"Testing on the device: {device}")
    
    loader = ut.test_loader(waves, labels, channels=channels, batch_size=batch_size)
    
    test_loss, test_rec, test_kl = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch_waves, _ in loader:
            batch_waves = batch_waves.to(device)
            
            hidden, mu, logvar, reconstructed = model(batch_waves)
            loss, reconstruction, kl = betaloss(reconstructed, batch_waves, mu, logvar, model.logvar_obs,
                                                 beta=beta, free_bits=free_bits)
            test_loss += loss.item()
            test_rec += reconstruction.item()
            test_kl += kl.item()
    
    result = dict(loss=test_loss / len(loader), rec=test_rec / len(loader), kl=test_kl / len(loader))
    print(f"Test result - Loss: {result['loss']:.6f} | Rec: {result['rec']:.6f} | KL: {result['kl']:.6f}")
    
    idx = int(np.random.default_rng(example_seed).integers(len(waves)))
    x = torch.as_tensor(waves[idx:idx + 1], dtype=torch.float32, device=device)
    result['example'] = dict(original=np.array(waves[idx]), reconstruction=model.reconstruct(x)[0].cpu().numpy(), index=idx)
    
    return result


# -----------------------------------------------------------------------------
# --------------------------- CLASSIFIER ON THE LATENT -------------------------
# -----------------------------------------------------------------------------

class Classifier(nn.Module):
    
    def __init__(self, latent_dim=8):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),          # regularisation: with only 8 input features overfitting comes fast
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            
            nn.Linear(16, 1),          # ONE output only: BCEWithLogitsLoss expects a single logit
        )
        
    def forward(self, x):
        # NOTE: the output is a LOGIT (any real number), not a probability.
        # BCEWithLogitsLoss applies the sigmoid internally, in a numerically stable way.
        # Never put a sigmoid here, otherwise it gets applied twice.
        return self.classifier(x)


def classifier_loss(predicted, label):
    return nn.BCEWithLogitsLoss()(predicted, label)


@torch.no_grad()
def encode_mu(vae, waves):
    """
    Extract mu from the frozen VAE encoder.
    no_grad: the VAE must not be trained here, only the classifier is.
    """
    _, mu, _ = vae.encode(waves)
    return mu


@torch.no_grad()
def latent_means(vae, waves, batch_size=2048, device=None):
    """mu of every waveform (numpy, (n, latent_dim)), computed in batches."""
    device = _get_device(device)
    vae.to(device)
    vae.eval()
    out = []
    for start in range(0, len(waves), batch_size):
        x = torch.as_tensor(waves[start:start + batch_size], dtype=torch.float32, device=device)
        out.append(encode_mu(vae, x).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def classifier_probabilities(classifier, mu, batch_size=8192, device=None):
    """Probability of being MSE for every latent vector mu (numpy in, numpy out)."""
    device = _get_device(device)
    classifier.to(device)
    classifier.eval()
    out = []
    for start in range(0, len(mu), batch_size):
        z = torch.as_tensor(mu[start:start + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(classifier(z)).cpu())
    return torch.cat(out).numpy().ravel()


def training_classifier(vae, classifier, waves_train, labels_train, waves_val, labels_val,
                        channels=1, batch_size=64, epochs=30, lr=1e-3, device=None):
    '''
    Trains the classifier on the frozen VAE latent means.
    At the end the classifier is brought back to the epoch with the LOWEST validation loss
    (history['best_epoch'], counted from 0), not left at the last epoch.
    '''
    device = _get_device(device)
    vae.to(device)
    classifier.to(device)
    print(f"Training on the device: {device}")
    
    vae.eval()   # the VAE stays frozen for the whole classifier training
    
    # test_loader, not epoch_loader: validation must be deterministic (no shuffling)
    validation_loader = ut.test_loader(waves_val, labels_val, channels=channels, batch_size=batch_size)
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, best_epoch = np.inf, None, 0
    
    for epoch in range(epochs):
        
        train_loader = ut.epoch_loader(waves_train, labels_train, channels=channels, batch_size=batch_size)
        
        # ---------------- TRAINING ----------------
        classifier.train()
        epoch_loss, correct_predictions, total_samples = 0.0, 0, 0
        
        for waves, labels in train_loader:
            waves = waves.to(device)
            # float() is needed by BCEWithLogitsLoss, unsqueeze(1) matches the (batch, 1) output shape
            labels = labels.to(device).float().unsqueeze(1)
            
            mu = encode_mu(vae, waves)
            
            optimizer.zero_grad()
            predictions = classifier(mu)
            loss = classifier_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # sigmoid turns the logit into a probability, round() turns it into a 0/1 prediction
            correct_predictions += (torch.sigmoid(predictions).round() == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples
        
        # ---------------- VALIDATION ----------------
        classifier.eval()
        validation_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for waves, labels in validation_loader:
                waves = waves.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                predictions = classifier(encode_mu(vae, waves))
                validation_loss += classifier_loss(predictions, labels).item()
                correct_val += (torch.sigmoid(predictions).round() == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss = validation_loss / len(validation_loader)
        val_accuracy = 100 * correct_val / total_val
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(classifier.state_dict())
        
        print(f"Epoch [{epoch+1}/{epochs}] train loss={train_loss:.4f} acc={train_accuracy:.2f}% | "
              f"val loss={val_loss:.4f} acc={val_accuracy:.2f}%")
    
    classifier.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch + 1}: classifier restored to that epoch")
    
    return classifier, history


def test_classifier(vae, classifier, waves, labels, channels=1, batch_size=64, device=None):
    '''
    Evaluates the classifier on a set. Returns a dict with loss, accuracy,
    p (probability of being MSE for every event) and y (true labels).
    '''
    device = _get_device(device)
    vae.to(device)
    classifier.to(device)
    vae.eval()
    classifier.eval()
    
    loader = ut.test_loader(waves, labels, channels=channels, batch_size=batch_size)
    
    test_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for waves_batch, labels_batch in loader:
            waves_batch = waves_batch.to(device)
            labels_batch = labels_batch.to(device).float().unsqueeze(1)
            
            predictions = classifier(encode_mu(vae, waves_batch))
            test_loss += classifier_loss(predictions, labels_batch).item()
            
            probs = torch.sigmoid(predictions)
            correct_predictions += (probs.round() == labels_batch).sum().item()
            total_samples += labels_batch.size(0)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels_batch.cpu())
    
    result = dict(loss=test_loss / len(loader), accuracy=100 * correct_predictions / total_samples,
                  p=torch.cat(all_probs).numpy().ravel(), y=torch.cat(all_labels).numpy().ravel())
    print(f"Classifier test - loss: {result['loss']:.4f} | accuracy: {result['accuracy']:.2f}%")
    return result

