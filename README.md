# Machine Learning PSD for LEGEND-200 – implementation

Implementation of the methods of my Master's thesis _Signal/background discrimination for the LEGEND-200 experiment using deep learning algorithms on GPUs_ which focuses on PSD of SSE and
multi-site MSE events in IC HPGe detectors of LEGEND-200, from their current waveforms of <sup>228</sup>Th calibration data.

| method | type | idea |
|---|---|---|
| **β-VAE + classifier** | semi-supervised | a β-VAE learns a latent space of the waveforms; a small classifier on the latent means separates SSE-like (²⁰⁸Tl DEP) from MSE-like (²¹²Bi FEP) events |
| **DTC** – Deep Temporal Clustering | unsupervised | temporal autoencoder (Conv1D + BiLSTM) + clustering layer with Student-t soft assignments, trained jointly; Euclidean, Complexity Invariant (CID), correlation and autocorrelation (ACF) distances |
| **SOM** – Self-Organizing Map | unsupervised | data preparation for a SOM (`torchsom` - [https://github.com/michelin/TorchSOM](https://github.com/michelin/TorchSOM)) on the same input of the DTC |

This repository contains **only the code**: no data, no trained models, no analysis notebooks and no results.
LEGEND-200 data are internal to the collaboration.

## Structure

```
.
├── common/
│   ├── detector.py            detector choice, folders (data / models / plots), detector name on every plot
│   ├── filereading.py         extraction of waveforms, energy, A/E, rise time from LEGEND lh5 files; per-period checkpoints
│   └── l200_utils.py          1. waveform cleaning  2. Gaussian peak fit and torch loaders  3. plotting functions
│
├── algorithms/
│   ├── VAE/
│   │   ├── VariationalAutoEncoder.py   β-VAE with learned observation noise, latent classifier, training / testing loops
│   │   └── vae_preliminary.py          loading, cleaning, DEP/FEP fits, SSE/MSE labels, train/val/test split, metrics
│   ├── DTC/
│   │   ├── TemporalAutoEncoder.py      temporal autoencoder (Conv1D + BiLSTM)
│   │   ├── TimeSeriesClustering.py     clustering layer: Student-t soft assignment, target distribution, KL loss
│   │   ├── DeepTemporalClustering.py   DTC model: TAE pre-training, KMeans initialisation, joint training
│   │   ├── tSdistances.py              time-series distances (Euclidean, CID, correlation, ACF)
│   │   ├── metrics.py                  clustering accuracy, purity, ROC AUC
│   │   └── dtc_preliminary.py          loading, cleaning, loaders, shape / gradient sanity checks
│   └── SOM/
│       └── som_preliminary.py          loading, cleaning and loaders for the SOM
```

## Methods

**Cleaning** (`common/l200_utils.py`, `newProcessing`): baseline subtraction → normalisation → current (derivative) →
Gaussian low-pass filter (σ = 1.25 samples; a Butterworth filter is also available) → alignment on the first current peak
with a prominence of at least 15 % of the highest one → 200-sample window (75 samples before the peak, 124 after).
The current can be divided by one global constant (`computeScale`, median peak of the training set), so that the
relative amplitude, i.e. the A/E information, is kept.

**β-VAE + classifier** (`algorithms/VAE`): convolutional encoder/decoder, Gaussian likelihood with a learned observation
noise per channel, β-weighted KL term with warm-up and free bits. After training, the VAE is frozen and a classifier
is trained on the latent means of the events labelled SSE (DEP window) and MSE (FEP window); its output probability is
the PSD variable, and the cut is set at a chosen survival fraction of the DEP.

**DTC** (`algorithms/DTC`): following Madiraju et al., *Deep Temporal Clustering* ([arXiv:1802.01059](https://arxiv.org/abs/1802.01059)).
The implementation is based on the Keras implementation by Florent Forest,
[FlorentF9/DeepTemporalClustering](https://github.com/FlorentF9/DeepTemporalClustering), rewritten in PyTorch and adapted
to the waveforms of LEGEND-200 (e.g. reconstruction loss as a Gaussian likelihood with a learned observation noise).
The temporal autoencoder is pre-trained alone; the cluster centroids are initialised with KMeans in the latent space;
then autoencoder and clustering layer are trained together with loss NLL + γ·KL(P‖Q). With 2 clusters, the soft
assignment q of the SSE-like cluster is the PSD variable.

## Usage

```bash
pip install -r requirements.txt
```

The modules import each other by name: add `common/` and the folder of the algorithm to `sys.path`, and set the
detector **before** importing the other modules:

```python
import os, sys
sys.path[:0] = [os.path.abspath(p) for p in ('common', 'algorithms/DTC', 'algorithms/VAE')]

import detector; detector.set_detector("V08682A")
import l200_utils as ut
import DeepTemporalClustering as DTC
import VariationalAutoEncoder as VAE
```

With real data, the checkpoints written by `common/filereading.py` are read from the folder in the environment variable
`L200_DATA` (default `./data`, ignored by git); models and plots are written in `algorithms/*/models`,
`algorithms/VAE/results` and `plots/`, also ignored by git.

## Author

Benedetta Rasera – Master's thesis in Physics, Università degli Studi di Padova.

_This README.md has been written exploiting Claude by Anthropic._
