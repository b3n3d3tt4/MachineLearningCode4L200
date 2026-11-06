"""
Common tools of the thesis, used by every notebook (VAE, DTC, SOM, 2nuBB):

    1. WAVEFORM CLEANING   baseline subtraction, normalisation, current, low-pass filter (Gaussian / Butterworth),
                           alignment on the first current peak            (was wfcleaningNEW.py)
    2. LOADERS             Gaussian fit of the peaks, multichannel input, torch DataLoaders   (was loadercreator.py)
    3. PLOTTING            3a. style, saving and plots of the clustering (DTC, SOM)   (was plottingfunctions.py)
                           3b. plots of the VAE analysis                              (was plotting_functions.py)

Usage:
    import detector; detector.set_detector("V08682A")     # BEFORE importing this module (folders, detector in titles)
    import l200_utils as ut

Names that were the same in the two plotting modules:
    panel_size, CHANNEL_NAMES, Y_LABELS, color1 ... color8   -> identical, defined once (section 3a)
    ORIGINAL_COLOR / RECONSTRUCTED_COLOR                      -> DTC; the VAE ones are VAE_ORIGINAL_COLOR / VAE_RECONSTRUCTED_COLOR
    plot_reconstruction_grid                                  -> DTC; the VAE one is vae_reconstruction_grid
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import roc_curve

import torch
from torch.utils.data import TensorDataset, DataLoader

import detector                # detector name in the titles, folders of the plots


####################################################################################################
# 1. WAVEFORM CLEANING
####################################################################################################

# Processing settings of this version of the file (used as defaults by newProcessing):
#   low-pass filter on the current: GAUSSIAN (no ringing before/after sharp peaks), sigma = 1.25 samples
#   (same -6 dB frequency of the old Butterworth, 0.15 cycles/sample)
#   alignment on the first current peak with a prominence of at least 15 % of the highest one
FILTER_TYPE = 'gauss'          # 'gauss' or 'butter'
GAUSS_SIGMA = 1.25             # samples
PROMINENCE_FRACTION = 0.15

USES_BUTTERWORTH = (FILTER_TYPE == 'butter')   # read by preliminary (print + config)


def baseline(waveforms, baseline_points=300):
    """
    Remove the baseline from the waveforms.
    """
    
    # We compute the baseline of each waveform for the firt baseline_points points
    baselines = np.mean(waveforms[:, :baseline_points], axis=1, keepdims=True)
    
    # We remove the baseline from each waveform
    bl_removed = waveforms - baselines
    
    return bl_removed
    

def normalisation(waveforms):
    """
    Normalises the waveforms in orer to get always a maximum of 1.
    """
    
    # We compute the maximum value of each waveform
    maximum = np.max(waveforms, axis=1, keepdims=True)
    
    # If a waveform has all zeros, we manually set the maximum to 1 in order to avoid divisions by zero
    maximum[maximum == 0] = 1.0
    
    # We compute the normalisation dividing the waveforms by their maximum
    normalised = waveforms / maximum
    
    return normalised
    
    
def current(waveforms, dt=1.0):
    """
    Compute the current fromn the charge
    """
    
    current = np.gradient(waveforms, dt, axis=1)
    
    return current


def butterFilter(waveforms, order=4): # order is a parametere to define how much the filter is strong, the higher the order the stronger the filter
    
    waveforms = np.asarray(waveforms)
    
    sampling_frequency = 1.0 # the sampling frequency is 1 sample per unit time but we are actually working in unit time so the sampling frequency is 1.0; fs= 1/ dt
    nyquist_frequency = 0.5 * sampling_frequency # this is the minimum sampling frequency  necessary to get an analogic signal and to reconstructi it without losing infroamtion or itnroducing errors
    
    cutoff_frequency = 0.15 #0.35 / typical_rise_time # this is the cutoff frequency, it is the frequency at which the filter starts to attenuate the signal, it is inversely proportional to the rise time of the signal
    normalised_cutoff_frequency = cutoff_frequency / nyquist_frequency # this is the normalised cutoff frequency, it is the ratio between the cutoff frequency and the nyquist frequency
    
    sos = butter(order, normalised_cutoff_frequency, btype='low', output='sos') # this output is already the define filter used in the subsequent step of filtfilt
    filtered = sosfiltfilt(sos, waveforms, axis=1) # these are the filtered waveforms
    
    return filtered


def gaussFilter(waveforms, sigma=GAUSS_SIGMA):
    """
    Gaussian low-pass filter along time (zero phase, like sosfiltfilt). Its impulse response is always
    positive, so it never produces the negative ringing that the Butterworth gives before/after very sharp
    current peaks. sigma in samples: 1.25 has the same -6 dB frequency of butterFilter (0.15 cycles/sample).
    """
    return gaussian_filter1d(np.asarray(waveforms, dtype=float), sigma=sigma, axis=1)


def getPeaks(waveforms, baseline_points=300, n_sigmas=3, alignment='first', prominence_fraction=PROMINENCE_FRACTION, min_distance=5):
    '''
    Find, for each waveform, the index of the peak used for the alignment.
    The search starts after the baseline region (sample baseline_points).
    
    alignment='first' (default): FIRST significant peak in time. A peak is significant if it is above
        baseline_mean + n_sigmas * baseline_std AND its prominence is at least prominence_fraction of the
        highest peak of that waveform: the threshold alone is not enough, a noise ripple can pass it.
        The LEGEND trigger fires on the first energy deposition, so for a multi-site event this puts the
        first deposition at the alignment point and all the later ones after it.
    alignment='highest': highest peak above the threshold (old behaviour).
 
    Returns
    -------
    peaks_idx : list of (int or None), length n_events
        Index of the chosen peak for each waveform, None if no peak was found.
    is_noise : np.ndarray of bool, shape (n_events,)
        True where no peak was found for that event ("noise" event).
    '''
    
    if alignment not in ('first', 'highest'):
        raise ValueError("alignment must be 'first' or 'highest'")
    
    waveforms = np.asarray(waveforms)
    n_events, _ = waveforms.shape
 
    baseline_slice = waveforms[:, :baseline_points]
    baseline_mean = np.mean(baseline_slice, axis=1)
    baseline_std = np.std(baseline_slice, axis=1)
 
    # Avoid divisions/heights based on a null std
    baseline_std[baseline_std == 0] = 1e-6
 
    peaks_idx = [None] * n_events
    is_noise = np.zeros(n_events, dtype=bool)
 
    for i in range(n_events):
        segment = waveforms[i, baseline_points:]
        threshold = baseline_mean[i] + n_sigmas * baseline_std[i]
        
        peaks, properties = find_peaks(segment, height=threshold)
        if len(peaks) == 0:
            is_noise[i] = True
            continue
        
        if alignment == 'highest':
            chosen = peaks[np.argmax(properties['peak_heights'])]
        else:
            # same threshold, plus a minimum prominence relative to the highest peak
            highest_height = properties['peak_heights'].max()
            significant, _ = find_peaks(segment, height=threshold,
                                        prominence=prominence_fraction * (highest_height - baseline_mean[i]),
                                        distance=min_distance)
            chosen = significant[0] if len(significant) > 0 else peaks[np.argmax(properties['peak_heights'])]
        
        peaks_idx[i] = chosen + baseline_points  # Adjust index to account for the baseline we had neglected before 
 
    return peaks_idx, is_noise

        
def peaksAlignment(charge_waveforms, current_waveforms, baseline_points=300, n_sigmas=3, pre_trigger_points=75, post_trigger_points=124, keep_noise=True, alignment='first', prominence_fraction=PROMINENCE_FRACTION):
    
    current_waveforms = np.asarray(current_waveforms)
    charge_waveforms = np.asarray(charge_waveforms)
    n_events, n_samples = current_waveforms.shape
    window_len = pre_trigger_points + post_trigger_points + 1
    
    peaks_idx, is_noise = getPeaks(current_waveforms, baseline_points=baseline_points, n_sigmas=n_sigmas,
                                   alignment=alignment, prominence_fraction=prominence_fraction)
    
    good_idx, noise_idx, edge_idx = [], [], []
    rows = {}  # event index -> aligned window, filled only for "good" events
 
    for i in range(n_events):
        if is_noise[i]:
            noise_idx.append(i)
            continue
        
        peak = peaks_idx[i]
        start = peak - pre_trigger_points
        stop = peak + post_trigger_points + 1
        
        if start < 0 or stop > n_samples:
            edge_idx.append(i)
            continue
        
        rows[i] = (charge_waveforms[i, start:stop], current_waveforms[i, start:stop])
        good_idx.append(i)
 
    good_idx = np.array(good_idx, dtype=int)
    noise_idx = np.array(noise_idx, dtype=int)
    edge_idx = np.array(edge_idx, dtype=int)
    
    if keep_noise:
        charge_aligned = np.full((n_events, window_len), np.nan)
        current_aligned = np.full((n_events, window_len), np.nan)
        for i, (charge_row, current_row) in rows.items():
            charge_aligned[i] = charge_row
            current_aligned[i] = current_row
    else:
        charge_aligned = (
            np.stack([rows[i][0] for i in good_idx]) if len(good_idx) > 0
            else np.empty((0, window_len))
        )
        current_aligned = (
            np.stack([rows[i][1] for i in good_idx]) if len(good_idx) > 0
            else np.empty((0, window_len))
        )
    
    return charge_aligned, current_aligned, good_idx, noise_idx, edge_idx 


def newProcessing(waveforms, baseline_points=300, pre_trigger_points=75, post_trigger_points=124, n_sigmas=3,
                  keep_noise=True, order=4, normalise=True, alignment='first',
                  prominence_fraction=PROMINENCE_FRACTION, filter_type=FILTER_TYPE, gauss_sigma=GAUSS_SIGMA):
    """
    Full cleaning pipeline:
    baseline subtraction -> normalisation -> current -> LOW-PASS FILTER -> peak alignment.
    The charge that gets aligned is the normalised one, the current is the filtered one.
    
    normalise=False skips the normalisation: the charge keeps its amplitude (baseline subtracted
    only) and the current is computed from that charge. Every step is linear, and the peak
    threshold (baseline mean + n_sigmas * std) scales with the waveform, so the events kept and
    the alignment are exactly the same as with normalise=True.
    
    filter_type : 'gauss' (default) -> Gaussian filter, sigma = gauss_sigma samples
                  'butter'          -> Butterworth of order `order` with sosfiltfilt (old processing)
    alignment='first': the window is centred on the FIRST current peak with a prominence of at least
    prominence_fraction of the highest one (see getPeaks), with pre_trigger_points before it and
    post_trigger_points after it (default 75 + 1 + 124 = 200).
    
    Old processing (Butterworth, prominence 0.1): filter_type='butter', prominence_fraction=0.1
    """
    if filter_type not in ('gauss', 'butter'):
        raise ValueError("filter_type must be 'gauss' or 'butter'")
    
    # Step 1: Baseline subtraction
    baseline_subtracted = baseline(waveforms, baseline_points=baseline_points)
    
    # Step 2: Normalisation (optional)
    normalised_waveforms = normalisation(baseline_subtracted) if normalise else baseline_subtracted
    
    # Step 3: Current computation
    current_waveforms = current(normalised_waveforms, dt=1.0)
    
    # Step 4: Low-pass filter
    if filter_type == 'gauss':
        filtered_waveforms = gaussFilter(current_waveforms, sigma=gauss_sigma)
    else:
        filtered_waveforms = butterFilter(current_waveforms, order=order)
    
    # Step 5: Peak alignment
    charge_aligned, current_aligned, good_idx, noise_idx, edge_idx = peaksAlignment(
        normalised_waveforms, filtered_waveforms,
        baseline_points=baseline_points,
        n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points,
        post_trigger_points=post_trigger_points,
        keep_noise=keep_noise,
        alignment=alignment,
        prominence_fraction=prominence_fraction
    )
    
    return charge_aligned, current_aligned, good_idx, noise_idx, edge_idx


def computeScale(waveforms):
    '''
    Rescale the current waveform in order not ot destroy the E/E dependece but to get 
    better inputs for the VAE
    '''
    
    # Find the maximum for each waveform
    max_peaks = np.max(waveforms, axis=1)
    
    scale = float(np.median(max_peaks))

    return scale


# NOTE: the rise time is not computed here. The official rise time (tp_90 - tp_10 from the psp
# tier) is read by filereading and realigned in preliminary.clean_data like energy and A/E.

####################################################################################################
# 2. LOADERS
####################################################################################################

# ------------- NORMAL FIT -------------
def gaussian_model(x, amp, mu, sigma, bkg):
    return amp * norm.pdf(x, loc=mu, scale=sigma) + bkg #densità di probabilità della distribuzione normale (gaussiana) + background costante

def gaussian(data, xmin, xmax, n=None):
    """
    Fits a Gaussian to `data` restricted to [xmin, xmax].
    Bins are computed as np.arange(int(data.min()), int(data.max())+1, 1).
    No plotting, no residuals computed.

    Returns:
        parameters   -> np.array([amp, mu, sigma])
        uncertainties  -> np.array([amp_unc, mu_unc, sigma_unc])
    """
    bins = np.arange(int(data.min()), int(data.max())+1, 1)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fit_mask = (bin_centers >= xmin) & (bin_centers <= xmax)
    bin_centers_fit = bin_centers[fit_mask]
    counts_fit = counts[fit_mask]
    sigma_counts_fit = np.maximum(np.sqrt(counts_fit), 1)

    initial_guess = [max(counts_fit), np.mean(bin_centers_fit), np.std(bin_centers_fit), counts_fit.min()]
    params, cov_matrix = curve_fit(gaussian_model, bin_centers_fit, counts_fit, p0=initial_guess, sigma=sigma_counts_fit)
    amp, mu, sigma, bkg = params
    uncertainties = np.sqrt(np.diag(cov_matrix))
    parameters = np.array([amp, mu, sigma, bkg])

    return parameters, uncertainties


# ------------ MULTICHANNEL to use current AND charge -------------
def multichannel(charge, current, use_charge=False):
    
    if use_charge:
        combined = np.stack([current, charge], axis=1) # axis=1 creates a third dimension with dimension of the channels
        # output (number of events, channels=2, length=200)
        
    else:
        combined = np.expand_dims(current, axis=1) # insert an empty dimension for the channel
        # output (number of events, channels=1, length=200)

    return combined
    


# ------------- EPOCH TENSOR CREATION -------------

def epoch_loader(waves_train, labels_train, channels=1, batch_size=64, shuffle=True, seed=None): 
    
    assert waves_train.shape[1] == channels, \
        f"Expected {channels} channels, data have {waves_train.shape[1]} channels. " \
        f"Check use_charge."
    
    waves_train  = torch.as_tensor(waves_train,  dtype=torch.float32)
    labels_train = torch.as_tensor(labels_train, dtype=torch.long)
    
    dataset = TensorDataset(waves_train, labels_train)
    
    # un generatore dedicato, non lo stato globale di torch: così possiamo
    # "riavvolgerlo" più avanti senza toccare nient'altro nel codice
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=generator)
    return loader


# -------------- TEST LOADER CREATION --------------
def test_loader(waves_test, labels_test, channels=1, batch_size=64):
    
    assert waves_test.shape[1] == channels, \
    f"Expected {channels} channels, data have {waves_test.shape[1]} channels. " \
    f"Check use_charge."
    
    waves_test  = torch.as_tensor(waves_test,  dtype=torch.float32)
    labels_test = torch.as_tensor(labels_test, dtype=torch.long)

    dataset = TensorDataset(waves_test, labels_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) # in this point dropping last would compromise the analysis
    

    return loader


####################################################################################################
# 3a. PLOTTING: STYLE, SAVING, CLUSTERING PLOTS (DTC, SOM)
####################################################################################################

# size of the axis labels (xlabel / ylabel) in ALL the plots of this module
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13   # numbers on the axes (uncomment if needed)
plt.rcParams['ytick.labelsize'] = 13


# ------------------------------------------------------------------------------------------
# ------------------------------------ SAVING THE PLOTS ------------------------------------
# ------------------------------------------------------------------------------------------
# The plots are saved in PLOT_DIR/<PREFIX><name>.pdf and overwritten every time.
# In the notebook, after the imports: ut.PLOT_DIR, ut.TAG = 'acf' (colors), ut.PREFIX = 'acf_' (file names)
PLOT_DIR = detector.DTC_PLOTS
TAG = ''        # metric of the plots: only for the colors (see METRIC_COLORS)
PREFIX = ''     # put in front of every file name, e.g. 'eucl_' -> PLOT_DIR/eucl_2cl_energy_test.pdf


def save(fig, name):
    """All the plots directly in PLOT_DIR (no subfolders), as PREFIX + name + '.pdf'."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, f"{PREFIX}{name}.pdf"), dpi=300)


# ------------------------------------------------------------------------------------------
# ----------------------------------------- COLORS -----------------------------------------
# ------------------------------------------------------------------------------------------
color1, color2, color3, color4, color5, color6, color7, color8 = '#ABB1BC', '#013162', "#ffbf00", '#fe2f33', '#006aa4', "#D602B3", "#82bf00d4", "#FF6F00"


PANEL = (9, 6)       # width and height of EACH panel with 1 or 2 panels: the figure is (9 * n_panels) x 6
PANEL_SQUARE = (6, 6)  # with 3 or more panels every panel is square: the figure is (6 * n_panels) x 6


def panel_size(n_panels):
    """(9, 6) for 1 or 2 panels side by side, (6, 6) for 3 or more."""
    return PANEL_SQUARE if n_panels >= 3 else PANEL

# ---------------- cluster colors ----------------
# The cluster with the most events in the DEP window gets the color of the metric (read from TAG),
# the other ones are black and, with 3 clusters, grey. At most 3 clusters.
METRIC_COLORS = {'acf': color6, 'eucl': color7, 'euclidean': color7, 'cid': color8}
OTHER_COLORS  = ['#4e4e4e', '#8c8c8c', '#c4c4c4']
DEP_WINDOW    = (1587, 1598)       # keV, same range used for the DEP gaussian fit


def metric_color():
    """Color of the metric of this notebook, from ut.TAG. No silent default: a wrong color is worse than an error."""
    tag = str(TAG).strip().lower()
    if tag not in METRIC_COLORS:
        raise ValueError(f"ut.TAG = {TAG!r}: set ut.TAG = 'acf', 'cid' or 'eucl' AFTER the import cell "
                         f"(reloading l200_utils resets TAG to '')")
    return METRIC_COLORS[tag]


def cluster_colors(y, energy, n_clusters):
    """One color per cluster: metric color for the cluster with the most events in DEP_WINDOW,
    black (and grey) for the others, in order of cluster index.
    y and energy: same events, same order (e.g. y2 and data['energy'])."""
    assert n_clusters <= 1 + len(OTHER_COLORS), f"at most {1 + len(OTHER_COLORS)} clusters are supported"
    y, energy = np.asarray(y), np.asarray(energy)
    in_dep = (energy >= DEP_WINDOW[0]) & (energy <= DEP_WINDOW[1])
    sse = int(np.argmax([np.sum(y[in_dep] == k) for k in range(n_clusters)]))
    others = iter(OTHER_COLORS)
    return [metric_color() if k == sse else next(others) for k in range(n_clusters)]


def _colors(colors, n_clusters):
    """colors given -> used as they are; None -> metric color for cluster 0, black/grey for the others."""
    assert n_clusters <= 1 + len(OTHER_COLORS), f"at most {1 + len(OTHER_COLORS)} clusters are supported"
    return colors if colors is not None else [metric_color()] + OTHER_COLORS[:n_clusters - 1]


def _risetime_bins(values, value_range, max_bins=80):
    """Bin edges for the rise time. tp_90 - tp_10 takes only discrete values (multiples of the
    sampling step): the edges are put halfway between them and every bin holds the same number
    of possible values, so there are no empty bins or spikes. Not discrete -> 'fd' rule."""
    v = values[np.isfinite(values)]
    lo, hi = value_range
    u = np.unique(v)
    d = np.diff(u)
    if len(d) > 0 and d.min() > 0 and np.allclose(d / d.min(), np.round(d / d.min()), atol=1e-3):
        step = d.min()
        width = step * max(1, int(np.ceil((hi - lo) / step / max_bins)))
        start = u[0] + step * np.floor((lo - u[0]) / step) - step / 2
        return np.arange(start, hi + width, width)
    return np.histogram_bin_edges(v, bins='fd', range=value_range)


LOSS_HEIGHT = 4        # height of the loss plots (pretraining and training history): flatter than the others


def make_panels(n_panels, height=None):
    """height=None -> standard height of panel_size; otherwise the given height (the width does not change)."""
    w, h = panel_size(n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(w * n_panels, h if height is None else height), squeeze=False)
    return fig, axes[0]


# ------------------------------------------------------------------------------------------
# ------------------------------------- DATA CLEANING --------------------------------------
# ------------------------------------------------------------------------------------------
def plot_aligned_currents(current_aligned, n_waveforms=100, name='aligned_currents'):
    """First n_waveforms aligned current waveforms, superimposed."""
    
    fig = plt.figure()

    for i in range(len(current_aligned)):
        if i < n_waveforms:
            plt.plot(current_aligned[i], alpha=0.2, color=color2)
    plt.grid(alpha=0.3, ls='--')
    plt.gca().set_axisbelow(True)   # grid behind the data
    save(fig, name)
    plt.show()


# ------------------------------------------------------------------------------------------
# ---------------------- RECONSTRUCTION (used by pretrain and training) ---------------------
# ------------------------------------------------------------------------------------------
CHANNEL_NAMES = ('current', 'charge')          # same order as multichannel
Y_LABELS = ('Current [a.u.]', 'Charge [a.u.]')
ORIGINAL_COLOR = '#013162'                      # dark blue
RECONSTRUCTED_COLOR = '#ff9100'                 # yellow
# size of EACH single panel: given by panel_size(n_cols), (9, 6) with 1-2 columns, (6, 6) with 3 or more

def plot_reconstruction_grid(original, reconstructions, titles, dt_ns=16):
    '''
    One column for each reconstruction, one row for each channel:
    first row = current, second row (only if present) = charge.
    Every single panel has size panel_size(n_cols): (9, 6) with 1-2 columns, (6, 6) with 3 or more.

    original:        array (channels, length)
    reconstructions: list of arrays (channels, length), one per column
    titles:          list of strings, one per column
    '''
    original = np.atleast_2d(original)
    n_channels = original.shape[0]
    n_cols = len(reconstructions)
    x_ns = np.arange(original.shape[-1]) * dt_ns

    w, h = panel_size(n_cols)
    fig, axes = plt.subplots(n_channels, n_cols,
                             figsize=(w * n_cols, h * n_channels),
                             sharex=True, sharey='row', squeeze=False)

    for col, (rec, title) in enumerate(zip(reconstructions, titles)):
        rec = np.atleast_2d(rec)
        for c in range(n_channels):
            ax = axes[c, col]
            ax.plot(x_ns, original[c], color=ORIGINAL_COLOR, lw=1.5, label='Original')
            ax.plot(x_ns, rec[c], color=RECONSTRUCTED_COLOR, lw=1.5, label='Reconstructed')
            ax.set_title(f'{title} - {CHANNEL_NAMES[c]}')
            ax.grid(alpha=0.3, ls='--')
            ax.set_axisbelow(True)   # grid behind the data
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('Time [ns]')
            if col == 0:
                ax.set_ylabel(Y_LABELS[c])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    return fig, axes, handles, labels


# ------------------------------------------------------------------------------------------
# ----------------------------------------- LOSSES -----------------------------------------
# ------------------------------------------------------------------------------------------
def _log_above_linear_below(ax, values, neg_fraction=0.4):
    """y axis with LOG scale above zero and LINEAR scale below zero.
    Positive values: log10(y), from 10^0 up. Values below 1 (negative ones included): linear,
    squeezed into neg_fraction of the axis height and joined continuously at y = 1."""
    values = np.asarray(values, dtype=float)
    y_max, y_min = values.max(), min(values.min(), 0.0)
    decades = max(np.log10(max(y_max, 10.0)), 1.0)
    h = neg_fraction / (1 - neg_fraction) * decades        # height of the linear part, in decades
    span = 1.0 - y_min                                     # linear part: from y_min to 1
    
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y >= 1, np.log10(np.maximum(y, 1)), (y - 1) / span * h)
    
    def inverse(z):
        z = np.asarray(z, dtype=float)
        return np.where(z >= 0, 10 ** np.clip(z, 0, None), 1 + z / h * span)
    
    ax.set_yscale('function', functions=(forward, inverse))
    neg_ticks = [t for t in MaxNLocator(4).tick_values(y_min, 0) if y_min <= t <= 0]
    pos_ticks = [10 ** p for p in range(1, int(np.ceil(np.log10(max(y_max, 10)))) + 1)]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(neg_ticks + pos_ticks))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"$10^{{{int(round(np.log10(y)))}}}$" if y >= 10 else f"{y:g}"))
    ax.axhline(0, color='black', lw=0.6, alpha=0.5)        # where the scale changes


def plot_pretrain_losses(pre_train, pre_val, name='pretrain_losses', zoom_from=None):
    """Left: whole history (symlog, the first epochs span orders of magnitude).
    Right: linear zoom on the converged part, from epoch zoom_from.
    zoom_from=None -> first epoch where both curves are close to the values of the last half of the training."""
    
    pre_train, pre_val = np.asarray(pre_train), np.asarray(pre_val)
    epochs = np.arange(1, len(pre_train) + 1)
    
    if zoom_from is None:
        tail = np.r_[pre_train[len(pre_train) // 2:], pre_val[len(pre_val) // 2:]]
        lo, hi = tail.min(), tail.max()
        margin = 3 * (hi - lo)
        inside = (pre_train >= lo - margin) & (pre_train <= hi + margin) & \
                 (pre_val   >= lo - margin) & (pre_val   <= hi + margin)
        # first epoch from which the curves stay inside the band
        zoom_from = int(epochs[np.argmax(np.cumprod(inside[::-1])[::-1].astype(bool))])
    z = epochs >= zoom_from
    
    fig, axes = make_panels(2)
    
    # ---- whole history
    axes[0].plot(epochs, pre_train, color=color2, lw=1.8, marker='o', ms=4, label='Train')
    axes[0].plot(epochs, pre_val,   color=color3, lw=1.8, marker='o', ms=4, label='Validation')
    _log_above_linear_below(axes[0], np.r_[pre_train, pre_val])
    axes[0].axvspan(zoom_from - 0.5, epochs[-1] + 0.5, color=color1, alpha=0.2, lw=0, label='zoom (right)')
    axes[0].set_title('Pretrain loss')
    
    # ---- zoom on the converged part
    axes[1].plot(epochs[z], pre_train[z], color=color2, lw=1.8, marker='o', ms=4,
                 label=f'Train (final {pre_train[-1]:.1f})')
    axes[1].plot(epochs[z], pre_val[z],   color=color3, lw=1.8, marker='o', ms=4,
                 label=f'Validation (final {pre_val[-1]:.1f})')
    axes[1].set_title(f'Pretrain loss - from epoch {zoom_from}')
    
    for ax in axes:
        ax.set_ylabel('NLL')
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
        ax.legend(frameon=False)
    
    plt.tight_layout()
    save(fig, name)
    plt.show()


def plot_training_history(history, name='training_history'):
    
    fig, axes = make_panels(3, height=LOSS_HEIGHT)
    epochs = np.arange(1, len(history['nll_train']) + 1)
    
    axes[0].plot(epochs, history['nll_train'], color=color2, lw=1.8, marker='o', label='Train')
    axes[0].plot(epochs, history['nll_val'],   color=color3, lw=1.8, marker='o', label='Validation')
    axes[0].set_title('Reconstruction loss (NLL)')
    axes[0].set_ylabel('NLL')
    
    axes[1].plot(epochs, history['kl_train'], color=color2, lw=1.8, marker='o', label='Train')
    axes[1].plot(epochs, history['kl_val'],   color=color3, lw=1.8, marker='o', label='Validation')
    axes[1].set_title('Clustering loss (KL)')
    axes[1].set_ylabel('KL')
    
    axes[2].plot(epochs, 100 * np.array(history['changes']), color=color2, lw=1.8, marker='o')
    axes[2].set_title('Events changing cluster since previous epoch')
    axes[2].set_ylabel('[%]')
    
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    
    plt.tight_layout()
    save(fig, name)
    plt.show()


# ------------------------------------------------------------------------------------------
# ---------------------------------------- CLUSTERS ----------------------------------------
# ------------------------------------------------------------------------------------------
def plot_clusters_vs_energy(y_all, energy, n_clusters, title="", bins=np.arange(1550, 1651, 1.0), name='clusters_vs_energy', colors=None):
    
    if colors is None:
        colors = cluster_colors(y_all, energy, n_clusters)
    centers = 0.5 * (bins[1:] + bins[:-1])
    total, _ = np.histogram(energy, bins=bins)
    
    fig, axes = make_panels(2)
    
    # panel 1: total spectrum (filled + black outline, one patch in the legend) and spectrum of every cluster
    axes[0].stairs(total, bins, fill=True, color=color1, alpha=0.2)
    axes[0].stairs(total, bins, color='black', alpha=0.8)
    total_patch = Patch(facecolor=to_rgba(color1, 0.2), edgecolor=to_rgba('black', 0.8), linewidth=1, label='all events')
    for k in range(n_clusters):
        h, _ = np.histogram(energy[y_all == k], bins=bins)
        axes[0].step(centers, h, where='mid', color=colors[k], lw=1.5,
                     label=f'cluster {k} ({np.sum(y_all == k)} events)')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Counts / keV')
    axes[0].set_xlim(1550, 1649)
    
    # panel 2: fraction of the events of every cluster, energy bin by energy bin
    for k in range(n_clusters):
        h, _ = np.histogram(energy[y_all == k], bins=bins)
        f = h / np.maximum(total, 1)
        axes[1].plot(centers, f, color=colors[k], marker='o', ms=3, lw=1,
                     label=f'cluster {k}')
    axes[1].set_ylabel('Fraction of events in the cluster')
    
    for ax in axes:
        ax.axvline(1592.5, color=color4, ls='--', lw=1.5, label=r'DEP of ${}^{208}$Tl')
        ax.axvline(1620.7, color=color5, ls='--', lw=1.5, label=r'FEP of ${}^{212}$Bi')
        ax.set_xlabel('Energy [keV]')
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([total_patch] + handles, ['all events'] + labels, frameon=False)
    axes[1].legend(frameon=False)
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()


def plot_cluster_prototypes(y, waves, n_clusters, title="", name='cluster_prototypes', colors=None):
    """Mean current waveform of every cluster: the 'prototype' in the data space."""
    
    colors = _colors(colors, n_clusters)
    assert waves.shape[0] == len(y), "different lengths: waves and y must have the same events"
    
    fig, axes = make_panels(1)
    x_ns = np.arange(waves.shape[-1]) * 16
    
    for k in range(n_clusters):
        m = y == k
        if m.sum() == 0:          # empty cluster: no mean to plot
            continue
        axes[0].plot(x_ns, waves[m, 0, :].mean(axis=0), color=colors[k], lw=1.8,
                     label=f"cluster {k} ({m.sum()} events)")
    
    axes[0].set_title("Mean current waveform per cluster")
    axes[0].set_xlabel("Time [ns]")
    axes[0].set_ylabel("Current [a.u.]")
    axes[0].grid(alpha=0.3, ls='--')
    axes[0].set_axisbelow(True)   # grid behind the data
    axes[0].legend(frameon=False)
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()


def plot_latent_centroids(model, title="", name='latent_centroids', colors=None):
    """Cluster centroids of the clustering layer: the 'prototype' in the latent space,
    one sequence of (input_length // pool_size) points per cluster."""
    
    centroids = model.tsc.clusters.detach().cpu().numpy()      # (n_clusters, latent_length, n_features)
    n_clusters, latent_length, n_features = centroids.shape
    colors = _colors(colors, n_clusters)
    
    fig, axes = make_panels(1)
    x = np.arange(latent_length)
    
    for k in range(n_clusters):
        for f in range(n_features):
            label = f"cluster {k}" if f == 0 else None
            axes[0].plot(x, centroids[k, :, f], color=colors[k], lw=1.8, marker='o', ms=4, label=label)
    
    axes[0].set_title(f"Cluster centroids in the latent space ({latent_length} points, {n_features} feature/s)")
    axes[0].set_xlabel("Latent time step")
    axes[0].set_ylabel("Latent value [a.u.]")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=0.3, ls='--')
    axes[0].set_axisbelow(True)   # grid behind the data
    axes[0].legend(frameon=False)
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()


def plot_cluster_variables(y, AoE, rise_time, n_clusters, title="", name='cluster_variables', colors=None):
    """Physical variables of the classical PSD, cluster by cluster: A/E and rise time (counts, not normalised)."""
    
    assert len(y) == len(AoE) == len(rise_time), "different lengths: y, AoE and rise_time must have the same events"
    colors = _colors(colors, n_clusters)
    
    a_range = (np.nanquantile(AoE, 0.01), np.nanquantile(AoE, 0.99))
    r_range = (np.nanquantile(rise_time, 0.01), np.nanquantile(rise_time, 0.99))
    a_bins = np.linspace(*a_range, 101)
    r_bins = _risetime_bins(rise_time, r_range)
    
    fig, axes = make_panels(2)
    for k in range(n_clusters):
        m = y == k
        a_k = AoE[m][np.isfinite(AoE[m])]
        r_k = rise_time[m][np.isfinite(rise_time[m])]          # NaN = rise time not computable
        if len(a_k) > 0:                                       # empty cluster: nothing to plot
            axes[0].hist(a_k, bins=a_bins, histtype='step',
                         color=colors[k], lw=1.5, label=f"cluster {k} ({m.sum()} events)")
        if len(r_k) > 0:
            axes[1].hist(r_k, bins=r_bins, histtype='step',
                         color=colors[k], lw=1.5, label=f"cluster {k} ({m.sum()} events)")
    
    axes[0].set_title("A/E per cluster")        
    axes[0].set_xlabel("A/E")
    axes[1].set_title("Rise time (charge, 10%-90%) per cluster")
    axes[1].set_xlabel("Rise time [ns]")
    axes[0].set_ylabel(f"Counts / {a_bins[1] - a_bins[0]:.2g}")
    axes[1].set_ylabel(f"Counts / {r_bins[1] - r_bins[0]:.0f} ns")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
        ax.legend(frameon=False)
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()
    
    
def plot_cluster_waveforms(y, waves, n_clusters, title="", n_samples=60, seed=0, channel=0, alpha=0.15, name='cluster_waveforms', colors=None):
    """A sample of individual waveforms per cluster, one panel per cluster: shows the spread
    of shapes behind the mean (plot_cluster_prototypes shows only the mean).
    Empty clusters get no panel. All the panels share the same y scale, so that the
    different heights of the peaks are directly comparable."""
    
    assert waves.shape[0] == len(y), "different lengths: waves and y must have the same events"
    
    colors = _colors(colors, n_clusters)
    rng = np.random.default_rng(seed)
    x_ns = np.arange(waves.shape[-1]) * 16
    
    # only the non-empty clusters get a panel
    non_empty = [k for k in range(n_clusters) if np.sum(y == k) > 0]
    empty = [k for k in range(n_clusters) if k not in non_empty]
    if empty:
        print(f"empty clusters, not plotted: {empty}")
    if len(non_empty) == 0:
        return
    
    fig, axes = make_panels(len(non_empty))
    y_min, y_max = np.inf, -np.inf
    
    for ax, k in zip(axes, non_empty):
        idx = np.where(y == k)[0]
        n_show = min(n_samples, len(idx))
        
        sample = rng.choice(idx, size=n_show, replace=False)
        for i in sample:
            ax.plot(x_ns, waves[i, channel, :], color=colors[k], lw=0.8, alpha=alpha)
        y_min = min(y_min, np.nanmin(waves[sample, channel, :]))
        y_max = max(y_max, np.nanmax(waves[sample, channel, :]))
        
        ax.set_title(f"cluster {k} ({len(idx)} events, {n_show} shown)")
        ax.set_xlabel("Time [ns]")
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
    
    # same y scale for every panel (5% margin)
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    for ax in axes:
        ax.set_ylim(y_min - margin, y_max + margin)
    axes[0].set_ylabel(Y_LABELS[channel])
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()


def plot_cluster_peaks(y, n_peaks, n_clusters, title="", name=None, colors=None):
    """colors: pass ut.cluster_colors(y, data['energy'], n_clusters) to get the metric color on the DEP-rich cluster."""
    
    assert len(y) == len(n_peaks), "different lengths: y and n_peaks must have the same events"
    colors = _colors(colors, n_clusters)
    if name is None:
        name = f"{n_clusters}cl_peaks"
    
    fig, axes = make_panels(2)
    max_peaks = max(int(n_peaks.max()), 2)
    edges = np.arange(0.5, min(max_peaks, 6) + 1.5, 1.0)
    
    # panel 1: distribution of the number of peaks, cluster by cluster
    for k in range(n_clusters):
        m = y == k
        axes[0].hist(np.clip(n_peaks[m], 0, min(max_peaks, 6)), bins=edges, histtype='step', density=True,
                     color=colors[k], lw=1.5, label=f"cluster {k}")
    axes[0].set_title("Number of peaks per cluster")
    axes[0].set_xlabel("Number of peaks")
    axes[0].set_ylabel("Normalised counts")
    axes[0].legend(frameon=False)
    
    # panel 2: fraction of multi-peak events, cluster by cluster
    sizes = np.array([(y == k).sum() for k in range(n_clusters)])
    fractions = np.array([(n_peaks[y == k] >= 2).mean() for k in range(n_clusters)])
    errors = np.sqrt(fractions * (1 - fractions) / np.maximum(sizes, 1))
    axes[1].bar(np.arange(n_clusters), fractions, yerr=errors, capsize=4,
                color=[colors[k] for k in range(n_clusters)])
    axes[1].set_title(r"Fraction of events with $\geq 2$ peaks")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Fraction")
    axes[1].set_xticks(np.arange(n_clusters))
    
    for ax in axes:
        ax.grid(alpha=0.3, ls='--')
        ax.set_axisbelow(True)   # grid behind the data
    
    fig.suptitle(title)
    plt.tight_layout()
    save(fig, name)
    plt.show()


# ------------------------------------------------------------------------------------------
# ------------------------------------- FULL REPORT ----------------------------------------
# ------------------------------------------------------------------------------------------
SUBSET_NAMES = {'train': 'training set', 'val': 'validation set', 'test': 'test set',
                'all': 'whole calibration dataset'}


def _title(n_clusters, subset, n_events, stage='after DTC training'):
    """Same title in every plot: '<n> clusters - <stage> - <subset name> (<n> events)'."""
    return f"{n_clusters} clusters - {stage} - {SUBSET_NAMES[subset]} ({n_events} events)"


def _cluster_shape_plots(y, waves, AoE, rise_time, n_clusters, tag, colors):
    """Prototypes, A/E and rise time, example waveforms (current and, if present, charge)
    for the events passed in. tag: 'train', 'val', 'test' or 'all' (titles and file names)."""
    
    head = _title(n_clusters, tag, len(y))
    plot_cluster_prototypes(y, waves, n_clusters, title=head, name=f"{n_clusters}cl_prototypes_{tag}", colors=colors)
    plot_cluster_variables(y, AoE, rise_time, n_clusters, title=head, name=f"{n_clusters}cl_AoE_risetime_{tag}", colors=colors)
    plot_cluster_waveforms(y, waves, n_clusters, title=f"{head} - examples of clustered waveforms (current)",
                           n_samples=100, seed=0, channel=0, alpha=0.2, name=f"{n_clusters}cl_waveforms_current_{tag}", colors=colors)
    if waves.shape[1] == 2:       # second channel = charge
        plot_cluster_waveforms(y, waves, n_clusters, title=f"{head} - examples of clustered waveforms (charge)",
                               n_samples=100, seed=0, channel=1, alpha=0.2, name=f"{n_clusters}cl_waveforms_charge_{tag}", colors=colors)


def cluster_report(model, y_init, y_after, n_clusters, data):
    """All the plots of a run, in the same order for every number of clusters.
    - training history
    - clusters vs energy, separately for the training, validation and test set
    - prototypes, A/E and rise time, example waveforms: on the TEST set only (events never seen in training)
    The same plots on ALL the events are in cluster_report_all (to be called in its own cell).
    data is the dictionary returned by preliminary.create_loaders.
    y_init and y_after are the assignments of ALL the events (all_loader), same order as data."""
    
    waves, energy, AoE, rise_time = data['waves'], data['energy'], data['AoE'], data['rise_time']
    colors = cluster_colors(y_after, energy, n_clusters)     # from ALL the events: same colors in every plot
    
    plot_training_history(model.history, name=f"{n_clusters}cl_training_history")
    for split in ('train', 'val', 'test'):
        idx = data[f'idx_{split}']
        plot_clusters_vs_energy(y_after[idx], energy[idx], n_clusters=n_clusters,
                                title=_title(n_clusters, split, len(idx)),
                                name=f"{n_clusters}cl_energy_{split}", colors=colors)
    # plot_latent_centroids(model, title=f"{n_clusters} clusters - after DTC training")
    
    idx = data['idx_test']
    _cluster_shape_plots(y_after[idx], waves[idx], AoE[idx], rise_time[idx], n_clusters, tag="test", colors=colors)


def cluster_report_all(y_init, y_after, n_clusters, data):
    """The same plots on ALL the events: clusters vs energy (right after KMeans and after the DTC training),
    prototypes, A/E and rise time, example waveforms."""
    
    waves, energy, AoE, rise_time = data['waves'], data['energy'], data['AoE'], data['rise_time']
    colors = cluster_colors(y_after, energy, n_clusters)
    
    plot_clusters_vs_energy(y_init,  energy, n_clusters=n_clusters,
                            title=_title(n_clusters, 'all', len(energy), stage='right after KMeans'),
                            name=f"{n_clusters}cl_energy_all_kmeans")
    plot_clusters_vs_energy(y_after, energy, n_clusters=n_clusters,
                            title=_title(n_clusters, 'all', len(energy)),
                            name=f"{n_clusters}cl_energy_all_dtc", colors=colors)
    _cluster_shape_plots(y_after, waves, AoE, rise_time, n_clusters, tag="all", colors=colors)
    
    
def plot_2nu_clusters(res, e_min=560, e_max=2039, bin_width=5, sse_color=color2, mse_color=color3):
    """
    Three separate figures:
      1. energy spectrum of every cluster
      2. fraction of every cluster vs energy (points joined by lines, no error bars)
      3. A/E classifier of every cluster (only if A/E is available)
    """
 
    y, energy, AoE, n_clusters = res['y'], res['energy'], res['AoE'], res['n_clusters']
    bins = np.arange(e_min, e_max + bin_width, bin_width)
    centres = 0.5 * (bins[:-1] + bins[1:])
    total, _ = np.histogram(energy, bins=bins)
    clusters = [k for k in range(n_clusters) if (y == k).sum() > 0]
    colors=[color2, color3]
 
    # ---------------- 1) energy spectrum per cluster ----------------
    fig = plt.figure(figsize=(9, 6))
    for k in clusters:
        sel = y == k
        counts, _ = np.histogram(energy[sel], bins=bins)
        plt.stairs(counts / bin_width, bins, lw=1.5,
                   label=f"cluster {k} ({sel.mean()*100:.1f} %)", color=colors[k])
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts / 5 keV")
    plt.yscale('log')
    plt.xlim(e_min, e_max)
    plt.title(fr"$2\nu\beta\beta$: {len(y)} events")
    plt.gca().set_axisbelow(True)   # grid behind the data
    plt.grid(alpha=0.3, ls='--')
    plt.legend(frameon=False)
    plt.tight_layout()
    save(fig, f"2nu_{n_clusters}cl_spectrum")
    plt.show()
 
    # ---------------- 2) fraction of every cluster vs energy ----------------
    fig = plt.figure(figsize=(9, 6))
    for k in clusters:
        counts, _ = np.histogram(energy[y == k], bins=bins)
        frac = np.divide(counts, total, out=np.full(len(counts), np.nan), where=total > 0)
        plt.plot(centres, frac, 'o-', ms=4, color=colors[k], label=f"cluster {k}")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Fraction of events in the cluster")
    plt.ylim(-0.1, 1.1)
    plt.xlim(e_min, e_max)
    plt.title(rf"$2\nu\beta\beta$ - fraction per cluster vs energy")
    plt.gca().set_axisbelow(True)
    plt.grid(alpha=0.3, ls='--')
    plt.legend(frameon=False)
    plt.tight_layout()
    save(fig, f"2nu_{n_clusters}cl_fraction")
    plt.show()
 
    # ---------------- 3) A/E per cluster ----------------
    if np.isfinite(AoE).any():
        aoe_bins = np.linspace(np.nanpercentile(AoE, 0.5), np.nanpercentile(AoE, 99.5), 60)
        fig = plt.figure(figsize=(9, 6))
        for k in clusters:
            a = AoE[(y == k) & np.isfinite(AoE)]
            plt.hist(a, bins=aoe_bins, histtype='step', density=True, lw=1.5,
                     color=colors[k], label=f"cluster {k}")
        plt.xlabel("A/E classifier")
        plt.ylabel("Density")
        plt.title(rf"$2\nu\beta\beta$ - A/E classifier per cluster")
        plt.gca().set_axisbelow(True)   # grid behind the data
        plt.grid(alpha=0.3, ls='--')
        plt.legend(frameon=False)
        plt.tight_layout()
        save(fig, f"2nu_{n_clusters}cl_AoE")
        plt.show()
    else:
        print("A/E not available: A/E plot skipped.")

####################################################################################################
# 3b. PLOTTING: VAE ANALYSIS
####################################################################################################



# -----------------------------------------------------------------------------
# ----------------------------------- STYLE ------------------------------------
# -----------------------------------------------------------------------------

color9 = "#2B6C50"          # color1 ... color8: same as the DTC section

PALETTE = [color1, color2, color3, color4, color5, color6, color7, color8, color9]

SSE_COLOR, MSE_COLOR = color4, color5
TRAIN_COLOR, VAL_COLOR = color2, color3
VAE_ORIGINAL_COLOR, VAE_RECONSTRUCTED_COLOR = color2, color3     # (DTC: ORIGINAL_COLOR, RECONSTRUCTED_COLOR)
CLF_COLOR, AOE_COLOR, PROBE_COLOR = color3, color2, color8

SSE_LABEL = r"${}^{208}$Tl DEP"
MSE_LABEL = r"${}^{212}$Bi FEP"

plt.rcParams['axes.axisbelow'] = True   # grid always behind the data, in every plot

SPECTRUM_FILL_ALPHA = 0.2   # grey fill of the full energy spectrum

DT_NS = 16                                      # sampling time
# CHANNEL_NAMES, Y_LABELS: same as the DTC section


def window_label(sigma):
    """The ONE text used in every title for the energy windows: '(±3σ windows)'."""
    return rf'(±{sigma:g}$\sigma$ windows)'


def _window(sigma):
    """' (±3σ windows)' to append to a title, or '' when sigma is None."""
    return f' {window_label(sigma)}' if sigma is not None else ''


# panel_size(n_cols): same as the DTC section (6x6 per panel with 3+ panels side by side, 9x6 with 1 or 2)


def make_figure(n_cols=1, n_rows=1, **kwargs):
    w, h = panel_size(n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), squeeze=False, **kwargs)
    return fig, axes


def _style(ax, title=None, xlabel=None, ylabel=None, legend=True, legend_loc='best'):
    if title:
        ax.set_title(title, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, ls='--')
    if legend:
        ax.legend(frameon=False, loc=legend_loc)


def _finish(fig, path=None, top=None):
    """
    tight_layout, then (if top is given) the panels are moved so that the highest panel
    title sits exactly at `top` (figure fraction): no extra empty band under the legend.
    """
    fig.tight_layout()
    if top is not None:
        renderer = fig.canvas.get_renderer()
        height_px = fig.get_figheight() * fig.dpi
        panels_top = max(ax.get_tightbbox(renderer).y1 for ax in fig.axes) / height_px
        fig.subplots_adjust(top=fig.subplotpars.top + (top - panels_top))
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(path, dpi=300)
    plt.show()
    plt.close(fig)


TITLE_GAP_IN = 0.05   # vertical gaps (inches) between the main title, the legend and the panels


def _top_title_and_legend(fig, title, handles, labels):
    """
    Title on top and ONE horizontal legend right below it (multi-panel figures).
    Positions are measured on the real text sizes, so the gaps are the same (TITLE_GAP_IN)
    whatever the figure height. Returns the top of the area left for the panels, to be
    passed to _finish as top=top.
    """
    title = f'{title} - {detector.BOLD_NAME}'
    height = fig.get_figheight()
    renderer = fig.canvas.get_renderer()
    to_fraction = lambda y_pixels: y_pixels / fig.dpi / height

    suptitle = fig.suptitle(title, x=0.5, y=1 - TITLE_GAP_IN / height, fontsize=16, ha='center', va='top')
    title_bottom = to_fraction(suptitle.get_window_extent(renderer).y0)

    legend = fig.legend(handles, labels, loc='upper center',
                        bbox_to_anchor=(0.5, title_bottom - TITLE_GAP_IN / height),
                        ncol=len(labels), frameon=False, fontsize=12, borderaxespad=0)
    legend_bottom = to_fraction(legend.get_window_extent(renderer).y0)

    return legend_bottom - TITLE_GAP_IN / height


def _top_title(fig, title):
    """Main title only (no figure legend), same measured spacing as _top_title_and_legend."""
    title = f'{title} - {detector.BOLD_NAME}'
    height = fig.get_figheight()
    renderer = fig.canvas.get_renderer()
    suptitle = fig.suptitle(title, x=0.5, y=1 - TITLE_GAP_IN / height, fontsize=14, ha='center', va='top')
    return suptitle.get_window_extent(renderer).y0 / fig.dpi / height - TITLE_GAP_IN / height


def plot_palette(path=None):
    fig, axes = make_figure(1)
    ax = axes[0, 0]
    for i, c in enumerate(PALETTE, start=1):
        ax.axvspan(i + 0.5, i + 1.5, color=c, lw=1)
    ax.set_xlim(1.5, len(PALETTE) + 1.5)
    ax.set_yticks([])
    _finish(fig, path)


# -----------------------------------------------------------------------------
# ------------------------------- DATA CLEANING --------------------------------
# -----------------------------------------------------------------------------

def plot_cleaning_steps(clean, sse_event=103, mse_event=96, save_dir=None):
    """
    One figure per cleaning step, same two events in all of them.
    sse_event / mse_event are indices in the ORIGINAL (not aligned) arrays.
    The aligned arrays have fewer rows (noise / edge events are dropped), so the same event
    is found there through good_idx (and the aligned plots are skipped if it was dropped).
    """
    steps = clean['steps']
    events = [(sse_event, f'{SSE_LABEL} event', SSE_COLOR), (mse_event, f'{MSE_LABEL} event', MSE_COLOR)]
    x = np.arange(steps['raw'].shape[1]) * DT_NS

    specs = [
        ('raw', f'Charge raw waveforms - {detector.BOLD_NAME}', 'Charge [ADC counts]', 'raw_wf.pdf'),
        ('baseline_subtracted', f'Charge waveforms after baseline subtraction - {detector.BOLD_NAME}', 'Shifted charge [ADC counts]', 'baseline_subtracted_wf.pdf'),
        ('normalised', f'Charge waveforms after normalisation - {detector.BOLD_NAME}', 'Normalised charge [a.u.]', 'normalised_wf.pdf'),
        ('current', f'Current waveforms - {detector.BOLD_NAME}', 'Current [a.u./ns]', 'current_wf.pdf'),
        ('filtered', f"Current waveforms after {({'gauss': 'Gaussian', 'butter': 'Butterworth'}).get(clean.get('filter_type'), 'low-pass')} filtering - {detector.BOLD_NAME}", 'Current [a.u./ns]', 'current_filtered_wf.pdf'),
    ]
    for key, title, ylabel, fname in specs:
        fig, axes = make_figure(1)
        ax = axes[0, 0]
        for i, label, color in events:
            ax.plot(x, steps[key][i], lw=1.5, label=label, color=color)
        _style(ax, title, 'Time [ns]', ylabel, legend_loc='upper left')
        _finish(fig, os.path.join(save_dir, fname) if save_dir else None)

    # ---- aligned windows: original index -> row of the aligned arrays ----
    positions = []
    for i, label, color in events:
        pos = np.where(clean['good_idx'] == i)[0]
        if len(pos) == 0:
            print(f"Event {i} was dropped by the alignment (noise or edge): aligned plots skipped for it.")
            continue
        positions.append((pos[0], label, color))

    x200 = np.arange(clean['current'].shape[1]) * DT_NS
    aligned_specs = [
        ('current', f'Current waveforms after peak alignment and trimming - {detector.BOLD_NAME}', 'current_aligned_filtered_wf.pdf'),
        ('current_normalised', f'Aligned and trimmed current waveforms after normalisation - {detector.BOLD_NAME}', 'current_aligned_filtered_normalised_wf.pdf'),
    ]
    for key, title, fname in aligned_specs:
        if not positions:
            break
        fig, axes = make_figure(1)
        ax = axes[0, 0]
        for pos, label, color in positions:
            ax.plot(x200, clean[key][pos], lw=1.5, label=label, color=color)
        _style(ax, title, 'Relative Time [ns]', 'Current [a.u./ns]', legend_loc='upper left')
        _finish(fig, os.path.join(save_dir, fname) if save_dir else None)


# -----------------------------------------------------------------------------
# --------------------------------- SPECTRUM -----------------------------------
# -----------------------------------------------------------------------------

def _energy_bins(energy, xlim=None):
    """1 keV bins; with xlim the edges are exactly xlim[0], ..., xlim[1] (no empty border)."""
    if xlim is not None:
        return np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 1, 1)
    return np.arange(np.floor(energy.min()), np.ceil(energy.max()) + 1, 1)


def _fit_curve(peak, n=200):
    x = np.linspace(peak['window'][0] - 1.5, peak['window'][1] + 1.5, n)
    return x, gaussian_model(x, *peak['params'])


def plot_peak_fits(energy, fit, path=None, xlim=(1550, 1650)):
    bins = _energy_bins(energy)
    fig, axes = make_figure(1)
    ax = axes[0, 0]
    ax.hist(energy, bins=bins, histtype='step', color='black', alpha=0.8)
    ax.hist(energy, bins=bins, histtype='stepfilled', color=color1, alpha=SPECTRUM_FILL_ALPHA)
    for name, label, color in (('sse', SSE_LABEL, SSE_COLOR), ('mse', MSE_LABEL, MSE_COLOR)):
        ax.plot(*_fit_curve(fit[name]), color=color, lw=2, label=f'Gaussian fit on the {label} peak')
    ax.set_xlim(*xlim)
    ax.set_yscale('log')
    _style(ax, f'DEP and FEP fits - {detector.BOLD_NAME}', 'Energy [keV]', 'Counts / keV')
    _finish(fig, path)


def plot_sigma_selection(energy, fit, sigmas=(3.0, 1.0), path=None, xlim=(1550, 1650)):
    """Shaded SSE / MSE windows for several ±k sigma cuts (darker = narrower)."""
    bins = _energy_bins(energy)
    counts, edges = np.histogram(energy, bins=bins)
    sigmas = sorted(sigmas, reverse=True)

    # Mappa le trasparenze (alpha) in base al valore del sigma
    alpha_map = {3.0: 0.2, 1.0: 0.5}

    fig, axes = make_figure(1)
    ax = axes[0, 0]

    ax.hist(energy, bins=bins, histtype='step', color='black', alpha=0.8)
    ax.hist(energy, bins=bins, histtype='stepfilled', color=color1, alpha=SPECTRUM_FILL_ALPHA)
            
    for name, label, color in (('sse', SSE_LABEL, SSE_COLOR), ('mse', MSE_LABEL, MSE_COLOR)):
        ax.plot(*_fit_curve(fit[name]), color=color, lw=2, label=f'Gaussian fit on the {label} peak')

    # Corretto l'unpacking di enumerate: i è l'indice, k è il valore del sigma
    for i, k in enumerate(sigmas):
        # Prende l'alpha corretto dal dizionario, con fallback a 0.5 nel caso di sigma diversi
        alpha_val = alpha_map.get(k, 0.5) 
        
        for name, label, color in (('sse', SSE_LABEL, SSE_COLOR), 
                                   ('mse', MSE_LABEL, MSE_COLOR)):
            p = fit[name]
            m = (edges[:-1] >= p['mu'] - k * p['sigma']) & (edges[:-1] <= p['mu'] + k * p['sigma'])
            
            # Etichetta formattata dinamicamente come richiesto (es. "DEP 208Tl 3sigma cut")
            cut_label = fr'{label} {k:g}$\sigma$ window'
            
            ax.fill_between(edges[:-1][m], counts[m], step='post', color=color, alpha=alpha_val,
                            label=cut_label)
                            
    ax.set_xlim(*xlim)
    ax.set_yscale('log')
    _style(ax, rf'SSE and MSE selection with different $\sigma$ windows - {detector.BOLD_NAME}', 'Energy [keV]', 'Counts / keV')
    _finish(fig, path)


def plot_spectrum_cuts(energy_all, energy_aoe, energy_clf, fit, sigma, sf_aoe, sf_clf, threshold,
                       path=None, xlim=(1550, 1650), ymax=30000, min_counts=20, extra_cuts=None):
    """
    TOP: spectrum before any cut (grey filled + black outline), after the A/E low-side cut
    (color2, filled) and after the VAE-classifier cut (color3 line, p < threshold kept).
    The ±sigma DEP / FEP windows are marked by dashed lines (color4 / color5): in the top panel
    they stop at the full spectrum, in the bottom panel they span the whole height.
    Drawing order (zorder): grid < grey fill < A/E fill < black outline < A/E line < VAE < window lines.
    BOTTOM: fraction of events kept in every bin by each cut (bins with fewer than
    min_counts events before the cuts are not shown).
    extra_cuts (optional): more VAE cuts to draw on top, e.g. the classifier trained on other
    windows, as a list of dicts {'energy', 'sf', 'threshold', 'color', 'trained_on'} where
    'trained_on' is the sigma of the training windows. When given, every VAE label says which
    windows its classifier was trained on; all the survival fractions refer to THIS plot's windows.
    """
    extra_cuts = extra_cuts or []
    bins = _energy_bins(energy_all, xlim)
    counts_all, edges = np.histogram(energy_all, bins=bins)
    counts_aoe, _ = np.histogram(energy_aoe, bins=bins)
    counts_clf, _ = np.histogram(energy_clf, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    n_all, n_aoe, n_clf = len(energy_all), len(energy_aoe), len(energy_clf)

    # filled areas need a positive baseline on a log axis
    extra_counts = [np.histogram(cut['energy'], bins=bins)[0] for cut in extra_cuts]
    positive = np.concatenate([c[c > 0] for c in (counts_all, counts_aoe, counts_clf, *extra_counts)])
    floor = 0.5 * positive.min() if len(positive) else 0.5

    # one column -> 9 wide; top panel 6 high, bottom panel 4 high (3:2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    handles = []

    # ================= TOP: spectra =================
    # full spectrum: filled grey + black outline
    ax1.stairs(counts_all, edges, baseline=floor, fill=True, color=color1, alpha=SPECTRUM_FILL_ALPHA, lw=0, zorder=1)
    ax1.stairs(counts_all, edges, baseline=None, color='black', alpha=0.8, lw=1, zorder=3)
    handles.append(Patch(facecolor=to_rgba(color1, SPECTRUM_FILL_ALPHA), edgecolor=to_rgba('black', 0.8), linewidth=1,
                         label=f"all events\n({n_all} events)"))

    # A/E cut: color2, filled + outline
    ax1.stairs(counts_aoe, edges, baseline=floor, fill=True, color=AOE_COLOR, alpha=0.5, lw=0, zorder=2.5)
    ax1.stairs(counts_aoe, edges, baseline=None, color=AOE_COLOR, lw=1.2, zorder=4)
    handles.append(Patch(facecolor=to_rgba(AOE_COLOR, 0.5), edgecolor=AOE_COLOR, linewidth=1,
                         label=f"A/E cut\n({n_aoe} events, {100 * n_aoe / n_all:.1f} %; "
                               f"DEP SF {sf_aoe['sse']:.1f} %, FEP SF {sf_aoe['mse']:.1f} %)"))

    # VAE cut: color3 line (+ the extra VAE cuts, each in its own colour)
    trained = rf" (trained on ±{sigma:g}$\sigma$)" if extra_cuts else ''
    handles.append(ax1.stairs(counts_clf, edges, baseline=None, color=CLF_COLOR, lw=1.5, zorder=5,
                              label=f"VAE classifier cut{trained}, p < {threshold:.3g}\n({n_clf} events, {100 * n_clf / n_all:.1f} %; "
                                    f"DEP SF {sf_clf['sse']:.1f} %, FEP SF {sf_clf['mse']:.1f} %)"))
    for cut, counts in zip(extra_cuts, extra_counts):
        n = len(cut['energy'])
        handles.append(ax1.stairs(counts, edges, baseline=None, color=cut['color'], lw=1.5, zorder=5,
                                  label=rf"VAE classifier cut (trained on ±{cut['trained_on']:g}$\sigma$), p < {cut['threshold']:.3g}"
                                        f"\n({n} events, {100 * n / n_all:.1f} %; "
                                        f"DEP SF {cut['sf']['sse']:.1f} %, FEP SF {cut['sf']['mse']:.1f} %)"))

    # ±sigma windows: dashed lines at mu ± sigma*fit_sigma, on both panels
    window_lines = []
    for name, color, label in (('sse', SSE_COLOR, r"DEP of ${}^{208}$Tl"), ('mse', MSE_COLOR, r"FEP of ${}^{212}$Bi")):
        p = fit[name]
        for edge in (p['mu'] - sigma * p['sigma'], p['mu'] + sigma * p['sigma']):
            # top: the line stops at the full spectrum (it never crosses the legend); bottom: full height
            height = counts_all[np.clip(np.searchsorted(edges, edge, side='right') - 1, 0, len(counts_all) - 1)]
            ax1.vlines(edge, floor, height, color=color, ls='--', lw=1.5, zorder=6)
            ax2.axvline(edge, color=color, ls='--', lw=1.5, zorder=6)
        window_lines.append(Line2D([], [], color=color, ls='--', lw=1.5,
                                   label=rf"{label}" + "\n" + rf"(±{sigma:g}$\sigma$ window)"))
    handles += window_lines

    ax1.set_yscale('log')
    ax1.set_ylim(floor, ymax)   # room for the legend above the peaks
    _style(ax1, fr'Energy spectrum of calibration data of $\mathit{{prl25}}$ - A/E cut vs VAE cut{_window(sigma)}',
           None, f'Counts / {bin_width:g} keV', legend=False)
    ax1.legend(handles=handles, frameon=False, loc='upper left', fontsize=9)

    # ================= BOTTOM: fraction of events kept =================
    enough = counts_all >= max(min_counts, 1)
    bottom = [(counts_aoe, AOE_COLOR, 'A/E cut'),
              (counts_clf, CLF_COLOR, 'VAE classifier cut' + (rf' (±{sigma:g}$\sigma$)' if extra_cuts else ''))]
    bottom += [(counts, cut['color'], rf"VAE classifier cut (±{cut['trained_on']:g}$\sigma$)")
               for cut, counts in zip(extra_cuts, extra_counts)]
    for counts, color, label in bottom:
        fraction = np.where(enough, counts / np.maximum(counts_all, 1), np.nan)
        ax2.plot(centres, fraction, 'o-', ms=3, lw=1, color=color, label=label)
    ax2.set_ylim(0, 1)
    _style(ax2, None, 'Energy [keV]', 'Fraction of events kept', legend=False)
    ax2.legend(frameon=False, loc='lower left', fontsize=9, ncol=len(bottom))

    ax1.set_xlim(edges[0], edges[-1])
    _finish(fig, path)


def plot_waveforms_agreement(waves, aoe_pass, vae_pass, path_disagree=None, path_agree=None,
                             sigma=None, n_show=1000, seed=42, ylabel='Current [a.u.]'):
    """
    Event-by-event comparison of the A/E cut and of the VAE-classifier cut, two figures:
        disagree: kept by A/E & rejected by VAE (color2) | rejected by A/E & kept by VAE (color3)
        agree:    rejected by both (color3)              | kept by both (color3)
    waves: (n_events, 200) current fed to the VAE (same scaling as the configuration);
    aoe_pass / vae_pass: boolean masks on the same events. At most n_show random waveforms per
    panel are drawn; the y axis is the same in all four panels.
    """
    waves = np.asarray(waves)
    n = len(waves)
    x_ns = np.arange(waves.shape[1]) * DT_NS
    rng = np.random.default_rng(seed)

    figures = [
        (path_disagree, 'A/E cut vs VAE cut comparison',
         [(aoe_pass & ~vae_pass, 'kept by A/E, rejected by VAE', AOE_COLOR),
          (~aoe_pass & vae_pass, 'rejected by A/E, kept by VAE', CLF_COLOR)]),
        (path_agree, 'events where A/E cut and VAE cut agree',
         [(~aoe_pass & ~vae_pass, 'rejected by both A/E and VAE', CLF_COLOR),
          (aoe_pass & vae_pass, 'kept by both A/E and VAE', CLF_COLOR)]),
    ]

    # random subsample of every group, chosen once, and a common y range
    shown = {}
    for _, _, groups in figures:
        for mask, label, _ in groups:
            idx = np.where(mask)[0]
            shown[label] = rng.choice(idx, min(n_show, len(idx)), replace=False) if len(idx) else idx
    all_shown = np.concatenate([waves[i] for i in shown.values() if len(i)]) if any(len(i) for i in shown.values()) else np.zeros((1, 1))
    ymin, ymax = np.nanmin(all_shown), np.nanmax(all_shown)
    margin = 0.05 * (ymax - ymin or 1)

    suffix = _window(sigma)
    for path, title, groups in figures:
        fig, axes = make_figure(2, 1, sharey=True)
        for ax, (mask, label, color) in zip(axes[0], groups):
            idx = shown[label]
            if len(idx):
                segments = np.stack([np.broadcast_to(x_ns, (len(idx), len(x_ns))), waves[idx]], axis=-1)
                ax.add_collection(LineCollection(segments, colors=[to_rgba(color, 0.1)], linewidths=1))
            else:
                ax.text(0.5, 0.5, 'no events', transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(x_ns[0], x_ns[-1])
            ax.set_ylim(ymin - margin, ymax + margin)
            _style(ax, f'{label}\n({mask.sum()} events, {100 * mask.sum() / n:.1f} %, {len(idx)} shown)',
                   'Time [ns]', ylabel if ax is axes[0, 0] else None, legend=False)
        top = _top_title(fig, rf'Processed current waveforms of calibration data of $\mathit{{prl25}}$ - {title}{suffix}')
        _finish(fig, path, top=top)


# -----------------------------------------------------------------------------
# ------------------------------------ VAE -------------------------------------
# -----------------------------------------------------------------------------

def vae_reconstruction_grid(original, reconstructions, titles):
    '''
    One column per reconstruction, one row per channel (current, then charge if present).
    Panel size follows the size rule on the number of COLUMNS.
    '''
    original = np.atleast_2d(original)
    n_channels = original.shape[0]
    n_cols = len(reconstructions)
    x_ns = np.arange(original.shape[-1]) * DT_NS

    fig, axes = make_figure(n_cols, n_channels, sharex=True, sharey='row')

    for col, (rec, title) in enumerate(zip(reconstructions, titles)):
        rec = np.atleast_2d(rec)
        for c in range(n_channels):
            ax = axes[c, col]
            ax.plot(x_ns, original[c], color=VAE_ORIGINAL_COLOR, lw=1.5, label='Original')
            ax.plot(x_ns, rec[c], color=VAE_RECONSTRUCTED_COLOR, lw=1.5, label='Reconstructed')
            ax.tick_params(labelbottom=True)
            _style(ax, f'{title} - {CHANNEL_NAMES[c]}', 'Time [ns]', Y_LABELS[c] if col == 0 else None, legend=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    return fig, axes, handles, labels


def plot_training_reconstruction(snapshots, beta, path=None, sigma=None):
    """Fixed validation waveform at the first, middle and last epoch (deterministic reconstruction)."""
    n = len(snapshots['reconstructions'])
    epochs_to_plot = sorted({0, max(n // 2 - 1, 0), n - 1})
    fig, axes, handles, labels = vae_reconstruction_grid(
        snapshots['original'],
        [snapshots['reconstructions'][e] for e in epochs_to_plot],
        [f'Epoch #{e + 1}' for e in epochs_to_plot])
    window = _window(sigma)
    top = _top_title_and_legend(fig, f'Waveform reconstruction through epochs - validation set{window}', handles, labels)
    _finish(fig, path, top=top)


def plot_test_reconstruction(example, beta, path=None, sigma=None):
    """
    Test-set waveform and its reconstruction. With current + charge the two channels are
    placed SIDE BY SIDE (one row, two columns -> 9x6 per panel), not stacked.
    """
    original = np.atleast_2d(example['original'])
    reconstruction = np.atleast_2d(example['reconstruction'])
    n_channels = original.shape[0]
    x_ns = np.arange(original.shape[-1]) * DT_NS

    fig, axes = make_figure(n_channels, 1, sharex=True)
    for c in range(n_channels):
        ax = axes[0, c]
        ax.plot(x_ns, original[c], color=VAE_ORIGINAL_COLOR, lw=1.5, label='Original')
        ax.plot(x_ns, reconstruction[c], color=VAE_RECONSTRUCTED_COLOR, lw=1.5, label='Reconstructed')
        _style(ax, f'Test set - {CHANNEL_NAMES[c]}', 'Time [ns]', Y_LABELS[c], legend=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    window = _window(sigma)
    top = _top_title_and_legend(fig, f'Original vs Reconstructed waveform of the test set{window}', handles, labels)
    _finish(fig, path, top=top)

def plot_losses(history, warmup, beta, path=None, sigma=None):
    """Total loss, NLL and KL (train vs validation), with a zoom after the warm-up."""
    fig, axes = make_figure(3)
    axes = axes[0]
    start = warmup if len(history['train_loss']) > warmup + 1 else 0

    curves = [('train_loss', 'val_loss', r'Total loss = NLL + β $\cdot$ KL', True),
              ('train_rec', 'val_rec', 'Reconstruction (NLL)', True),
              ('train_kl', 'val_kl', 'KL [nat]', False)]

    for ax, (k_train, k_val, name, zoom) in zip(axes, curves):
        train, val = history[k_train], history[k_val]
        line_train, = ax.plot(train, color=TRAIN_COLOR)
        line_val, = ax.plot(val, color=VAL_COLOR)
        _style(ax, name, 'epoch', legend=False)
        if zoom and start > 0:
            axins = inset_axes(ax, width="40%", height="40%", loc='center right')
            axins.plot(train, color=TRAIN_COLOR)
            axins.plot(val, color=VAL_COLOR)
            axins.set_xlim(start, len(train) - 1)
            ymin = min(min(train[start:]), min(val[start:]))
            ymax = max(max(train[start:]), max(val[start:]))
            margin = (ymax - ymin) * 0.1 or 1e-6
            axins.set_ylim(ymin - margin, ymax + margin)
            axins.set_axisbelow(True)
            axins.grid(alpha=0.2, ls='--')
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls=':')

    window = _window(sigma)
    top = _top_title_and_legend(fig, f'Losses analysis{window}',
                                [line_train, line_val], ['training set', 'validation set'])
    _finish(fig, path, top=top)


# -----------------------------------------------------------------------------
# --------------------------------- CLASSIFIER ---------------------------------
# -----------------------------------------------------------------------------

def plot_classifier_training(history, path=None, sigma=None):
    """Loss and accuracy of the classifier; the dotted line marks the restored (best) epoch."""
    fig, axes = make_figure(2)
    axes = axes[0]
    best = history.get('best_epoch')

    for ax, (k_train, k_val, title, ylabel) in zip(axes, [('train_loss', 'val_loss', 'Classifier loss (BCE)', 'loss'),
                                                          ('train_acc', 'val_acc', 'Classifier accuracy', 'accuracy [%]')]):
        line_train, = ax.plot(history[k_train], color=TRAIN_COLOR, lw=1.5)
        line_val, = ax.plot(history[k_val], color=VAL_COLOR, lw=1.5)
        if best is not None:
            ax.axvline(best, color='black', ls=':', lw=1)
        _style(ax, title, 'epoch', ylabel, legend=False)

    handles, labels = [line_train, line_val], ['training set', 'validation set']
    if best is not None:
        handles.append(plt.Line2D([], [], color='black', ls=':', lw=1))
        labels.append(f'restored epoch ({best + 1})')
    window = _window(sigma)
    top = _top_title_and_legend(fig, f'Classifier training on the VAE latent space{window}', handles, labels)
    _finish(fig, path, top=top)


def plot_classifier_output(p, y, title, path=None, threshold=0.5):
    """Distribution of the predicted probability of being MSE, for true SSE and true MSE."""
    bins = np.linspace(0, 1, 101)
    fig, axes = make_figure(1)
    ax = axes[0, 0]
    ax.hist(p[y == 0], bins=bins, histtype='step', lw=1.5, color=SSE_COLOR, label='true SSE')
    ax.hist(p[y == 1], bins=bins, histtype='step', lw=1.5, color=MSE_COLOR, label='true MSE')
    ax.axvline(threshold, color='black', ls='--', lw=1, label=f'threshold = {threshold:.3f}')
    _style(ax, title, 'Predicted probability of being MSE', 'Counts')
    _finish(fig, path)


def plot_roc(y, scores, auc_results, title='ROC curves', path=None, operating_points=None):
    """
    ROC curves of several discriminators on the SAME events.
    scores:           {name: score}, higher score = more MSE-like (pass -A/E for A/E)
    auc_results:      output of preliminary.bootstrap_auc (AUC ± error in the legend)
    operating_points: {name: (fpr, tpr, label)} -> a star on the curve of `name` where the actual
                      cut sits (see preliminary.operating_point), with its own legend entry
    A/E is always drawn and listed first (blue), then the VAE classifier (yellow).
    """
    operating_points = operating_points or {}
    fig, axes = make_figure(1)
    ax = axes[0, 0]

    names = sorted(scores, key=lambda n: n != 'A/E')
    for i, name in enumerate(names):
        color = AOE_COLOR if name == 'A/E' else (CLF_COLOR if i == 1 else PALETTE[i % len(PALETTE)])
        fpr, tpr, _ = roc_curve(y, scores[name])
        r = auc_results[name]
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{name}: AUC = {r['auc']:.3f} ± {r['err']:.3f}")
        if name in operating_points:
            x0, y0, label = operating_points[name]
            ax.plot(x0, y0, marker='*', ms=16, color=color, markeredgecolor='black', ls='none', zorder=5,
                    label=label)

    ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=1, label='random')
    _style(ax, title, 'False positive rate (SSE classified as MSE)', 'True positive rate (MSE classified as MSE)',
           legend_loc='lower right')
    _finish(fig, path)


def plot_auc_summary(summary, path=None):
    """
    Final comparison: AUC on the test set of A/E and of the VAE classifier for every
    configuration, with bootstrap errors (A/E first in the legend).
    """
    labels = [f"±{r.sigma:g}σ"
              for r in summary.itertuples()]
    x = np.arange(len(summary))

    fig, axes = make_figure(1)
    ax = axes[0, 0]
    series = [('auc_aoe', 'auc_aoe_err', 'A/E (test set)', AOE_COLOR, -0.1, 's'),
              ('auc_clf', 'auc_clf_err', 'VAE classifier (test set)', CLF_COLOR, 0.1, 'o')]
    for col, err, label, color, dx, marker in series:
        ax.errorbar(x + dx, summary[col], yerr=summary[err], fmt=marker, color=color,
                    capsize=3, lw=1.2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    _style(ax, 'AUC comparison - all configurations', None, 'AUC')
    _finish(fig, path)

# -----------------------------------------------------------------------------
# -------------------------------- LATENT SPACE --------------------------------
# -----------------------------------------------------------------------------

LATENT_PANEL_SIZE = (6, 3.5)   # wider than tall: the 2 x 4 grid of the latent distributions stays low


def _robust_range(x, q=(0.005, 0.995)):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanquantile(x, q[0]), np.nanquantile(x, q[1])
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def plot_latent_distributions(mu, labels, rho_aoe=None, rho_risetime=None, path=None, sigma=None, set_name=None):
    """
    Distribution (counts) of every latent mean mu_j for the true SSE (DEP window) and true MSE
    (FEP window), 2 rows x 4 columns of panels wider than tall.
    rho_aoe / rho_risetime (optional): Spearman correlation of every dimension with A/E and with the
    rise time (preliminary.latent_correlations), written inside each panel as rho_A/E and rho_rt.
    """
    labels = np.asarray(labels)
    n_dims = mu.shape[1]
    n_cols = int(np.ceil(n_dims / 2))
    w, h = LATENT_PANEL_SIZE
    fig, axes = plt.subplots(2, n_cols, figsize=(w * n_cols, h * 2), squeeze=False)
    axes = axes.ravel()

    for j, ax in enumerate(axes[:n_dims]):
        bins = np.linspace(*_robust_range(mu[:, j]), 81)
        ax.hist(mu[labels == 0, j], bins=bins, histtype='step', lw=1.5, color=SSE_COLOR, label='true SSE (DEP)')
        ax.hist(mu[labels == 1, j], bins=bins, histtype='step', lw=1.5, color=MSE_COLOR, label='true MSE (FEP)')
        _style(ax, None, rf'$\mu_{{{j + 1}}}$', 'Counts' if j % n_cols == 0 else None, legend=False)
        if rho_aoe is not None and rho_risetime is not None:
            # correlations written inside the panel as a small legend without symbols
            empty = [Line2D([], [], ls='none')] * 2
            ax.legend(empty, [rf'$\rho_{{A/E}}$ = {rho_aoe[j]:+.2f}', rf'$\rho_{{rt}}$ = {rho_risetime[j]:+.2f}'],
                      handlelength=0, handletextpad=0, frameon=False, loc='best', fontsize=10)
    for ax in axes[n_dims:]:
        ax.set_visible(False)

    handles, names = axes[0].get_legend_handles_labels()
    window = _window(sigma)
    which = f' - {set_name}' if set_name else ''
    top = _top_title_and_legend(fig, f'Distributions of the VAE latent means{which}{window}', handles, names)
    _finish(fig, path, top=top)
