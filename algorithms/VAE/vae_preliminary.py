"""
Preliminary steps of the VAE analysis (LEGEND-200, 228Th calibration).

Everything that is not a loop over configurations and not a plot lives here:

    load_data                     -> waveforms / energies / A/E from the checkpoints
    preprocess                    -> cleaning pipeline (called step by step) on both sets
    fit_peaks, purity_table, ...  -> DEP / FEP fits, purity, A/E survival fractions
    prepare_data                  -> SSE/MSE masks, train/val/test split, current scaling
    bootstrap_auc, linear_probe   -> metrics used in the final comparison
    config_dir, run_tag           -> folder / name of every configuration

Current modes
-------------
    'normalised' -> aligned current divided by its own peak (every event has peak = 1):
                    no amplitude information at all, only the time shape
    'scaled'     -> aligned current divided by ONE global constant (median peak of the
                    training set): the relative amplitude, i.e. the A/E information, survives
"""

import os
import random
import importlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import detector                # common/: detector name on every plot, folders
import filereading             # common/: reading of the checkpoints
import l200_utils as ut        # common/: waveform cleaning, loaders, plots



CURRENT_MODES_ALLOWED = ('normalised', 'scaled')


# -----------------------------------------------------------------------------
# ------------------------------- REPRODUCIBILITY ------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# --------------------------------- DATA LOADING -------------------------------
# -----------------------------------------------------------------------------

def load_data(periods, chosen_detector, e_min, e_max, output_dir, verbose=True):
    """
    Loads the checkpoints of all the periods (filereading of the DTC project: also the rise time,
    tp_90 - tp_10 from the psp tier, aligned event by event with energy and A/E).
    'charge_AoE' / 'energy_AoE' are the events that ALSO pass the A/E low-side cut.
    """
    charge, energy, aoe, risetime, charge_aoe, energy_aoe, risetime_aoe = filereading.partition_from_checkpoint(
        periods, chosen_detector=chosen_detector, e_min=e_min, e_max=e_max,
        output_dir=output_dir, verbose=verbose
    )

    print("Waveforms shape (all periods):", charge.shape)
    print("Energy shape (all periods):", energy.shape)
    print("A/E shape (all periods):", aoe.shape)
    print("Rise time shape (all periods):", risetime.shape)
    print("Waveforms shape after A/E cut (all periods):", charge_aoe.shape)
    print("Energy shape after A/E cut (all periods):", energy_aoe.shape)

    return dict(charge=charge, energy=energy, aoe=aoe, risetime=risetime,
                charge_AoE=charge_aoe, energy_AoE=energy_aoe, risetime_AoE=risetime_aoe)


# -----------------------------------------------------------------------------
# -------------------------------- DATA CLEANING -------------------------------
# -----------------------------------------------------------------------------

FILTERS_ALLOWED = ('gauss', 'butter')        # same names as l200_utils.FILTER_TYPE
FILTER_NAMES = {'gauss': 'Gaussian', 'butter': 'Butterworth'}


def filter_current(current, filter_type=ut.FILTER_TYPE, gauss_sigma=ut.GAUSS_SIGMA, butter_order=4):
    """
    Low-pass filter of the current, with the SAME functions of common/l200_utils.py:
      'gauss'  -> ut.gaussFilter, sigma in SAMPLES (1 sample = 16 ns). The Gaussian kernel is
                  positive everywhere: no ringing, no dip before/after sharp current peaks.
                  sigma = 1.25 has the same -6 dB frequency of the old Butterworth (0.15 cycles/sample).
      'butter' -> ut.butterFilter (old processing). Zero-phase (sosfiltfilt): its impulse
                  response has negative lobes on BOTH sides -> small dip before fast current peaks.
    """
    if filter_type == 'gauss':
        return ut.gaussFilter(current, sigma=gauss_sigma)
    if filter_type == 'butter':
        return ut.butterFilter(current, order=butter_order)
    raise ValueError(f"filter_type must be one of {FILTERS_ALLOWED}, got {filter_type!r}")


def clean_waveforms(charge, baseline_points=300, n_sigmas=3, pre_trigger_points=75,
                    post_trigger_points=124, filter_type=ut.FILTER_TYPE, gauss_sigma=ut.GAUSS_SIGMA,
                    butter_order=4, alignment='first', prominence_fraction=ut.PROMINENCE_FRACTION,
                    keep_steps=False, name=''):
    """
    Cleaning pipeline, every cleaning step of l200_utils called explicitly:
        1. baseline subtraction
        2. per-event normalisation of the charge (peak = 1)
        3. current = gradient of the normalised charge
        4. low-pass filter on the current (Gaussian or Butterworth, see filter_current)
        5. alignment on the current peak + 200-sample window (noise / edge events dropped):
           alignment='first'   -> FIRST peak with prominence >= prominence_fraction of the highest
                                  (as in the DTC project: the trigger fires on the first deposition)
           alignment='highest' -> highest peak (old VAE behaviour)
        6. second normalisation of the aligned current (peak = 1) -> 'normalised' mode

    keep_steps=True also returns the intermediate full-length arrays (needed only for the plots).
    """
    baseline_subtracted = ut.baseline(charge, baseline_points=baseline_points)
    normalised = ut.normalisation(baseline_subtracted)
    current = ut.current(normalised, dt=1.0)
    filtered = filter_current(current, filter_type=filter_type, gauss_sigma=gauss_sigma, butter_order=butter_order)

    charge_aligned, current_aligned, good_idx, noise_idx, edge_idx = ut.peaksAlignment(
        normalised, filtered,
        baseline_points=baseline_points, n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points, post_trigger_points=post_trigger_points,
        keep_noise=False,   # no NaN rows for the training
        alignment=alignment, prominence_fraction=prominence_fraction,
    )
    current_aligned_normalised = ut.normalisation(current_aligned)

    print(f"{name} Kept: {len(good_idx)} | noise: {len(noise_idx)} | edge: {len(edge_idx)} | total: {len(charge)}")

    out = dict(charge_aligned=charge_aligned, current_aligned=current_aligned,
               current_aligned_normalised=current_aligned_normalised,
               good_idx=good_idx, noise_idx=noise_idx, edge_idx=edge_idx)

    if keep_steps:
        out['steps'] = dict(raw=charge, baseline_subtracted=baseline_subtracted,
                            normalised=normalised, current=current, filtered=filtered)
    return out


def aoe_pass_mask(energy, energy_aoe):
    """
    Event-by-event A/E decision for the events of `energy`.
    filereading builds the A/E set as mask & AoE_Low_Cut on the SAME events, in the SAME order,
    so energy_aoe is an ordered subsequence of energy: walking both arrays in parallel finds,
    for every event, whether it passed the A/E cut.
    """
    energy, energy_aoe = np.asarray(energy), np.asarray(energy_aoe)
    passed = np.zeros(len(energy), dtype=bool)
    j = 0
    for i in range(len(energy)):
        if j < len(energy_aoe) and energy[i] == energy_aoe[j]:
            passed[i] = True
            j += 1
    if j != len(energy_aoe):
        print(f"WARNING: only {j} of {len(energy_aoe)} A/E events matched: the event-by-event A/E mask is not reliable")
    return passed


def preprocess(data, baseline_points=300, n_sigmas=3, pre_trigger_points=75, post_trigger_points=124,
               filter_type=ut.FILTER_TYPE, gauss_sigma=ut.GAUSS_SIGMA, butter_order=4,
               alignment='first', prominence_fraction=ut.PROMINENCE_FRACTION, keep_steps=True):
    """
    Cleans both sets (all events, events after the A/E cut) with the same parameters and
    realigns energies and A/E to the surviving events.

    Returned dict (all arrays refer to the events that survive the alignment):
        charge              aligned normalised charge                (n, 200)
        current             aligned filtered current (NOT renormalised) -> 'scaled' mode
        current_normalised  aligned current with peak = 1             -> 'normalised' mode
        energy, aoe         energies and A/E classifier values
        risetime            official rise time tp_90 - tp_10 [ns] (NaN where not computable)
        energy_AoE          energies of the events that pass the A/E low-side cut
        aoe_pass            boolean, event-by-event A/E decision for the cleaned events
        good_idx            positions in the ORIGINAL arrays of the surviving events
        steps               intermediate arrays (original indexing), only if keep_steps
    """
    kwargs = dict(baseline_points=baseline_points, n_sigmas=n_sigmas, pre_trigger_points=pre_trigger_points,
                  post_trigger_points=post_trigger_points, filter_type=filter_type, gauss_sigma=gauss_sigma,
                  butter_order=butter_order, alignment=alignment, prominence_fraction=prominence_fraction)

    full = clean_waveforms(data['charge'], keep_steps=keep_steps, name='[all events]', **kwargs)
    after_aoe = clean_waveforms(data['charge_AoE'], keep_steps=False, name='[after A/E cut]', **kwargs)

    clean = dict(
        charge=full['charge_aligned'],
        current=full['current_aligned'],
        current_normalised=full['current_aligned_normalised'],
        energy=data['energy'][full['good_idx']],
        aoe=data['aoe'][full['good_idx']],
        risetime=data['risetime'][full['good_idx']],
        energy_AoE=data['energy_AoE'][after_aoe['good_idx']],
        # event-by-event A/E decision for the cleaned events (True = passes the A/E low-side cut)
        aoe_pass=aoe_pass_mask(data['energy'], data['energy_AoE'])[full['good_idx']],
        good_idx=full['good_idx'], noise_idx=full['noise_idx'], edge_idx=full['edge_idx'],
        pre_trigger_points=pre_trigger_points,
        filter_type=filter_type, gauss_sigma=gauss_sigma, alignment=alignment,
    )
    if keep_steps:
        clean['steps'] = full['steps']

    print(f"Filter: {FILTER_NAMES[filter_type]}"
          + (f" (sigma = {gauss_sigma:g} samples = {16 * gauss_sigma:g} ns)" if filter_type == 'gauss' else f" (order {butter_order})")
          + f" | alignment: {alignment} peak" + (f" (prominence >= {prominence_fraction:g} of the highest)" if alignment == 'first' else ''))
    print("Charge aligned:", clean['charge'].shape, "| Current aligned:", clean['current'].shape)
    print(f"Events passing A/E (event by event): {clean['aoe_pass'].sum()} "
          f"| events in the cleaned A/E set: {len(clean['energy_AoE'])} (the two numbers must coincide)")
    return clean


# -----------------------------------------------------------------------------
# ---------------------------- DEP / FEP SELECTION -----------------------------
# -----------------------------------------------------------------------------

def fit_peaks(energy, sse_window=(1587, 1598), mse_window=(1615, 1626)):
    """
    Gaussian + constant background fit on the DEP (SSE) and on the FEP (MSE), done ONCE:
    every mask in the analysis uses these parameters.
    amp is the AREA of the peak (counts), not its height.
    """
    fit = {}
    for name, window in (('sse', sse_window), ('mse', mse_window)):
        params, errors = ut.gaussian(energy, xmin=window[0], xmax=window[1])
        fit[name] = dict(amp=params[0], mu=params[1], sigma=params[2], bkg=params[3],
                         amp_err=errors[0], mu_err=errors[1], sigma_err=errors[2], bkg_err=errors[3],
                         params=params, window=window)
    return fit


def print_fit(fit):
    for name, label in (('sse', 'SSE - 208Tl DEP'), ('mse', 'MSE - 212Bi FEP')):
        p = fit[name]
        print(f'----------------- {label} -----------------')
        print(f"Area:       {p['amp']:.1f} ± {p['amp_err']:.1f}")
        print(f"Mu:         {p['mu']:.3f} ± {p['mu_err']:.3f} keV")
        print(f"Sigma:      {p['sigma']:.3f} ± {p['sigma_err']:.3f} keV")
        print(f"Background: {p['bkg']:.2f} ± {p['bkg_err']:.2f} counts/keV\n")


def window_mask(energy, peak, k):
    """Events inside mu ± k*sigma of the given peak (fit['sse'] or fit['mse'])."""
    return (energy >= peak['mu'] - k * peak['sigma']) & (energy <= peak['mu'] + k * peak['sigma'])


def purity_table(fit, ks=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """
    Expected purity in a ±k sigma window, from the fit only:
        peak events = A * [Phi(k) - Phi(-k)],   background events = B * 2 k sigma
    """
    rows = []
    for k in ks:
        row = {'k_sigma': k}
        for name in ('sse', 'mse'):
            p = fit[name]
            peak = p['amp'] * (norm.cdf(k) - norm.cdf(-k))
            bkg = p['bkg'] * 2 * k * p['sigma']
            row[f'{name}_purity_%'] = 100 * peak / (peak + bkg)
            row[f'{name}_peak_events'] = peak
            row[f'{name}_bkg_events'] = bkg
        rows.append(row)
    return pd.DataFrame(rows)


def aoe_survival_fractions(clean, fit, k):
    """
    A/E survival fraction in the ±k sigma windows: events after / before the A/E low-side cut.
    """
    out = {}
    for name in ('sse', 'mse'):
        before = window_mask(clean['energy'], fit[name], k).sum()
        after = window_mask(clean['energy_AoE'], fit[name], k).sum()
        out[name] = 100 * after / before
        out[f'n_{name}_before'], out[f'n_{name}_after'] = int(before), int(after)
    return out


def print_survival_fractions(clean, fit, k=3):
    sf = aoe_survival_fractions(clean, fit, k)
    for name, label in (('sse', 'DEP (assumed SSE)'), ('mse', 'FEP (assumed MSE)')):
        p = fit[name]
        print(f"{label}: window [{p['mu'] - k*p['sigma']:.2f}, {p['mu'] + k*p['sigma']:.2f}] keV (±{k}σ)")
        print(f"  events before A/E cut = {sf[f'n_{name}_before']} | after = {sf[f'n_{name}_after']} "
              f"| SF = {sf[name]:.2f}%\n")
    return sf


# -----------------------------------------------------------------------------
# ------------------------------ DATA PREPARATION ------------------------------
# -----------------------------------------------------------------------------

def prepare_data(clean, fit, sigma, use_charge, current_mode, test_fraction=0.1, val_fraction=0.1, split_seed=42):
    """
    Builds the inputs of ONE configuration.

    - labels: SSE = DEP window ±sigma, MSE = FEP window ±sigma (from the single fit)
    - split on the labelled events: test = test_fraction, then val = val_fraction of the rest
      (same random_state and stratify as the old train_test -> same events)
    - channel 0 (current) is divided by the median peak of the TRAINING set
      ('normalised' -> scale = 1, nothing changes; 'scaled' -> global scale)
    - x_all contains EVERY event in the energy range (not only the labelled ones), with the
      same preprocessing and the same scale: it is used for the spectrum after the cuts.
      waves_train / waves_val / waves_test are rows of x_all (idx_train / idx_val / idx_test).
    """
    if current_mode not in CURRENT_MODES_ALLOWED:
        raise ValueError(f"current_mode must be one of {CURRENT_MODES_ALLOWED}, got {current_mode!r}")

    current_source = clean['current_normalised'] if current_mode == 'normalised' else clean['current']

    sse_mask = window_mask(clean['energy'], fit['sse'], sigma)
    mse_mask = window_mask(clean['energy'], fit['mse'], sigma)
    assert not (sse_mask & mse_mask).any(), "DEP and FEP windows overlap"

    idx_sse, idx_mse = np.where(sse_mask)[0], np.where(mse_mask)[0]
    idx_labeled = np.concatenate([idx_sse, idx_mse])
    labels_labeled = np.concatenate([np.zeros(len(idx_sse)), np.ones(len(idx_mse))])

    idx_train, idx_test, labels_train, labels_test = train_test_split(
        idx_labeled, labels_labeled, test_size=test_fraction, shuffle=True,
        random_state=split_seed, stratify=labels_labeled)

    # every event, (n, channels, 200); astype makes a copy so clean[...] is never modified
    x_all = ut.multichannel(clean['charge'], current_source, use_charge=use_charge).astype(np.float32)

    scale = float(ut.computeScale(x_all[idx_train, 0, :]))
    x_all[:, 0, :] /= scale

    idx_train, idx_val, labels_train, labels_val = train_test_split(
        idx_train, labels_train, test_size=val_fraction, shuffle=True,
        random_state=split_seed, stratify=labels_train)

    waves_train = x_all[idx_train]

    # initial observation noise, one value per channel: baseline std of the first 20 samples
    noise_init = [float(np.median(np.std(waves_train[:, c, :20], axis=1))) for c in range(x_all.shape[1])]

    return dict(
        x_all=x_all, scale=scale, noise_init=noise_init,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        waves_train=waves_train, waves_val=x_all[idx_val], waves_test=x_all[idx_test],
        labels_train=labels_train, labels_val=labels_val, labels_test=labels_test,
        aoe_test=clean['aoe'][idx_test],
        idx_labeled=idx_labeled, labels_labeled=labels_labeled, aoe_labeled=clean['aoe'][idx_labeled],
        sse_mask=sse_mask, mse_mask=mse_mask,
        n_sse=len(idx_sse), n_mse=len(idx_mse),
    )


# -----------------------------------------------------------------------------
# ---------------------------------- METRICS -----------------------------------
# -----------------------------------------------------------------------------

def bootstrap_auc(y, scores, n_boot=1000, seed=42):
    """
    AUC of every score in `scores` ({name: score}, higher score = more MSE-like) on the
    SAME events, with errors from a paired, stratified bootstrap:
    every bootstrap sample resamples the events (SSE and MSE separately, so the class sizes
    stay fixed) and recomputes ALL the AUCs on the same resampled events.

    Why paired: classifier and A/E are evaluated on the same events, so their AUC fluctuations
    are correlated. The error on the DIFFERENCE is taken directly from the bootstrap
    distribution of the difference, which accounts for that correlation automatically.

    Returns {name: {'auc', 'err'}} and, if there are exactly two scores,
    'difference': {'value', 'err', 'significance'} (first minus second).
    """
    y = np.asarray(y)
    names = list(scores)
    s = {k: np.asarray(v) for k, v in scores.items()}

    out = {k: {'auc': float(roc_auc_score(y, s[k]))} for k in names}

    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k in names:
            boot[k][b] = roc_auc_score(y[idx], s[k][idx])

    for k in names:
        out[k]['err'] = float(boot[k].std(ddof=1))

    if len(names) == 2:
        a, b = names
        diff = boot[a] - boot[b]
        value = out[a]['auc'] - out[b]['auc']
        err = float(diff.std(ddof=1))
        out['difference'] = dict(value=value, err=err, significance=value / err if err > 0 else np.inf)

    return out


def linear_probe(mu_train, labels_train, mu_eval):
    """
    Linear probe: logistic regression on the VAE latent means, fitted on the TRAINING set only
    and evaluated on mu_eval (e.g. the test set). It measures how much SSE/MSE information is
    LINEARLY readable in mu. Returns the probability of being MSE for every event of mu_eval.
    """
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(mu_train, labels_train)
    return probe.predict_proba(mu_eval)[:, 1]


def operating_point(rejected, labels):
    """
    Point (FPR, TPR) of a cut on the ROC plane. rejected[i] = True if the cut classifies event i
    as MSE (i.e. removes it). FPR = fraction of true SSE removed, TPR = fraction of true MSE removed.
    Equivalently FPR = 1 - SF(DEP), TPR = 1 - SF(FEP) on the same events.
    """
    rejected, labels = np.asarray(rejected, dtype=bool), np.asarray(labels)
    return float(rejected[labels == 0].mean()), float(rejected[labels == 1].mean())


def latent_correlations(mu, variable):
    """
    Spearman rank correlation between every latent dimension and a physical variable (A/E,
    rise time), on the events where the variable is finite. Spearman, not Pearson: it measures
    any MONOTONIC relation, not only a linear one. The sign of a latent dimension is arbitrary,
    so only |rho| matters.
    """
    variable = np.asarray(variable, dtype=float)
    ok = np.isfinite(variable) & np.all(np.isfinite(mu), axis=1)
    return np.array([spearmanr(mu[ok, j], variable[ok]).statistic for j in range(mu.shape[1])])


def latent_dimension_auc(mu, labels):
    """
    AUC of every latent dimension used ALONE as SSE/MSE score, folded to >= 0.5 (the sign of a
    latent dimension is arbitrary): tells which dimensions carry the SSE/MSE information.
    """
    aucs = np.array([roc_auc_score(labels, mu[:, j]) for j in range(mu.shape[1])])
    return np.maximum(aucs, 1 - aucs)


def matched_threshold(p_sse, target_sf):
    """
    Classifier threshold that keeps the same fraction of DEP events as the A/E cut
    (events with p < threshold are accepted as SSE-like): this makes the comparison of the
    FEP suppression fair, both cuts having the same DEP survival fraction.
    """
    return float(np.quantile(p_sse, target_sf / 100))


def classifier_survival_fractions(p, labels, threshold):
    """Survival fraction = events accepted as SSE-like (p < threshold), same meaning as the A/E SF."""
    return dict(sse=float(100 * (p[labels == 0] < threshold).mean()),
                mse=float(100 * (p[labels == 1] < threshold).mean()))


# -----------------------------------------------------------------------------
# ------------------------------ CONFIGURATIONS --------------------------------
# -----------------------------------------------------------------------------

def config_tags(use_charge, current_mode):
    channel_tag = 'current_charge' if use_charge else 'current'
    return channel_tag, current_mode


def run_tag(sigma, use_charge, current_mode, beta):
    channel_tag, current_tag = config_tags(use_charge, current_mode)
    return f'sigma{sigma:g}_{channel_tag}_{current_tag}_beta{beta:g}'


def config_dir(base_dir, sigma, use_charge, current_mode):
    channel_tag, current_tag = config_tags(use_charge, current_mode)
    return os.path.join(base_dir, f'sigma{sigma:g}', channel_tag, current_tag)
