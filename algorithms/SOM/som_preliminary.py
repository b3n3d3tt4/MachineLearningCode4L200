"""
Versione ridotta di algorithms/DTC/dtc_preliminary.py per la pipeline SOM: stesse load_data / clean_data /
create_loaders / count_peaks, ma SENZA importare TemporalAutoEncoder / DeepTemporalClustering,
che qui non servono (le uniche funzioni che li usavano erano sanity_check e sanity_check_dtc,
specifiche del DTC). In questo modo il modulo si importa anche in una cartella che non ha
TemporalAutoEncoder.py / DeepTemporalClustering.py.

I moduli comuni (detector, filereading, l200_utils) sono in common/: il notebook aggiunge common/
al sys.path prima di importare questo modulo.
"""

import numpy as np
import importlib
from sklearn.model_selection import train_test_split

import detector                # common/: detector name and folders
import filereading             # common/: reading of the checkpoints
import l200_utils as ut        # common/: waveform cleaning, loaders, plots


# ------------------------------------------------------------------------------------------
# ------------------------------------ 1. DATA LOADING --------------------------------------
# ------------------------------------------------------------------------------------------
def load_data(chosen_detector=None,                               # default: detector.DETECTOR
              periods=None,                                     # default: p03, p04, ..., p09
              e_min=1550, e_max=1650,                            # keV, energy window
              output_dir=None,                                   # folder of the checkpoints (default: detector.DATA_DIR)
              verbose=True):
    """
    Reads the checkpoints of all the periods. Returns a dictionary with
    charge_all, energy_all, AoE_all, risetime_all, charge_AoE_all, energy_AoE_all, risetime_AoE_all.
    The rise times (tp_90 - tp_10 from the psp tier) come from the risetime_pXX-pYY.npz file
    (or from the risetime_{period}.npy files) in output_dir.
    """

    chosen_detector = detector.DETECTOR if chosen_detector is None else chosen_detector
    output_dir = detector.DATA_DIR if output_dir is None else output_dir
    if periods is None:
        periods = [f"p{n:02d}" for n in range(3, 10)]  # p03, p04, ..., p09
    print("Periods to process:", periods)

    (charge_all, energy_all, AoE_all, risetime_all,
     charge_AoE_all, energy_AoE_all, risetime_AoE_all) = filereading.partition_from_checkpoint(
        periods,
        chosen_detector=chosen_detector,
        e_min=e_min,
        e_max=e_max,
        output_dir=output_dir,
        verbose=verbose  # every period is saved as a separate checkpoint
    )

    print("Waveforms shape (all periods):", charge_all.shape)
    print("Energy shape (all periods):", energy_all.shape)
    print("Amplitude vs Energy shape (all periods):", AoE_all.shape)
    print("Rise time shape (all periods):", risetime_all.shape)
    print("Waveforms shape after AoE cut (all periods):", charge_AoE_all.shape)
    print("Energy shape after AoE cut (all periods):", energy_AoE_all.shape)
    print("Rise time shape after AoE cut (all periods):", risetime_AoE_all.shape)

    return dict(charge_all=charge_all, energy_all=energy_all, AoE_all=AoE_all, risetime_all=risetime_all,
                charge_AoE_all=charge_AoE_all, energy_AoE_all=energy_AoE_all, risetime_AoE_all=risetime_AoE_all)


# ------------------------------------------------------------------------------------------
# ------------------------------------ 2. DATA CLEANING -------------------------------------
# ------------------------------------------------------------------------------------------
# first samples of the ALIGNED window used as baseline (noise, peak counting).
WINDOW_BASELINE = 20


def _print_rise_time_summary(rise_time, label):
    """NaN / non-positive values are kept (removing them would misalign the arrays), only counted."""
    finite = np.isfinite(rise_time)
    print(f"rise time ({label}): {len(rise_time)} events | NaN/inf: {(~finite).sum()} "
          f"| <= 0: {(rise_time[finite] <= 0).sum()}")
    if finite.any():
        print(f"  mean rise time [ns]: {np.mean(rise_time[finite]):.1f} "
              f"| median [ns]: {np.median(rise_time[finite]):.1f}")


def clean_data(raw, baseline_points=300, n_sigmas=3, pre_trigger_points=75, post_trigger_points=124, plot=True,
               normalise_charge=True):
    """
    Identico a dtc_preliminary.clean_data: pulizia con l200_utils.newProcessing, riallineamento
    di energia, A/E e rise time agli eventi sopravvissuti (il rise time è quello ufficiale
    tp_90 - tp_10 letto da filereading, non viene più calcolato qui). Stesso per gli eventi
    dopo il taglio A/E.
    """

    print(f"waveform cleaning in use: {ut.__file__}")
    print(f"Butterworth filter: {'ON' if ut.USES_BUTTERWORTH else 'OFF'}")
    print(f"Charge normalised: {'YES' if normalise_charge else 'NO'}\n")

    norm_kwargs = {} if normalise_charge else {'normalise': False}

    # ---------------- all the events (no A/E cut) ----------------
    charge_all = raw['charge_all']

    charge_aligned, current_aligned, good_idx, noise_idx, edge_idx = ut.newProcessing(
        charge_all,
        baseline_points=baseline_points, n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points, post_trigger_points=post_trigger_points,
        keep_noise=False,
        **norm_kwargs
    )
    current_aligned_normalised = ut.normalisation(current_aligned)

    if plot:
        ut.plot_aligned_currents(current_aligned, n_waveforms=100)

    print("Charge aligned:", charge_aligned.shape)
    print("Current aligned:", current_aligned.shape)
    print(f"Kept: {len(good_idx)} | noise: {len(noise_idx)} | edge: {len(edge_idx)} | total: {len(charge_all)}")

    energy_all = raw['energy_all'][good_idx]
    AoE_all = raw['AoE_all'][good_idx]
    rise_time_all = raw['risetime_all'][good_idx]

    _print_rise_time_summary(rise_time_all, "all events")

    # ---------------- events after the A/E cut ----------------
    charge_AoE_aligned, current_AoE_aligned, good_idx_AoE, noise_idx_AoE, edge_idx_AoE = ut.newProcessing(
        raw['charge_AoE_all'],
        baseline_points=baseline_points, n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points, post_trigger_points=post_trigger_points,
        keep_noise=False,
        **norm_kwargs
    )
    current_AoE_aligned_normalised = ut.normalisation(current_AoE_aligned)

    print("Charge AoE aligned:", charge_AoE_aligned.shape)
    print("Current AoE aligned:", current_AoE_aligned.shape)
    print(f"Kept: {len(good_idx_AoE)} | noise: {len(noise_idx_AoE)} | edge: {len(edge_idx_AoE)} | total: {len(raw['charge_AoE_all'])}")

    energy_AoE_all = raw['energy_AoE_all'][good_idx_AoE]
    rise_time_AoE_all = raw['risetime_AoE_all'][good_idx_AoE]

    _print_rise_time_summary(rise_time_AoE_all, "after A/E cut")

    for name, arr in [("corrente", current_aligned),
                      ("carica", charge_aligned)]:
        s50 = arr[:, :WINDOW_BASELINE].std(axis=1)
        s10 = arr[:, :10].std(axis=1)
        print(f"{name}: mediana su 50 punti {np.median(s50):.6f} | mediana su 10 punti {np.median(s10):.6f}")

    return dict(
        charge_aligned=charge_aligned, current_aligned=current_aligned,
        current_aligned_normalised=current_aligned_normalised,
        good_idx=good_idx, noise_idx=noise_idx, edge_idx=edge_idx,
        energy_all=energy_all, AoE_all=AoE_all, rise_time_all=rise_time_all,
        charge_AoE_all=charge_AoE_aligned, current_AoE_all=current_AoE_aligned,
        current_AoE_aligned_normalised=current_AoE_aligned_normalised,
        good_idx_AoE=good_idx_AoE, noise_idx_AoE=noise_idx_AoE, edge_idx_AoE=edge_idx_AoE,
        energy_AoE_all=energy_AoE_all, rise_time_AoE_all=rise_time_AoE_all,
        normalise_charge=normalise_charge,
    )


# ------------------------------------------------------------------------------------------
# ---------------------------------- 3. LOADERS CREATION ------------------------------------
# ------------------------------------------------------------------------------------------
def create_loaders(clean,
                   use_charge=False,
                   normalised=False,
                   use_sigma=False,
                   scale=False,
                   scale_charge=False,
                   sigma=1,
                   val_size=0.1, test_size=0.1, split_seed=42,
                   batch_size=128, loader_seed=0, all_batch_size=256,
                   n_debug=None):
    """
    Identico a preliminary.create_loaders. Per la SOM userai direttamente i tensori in `data`
    (es. data['current_used'][data['idx_train']]), non i tre DataLoader restituiti: torchsom
    vuole un tensore (N, num_features), non un DataLoader a mini-batch.
    """

    energy_all, AoE_all, rise_time_all = clean['energy_all'], clean['AoE_all'], clean['rise_time_all']
    current_aligned, charge_aligned = clean['current_aligned'], clean['charge_aligned']

    channels = 2 if use_charge else 1

    if use_sigma:
        sse_parameters, _ = ut.gaussian(energy_all, xmin=1587, xmax=1598)
        mse_parameters, _ = ut.gaussian(energy_all, xmin=1614, xmax=1626)
        sse_mask = np.abs(energy_all - sse_parameters[1]) <= sigma * sse_parameters[2]
        mse_mask = np.abs(energy_all - mse_parameters[1]) <= sigma * mse_parameters[2]
        idx_sel = np.concatenate([np.where(sse_mask)[0], np.where(mse_mask)[0]])
        labels = np.concatenate([np.zeros(sse_mask.sum()), np.ones(mse_mask.sum())]).astype(np.int64)
        print(f"sigma = {sigma} | SSE: {sse_mask.sum()} | MSE: {mse_mask.sum()}")
    else:
        idx_sel = np.arange(len(energy_all))
        labels = np.zeros(len(idx_sel), dtype=np.int64)

    energy       = energy_all[idx_sel]
    AoE          = AoE_all[idx_sel]
    rise_time    = rise_time_all[idx_sel]
    current_base = current_aligned[idx_sel]
    charge_base  = charge_aligned[idx_sel]

    idx_train, idx_rest = train_test_split(np.arange(len(energy)), test_size=val_size + test_size, shuffle=True, random_state=split_seed)
    idx_val, idx_test = train_test_split(idx_rest, test_size=test_size / (val_size + test_size), shuffle=True, random_state=split_seed)

    if normalised:
        current_used = ut.normalisation(current_base)
        scale_factor = 1.0
    elif scale:
        scale_factor = np.median(current_base[idx_train].max(axis=1))
        current_used = current_base / scale_factor
    else:
        scale_factor = 1.0
        current_used = current_base

    if scale_charge:
        charge_scale_factor = np.median(charge_base[idx_train].max(axis=1))
        charge_used = charge_base / charge_scale_factor
    else:
        charge_scale_factor = 1.0
        charge_used = charge_base

    current_noise = float(np.median(current_used[:, :WINDOW_BASELINE].std(axis=1)))
    charge_noise  = float(np.median(charge_used[:, :WINDOW_BASELINE].std(axis=1)))

    waves = ut.multichannel(charge_used, current_used, use_charge=use_charge).astype(np.float32)

    print("use_charge:", use_charge, "| use_sigma:", use_sigma, "| normalised:", normalised,
          "| scale:", scale, f"(scale factor = {scale_factor})",
          "| scale_charge:", scale_charge, f"(charge scale factor = {charge_scale_factor})")
    print("waves:", waves.shape, "| NaN/inf:", (~np.isfinite(waves)).sum())
    print("rumore corrente su questa scala (current_noise):", current_noise)
    print("rumore carica su questa scala (charge_noise):  ", charge_noise)
    if not normalised:
        print("correlazione altezza picco - A/E:", np.corrcoef(current_used.max(axis=1), AoE)[0, 1])

    if n_debug is not None:
        idx_train = idx_train[:n_debug]

    train_loader = ut.epoch_loader(waves[idx_train], labels[idx_train], channels=channels, batch_size=batch_size, seed=loader_seed)
    validation_loader = ut.test_loader(waves[idx_val], labels[idx_val], channels=channels, batch_size=batch_size)
    test_loader = ut.test_loader(waves[idx_test], labels[idx_test], channels=channels, batch_size=batch_size)

    all_loader = ut.test_loader(waves, labels, channels=channels, batch_size=all_batch_size)

    print("train:", len(idx_train), "| val:", len(idx_val), "| test:", len(idx_test))
    print("lunghezze coerenti:", len(waves) == len(energy) == len(AoE) == len(rise_time) == len(labels))

    data = dict(waves=waves, energy=energy, AoE=AoE, rise_time=rise_time, labels=labels,
                current_used=current_used, charge_base=charge_base, charge_used=charge_used,
                charge_scale_factor=charge_scale_factor,
                current_noise=current_noise, charge_noise=charge_noise,
                idx_sel=idx_sel, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
                test_loader=test_loader,
                scale_factor=scale_factor, channels=channels,
                config=dict(use_charge=use_charge, normalised=normalised, use_sigma=use_sigma,
                            scale=scale, scale_charge=scale_charge, sigma=sigma,
                            normalise_charge=clean.get('normalise_charge', True),
                            butterworth=ut.USES_BUTTERWORTH))

    return train_loader, validation_loader, all_loader, data


# ------------------------------------------------------------------------------------------
# -------------------------------- (OPTIONAL) PEAK DETECTION --------------------------------
# ------------------------------------------------------------------------------------------
def count_peaks(current, baseline_points=None, n_sigmas=3, prominence_fraction=0.2, min_distance=5):
    """
    Number of peaks of every current waveform. A peak counts if it is above
    baseline_mean + n_sigmas * baseline_std AND its prominence is at least
    prominence_fraction of the maximum of that waveform.
    """
    from scipy.signal import find_peaks

    if baseline_points is None:
        baseline_points = WINDOW_BASELINE
    baseline_slice = current[:, :baseline_points]
    threshold = baseline_slice.mean(axis=1) + n_sigmas * baseline_slice.std(axis=1)
    maximum = current.max(axis=1)

    n_peaks = np.zeros(len(current), dtype=int)
    for i in range(len(current)):
        peaks, _ = find_peaks(current[i],
                              height=threshold[i],
                              prominence=prominence_fraction * maximum[i],
                              distance=min_distance)
        n_peaks[i] = len(peaks)
    return n_peaks
