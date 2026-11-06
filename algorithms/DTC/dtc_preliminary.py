"""
Everything that was previously done cell by cell in the notebooks before the DTC model:

    sanity_check, sanity_check_dtc  -> shape / gradient checks of the TAE and of the DTC
    load_data                       -> 1. DATA LOADING   (from the saved checkpoints)
    clean_data                      -> 2. DATA CLEANING  (via l200_utils.newProcessing)
                                       the rise time is NOT computed here any more: it is the
                                       official one (tp_90 - tp_10 from the psp tier) read by
                                       filereading, and it is realigned exactly like energy and A/E
    create_loaders                  -> 3. LOADERS CREATION
    count_peaks                     -> (optional) peak detection

The shared modules (detector, filereading, l200_utils) are in common/, shared by the DTC, the VAE and
the SOM: the notebook adds common/ and this folder to sys.path before importing this module.
`detector` gives the detector name and the folders, and puts the detector name on every plot.
"""

import numpy as np
import importlib
import torch
from sklearn.model_selection import train_test_split

import detector                       # common/: detector name and folders
import filereading                    # common/: reading of the checkpoints
import l200_utils as ut               # common/: waveform cleaning, loaders, plots
import TemporalAutoEncoder as tae
import DeepTemporalClustering as DTC


# ------------------------------------------------------------------------------------------
# -------------------------------------- SANITY CHECKS -------------------------------------
# ------------------------------------------------------------------------------------------
def sanity_check(use_charge=False, batch_size=4, input_length=200):
    channels = 2 if use_charge else 1
    
    model = tae.TemporalAutoEncoder(input_length=input_length, use_charge=use_charge)
    
    # Tensore finto, stessa forma dei tuoi dati veri: (batch, channels, length)
    x = torch.randn(batch_size, channels, input_length)
    
    latent, reconstructed = model(x)
    
    expected_latent_len = input_length // model.pool_size
    print(f"use_charge={use_charge}")
    print(f"  input shape:         {tuple(x.shape)}")
    print(f"  latent shape:        {tuple(latent.shape)}  (atteso: ({batch_size}, {model.n_units[-1]}, {expected_latent_len}))")
    print(f"  reconstructed shape: {tuple(reconstructed.shape)}  (atteso: ({batch_size}, {channels}, {input_length}))")
    
    assert latent.shape == (batch_size, model.n_units[-1], expected_latent_len), "Shape del latente sbagliata"
    assert reconstructed.shape == x.shape, "Shape della ricostruzione sbagliata"
    
    # nll_loss, non mse_loss: è l'unica delle due che usa anche logvar_obs,
    # quindi è l'unica per cui ha senso pretendere che OGNI parametro abbia gradiente
    loss = tae.nll_loss(reconstructed, x, model.logvar_obs)
    loss.backward()
    
    n_missing_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"  loss: {loss.item():.4f}")
    print(f"  parametri senza gradiente: {n_missing_grad} (atteso: 0)")
    assert n_missing_grad == 0, "Qualche parametro non riceve gradiente: c'è un pezzo del grafo scollegato dalla loss"
    
    # Controllo in più, specifico per logvar_obs: non basta che abbia un gradiente,
    # verifichiamo che non sia banalmente zero (altrimenti backward() "gira" ma
    # quel parametro di fatto non si muoverebbe mai durante l'ottimizzazione)
    print(f"  logvar_obs.grad: {model.logvar_obs.grad.numpy()}")
    assert not torch.allclose(model.logvar_obs.grad, torch.zeros_like(model.logvar_obs.grad)), \
        "Il gradiente di logvar_obs è zero: non si aggiornerebbe durante il training"
    
    print("  OK\n")


def sanity_check_dtc(use_charge=False, n_clusters=2, batch_size=4, input_length=200, pool_size=10):
    channels = 2 if use_charge else 1
    
    dtc = DTC.DTC(
        input_length=input_length,
        pool_size=pool_size,
        n_clusters=n_clusters,
        use_charge=use_charge
    )
    
    x = torch.randn(batch_size, channels, input_length)
    
    reconstructed, q = dtc(x)
    
    print(f"use_charge={use_charge}")
    print(f"  input shape:         {tuple(x.shape)}")
    print(f"  reconstructed shape: {tuple(reconstructed.shape)}  (atteso: {tuple(x.shape)})")
    print(f"  q shape:             {tuple(q.shape)}  (atteso: ({batch_size}, {n_clusters}))")
    
    assert reconstructed.shape == x.shape, "Shape della ricostruzione sbagliata"
    assert q.shape == (batch_size, n_clusters), "Shape di q sbagliata"
    
    row_sums = q.sum(dim=1)
    print(f"  q.sum(dim=1):        {row_sums.detach().numpy()}  (atteso: tutti ~1.0)")
    assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5), "Le righe di q non sommano a 1"
    
    n_params = len(list(dtc.parameters()))
    print(f"  numero di tensori allenabili: {n_params}  (atteso: > 0)")
    assert n_params > 0, "Nessun parametro registrato: tae/tsc non sono stati riconosciuti come sotto-moduli"
    
    # Loss combinata: nll_loss copre encoder+decoder+logvar_obs (ramo TAE),
    # q.sum() copre encoder+centroidi (ramo clustering). Serve entrambe insieme
    # per toccare OGNI parametro del modello in un colpo solo.
    loss = tae.nll_loss(reconstructed, x, dtc.tae.logvar_obs) + q.sum()
    loss.backward()
    
    n_missing_grad = sum(1 for p in dtc.parameters() if p.requires_grad and p.grad is None)
    print(f"  loss: {loss.item():.4f}")
    print(f"  parametri senza gradiente: {n_missing_grad} (atteso: 0)")
    assert n_missing_grad == 0, "Qualche parametro non riceve gradiente"
    
    # Controllo specifico sui centroidi
    print(f"  clusters.grad è None: {dtc.tsc.clusters.grad is None} (atteso: False)")
    assert dtc.tsc.clusters.grad is not None, "I centroidi non ricevono gradiente"
    
    print("  OK\n")


# ------------------------------------------------------------------------------------------
# ------------------------------------- 1. DATA LOADING ------------------------------------
# ------------------------------------------------------------------------------------------
def load_data(chosen_detector=None,                            # default: detector.DETECTOR
              periods=None,                                    # default: p03, p04, ..., p09
              e_min=1550, e_max=1650,                          # keV, energy window
              output_dir=None,                                 # folder of the checkpoints (default: detector.DATA_DIR)
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
# ------------------------------------ 2. DATA CLEANING ------------------------------------
# ------------------------------------------------------------------------------------------
# first samples of the ALIGNED window used as baseline (noise, peak counting).
# The window has pre_trigger_points = 50 samples before the first current peak: the slowest rises
# start a few tens of samples before the peak, so only the first 20 are taken as flat baseline.
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
    normalise_charge=False: the charge is NOT normalised (only baseline subtracted) and the current is
    computed from it. Requires a l200_utils.newProcessing with the `normalise` argument.
    Cleaning of the waveforms with l200_utils.newProcessing (with or without Butterworth
    depending on l200_utils.FILTER_TYPE), realignment of energy, A/E and rise time to the
    events that survive (the rise time is the official tp_90 - tp_10 read by filereading, it is
    no longer computed here). Same for the events after the A/E cut.
    Returns a dictionary with everything that is needed later.
    """
    
    print(f"waveform cleaning in use: {ut.__file__}")
    print(f"Butterworth filter: {'ON' if ut.USES_BUTTERWORTH else 'OFF'}")
    print(f"Charge normalised: {'YES' if normalise_charge else 'NO'}\n")
    
    # the argument is passed only when needed: kept from older versions of the cleaning
    norm_kwargs = {} if normalise_charge else {'normalise': False}
    
    # ---------------- all the events (no A/E cut) ----------------
    charge_all = raw['charge_all']
    
    charge_aligned, current_aligned, good_idx, noise_idx, edge_idx = ut.newProcessing(
        charge_all,
        baseline_points=baseline_points, n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points, post_trigger_points=post_trigger_points,
        keep_noise=False,   # per il training niente righe di NaN
        **norm_kwargs
    )
    current_aligned_normalised = ut.normalisation(current_aligned)
    
    if plot:
        ut.plot_aligned_currents(current_aligned, n_waveforms=100)
    
    print("Charge aligned:", charge_aligned.shape)
    print("Current aligned:", current_aligned.shape)
    print(f"Kept: {len(good_idx)} | noise: {len(noise_idx)} | edge: {len(edge_idx)} | total: {len(charge_all)}")
    
    # riallineamento alle nuove wf anche per AoE, energia e rise time
    energy_all = raw['energy_all'][good_idx]
    AoE_all = raw['AoE_all'][good_idx]
    rise_time_all = raw['risetime_all'][good_idx]
    
    _print_rise_time_summary(rise_time_all, "all events")
    
    # ---------------- events after the A/E cut ----------------
    charge_AoE_aligned, current_AoE_aligned, good_idx_AoE, noise_idx_AoE, edge_idx_AoE = ut.newProcessing(
        raw['charge_AoE_all'],
        baseline_points=baseline_points, n_sigmas=n_sigmas,
        pre_trigger_points=pre_trigger_points, post_trigger_points=post_trigger_points,   # 200 campioni, stesso criterio usato sopra
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
    
    # ---------------- noise of the baseline ----------------
    for name, arr in [("corrente", current_aligned),
                      ("carica", charge_aligned)]:
        s50 = arr[:, :WINDOW_BASELINE].std(axis=1)      # baseline: primi WINDOW_BASELINE campioni della finestra
        s10 = arr[:, :10].std(axis=1)
        print(f"{name}: mediana su 50 punti {np.median(s50):.6f} | mediana su 10 punti {np.median(s10):.6f}")
    
    return dict(
        # all the events (no A/E cut)
        charge_aligned=charge_aligned, current_aligned=current_aligned,
        current_aligned_normalised=current_aligned_normalised,
        good_idx=good_idx, noise_idx=noise_idx, edge_idx=edge_idx,
        energy_all=energy_all, AoE_all=AoE_all, rise_time_all=rise_time_all,
        # events after the A/E cut
        charge_AoE_all=charge_AoE_aligned, current_AoE_all=current_AoE_aligned,
        current_AoE_aligned_normalised=current_AoE_aligned_normalised,
        good_idx_AoE=good_idx_AoE, noise_idx_AoE=noise_idx_AoE, edge_idx_AoE=edge_idx_AoE,
        energy_AoE_all=energy_AoE_all, rise_time_AoE_all=rise_time_AoE_all,
        normalise_charge=normalise_charge,
    )


# ------------------------------------------------------------------------------------------
# ---------------------------------- 3. LOADERS CREATION -----------------------------------
# ------------------------------------------------------------------------------------------
def create_loaders(clean,
                   use_charge=False,   # True: current + charge as 2 channels | False: current only
                   normalised=False,   # True: corrente normalizzata a 1 evento per evento | False: corrente con ampiezza (A/E)
                   use_sigma=False,    # True: solo gli eventi nelle finestre SSE (DEP) e MSE (FEP) | False: tutta la finestra 1550-1650 keV
                   scale=False,        # True: corrente divisa per la mediana dei massimi del training set | False: nessuna scala
                   scale_charge=False, # True: carica divisa per la mediana dei massimi del training set | False: nessuna scala
                   sigma=1,            # width of the DEP/FEP windows in sigmas (used only if use_sigma=True)
                   val_size=0.1, test_size=0.1, split_seed=42,
                   batch_size=128, loader_seed=0, all_batch_size=256,
                   n_debug=None):      # metti 20000 per un test veloce
    """
    Builds train_loader (training set, shuffled), validation_loader (validation set) and
    all_loader (ALL the events, no shuffle, for the final plots).
    The events are split in three: training (used to train), validation (used during the
    training to monitor the losses) and test (never seen during the training, only for the
    final plots). val_size and test_size are fractions of all the selected events.
    The test loader is in data['test_loader'].
    
    Returns train_loader, validation_loader, all_loader, data
    where data is a dictionary with waves, energy, AoE, rise_time, labels, idx_train, idx_val,
    idx_test, ... (same events, same order as all_loader).
    
    Note on `scale`: it only acts when normalised=False (a normalised current has peak 1 already).
    Note on `scale_charge`: one single number for all the events, so the amplitude differences
    between events are kept; meant for the NON-normalised charge (clean_data(normalise_charge=False)).
    data['current_noise'] and data['charge_noise'] are the baseline noise of the two channels on the
    scale actually fed to the network (median std of the first WINDOW_BASELINE samples of the window).
    """
    
    energy_all, AoE_all, rise_time_all = clean['energy_all'], clean['AoE_all'], clean['rise_time_all']
    current_aligned, charge_aligned = clean['current_aligned'], clean['charge_aligned']
    
    channels = 2 if use_charge else 1
    
    # --- 1) quali eventi usare ---
    if use_sigma:
        # stessi tagli della vecchia sse_mse_separation, ma tenuti come INDICI:
        # così la stessa selezione si applica anche a rise_time, non solo a energia e A/E
        sse_parameters, _ = ut.gaussian(energy_all, xmin=1587, xmax=1598)
        mse_parameters, _ = ut.gaussian(energy_all, xmin=1614, xmax=1626)
        sse_mask = np.abs(energy_all - sse_parameters[1]) <= sigma * sse_parameters[2]
        mse_mask = np.abs(energy_all - mse_parameters[1]) <= sigma * mse_parameters[2]
        idx_sel = np.concatenate([np.where(sse_mask)[0], np.where(mse_mask)[0]])      # SSE prima, poi MSE
        labels = np.concatenate([np.zeros(sse_mask.sum()), np.ones(mse_mask.sum())]).astype(np.int64)   # 0 = SSE, 1 = MSE (solo per misurare)
        print(f"sigma = {sigma} | SSE: {sse_mask.sum()} | MSE: {mse_mask.sum()}")
    else:
        idx_sel = np.arange(len(energy_all))
        labels = np.zeros(len(idx_sel), dtype=np.int64)          # i loader vogliono le etichette, ma qui non le usiamo
    
    energy       = energy_all[idx_sel]
    AoE          = AoE_all[idx_sel]
    rise_time    = rise_time_all[idx_sel]
    current_base = current_aligned[idx_sel]
    charge_base  = charge_aligned[idx_sel]
    
    # --- 2) split train / validation / test (prima della scala, che si calcola sul solo training) ---
    idx_train, idx_rest = train_test_split(np.arange(len(energy)), test_size=val_size + test_size, shuffle=True, random_state=split_seed)
    idx_val, idx_test = train_test_split(idx_rest, test_size=test_size / (val_size + test_size), shuffle=True, random_state=split_seed)
    
    # --- 3) corrente: normalizzata a picco 1, oppure con ampiezza (eventualmente divisa per una scala unica) ---
    if normalised:
        current_used = ut.normalisation(current_base)
        scale_factor = 1.0
    elif scale:
        scale_factor = np.median(current_base[idx_train].max(axis=1))
        current_used = current_base / scale_factor
    else:
        scale_factor = 1.0
        current_used = current_base
    
    # --- 4) carica: eventualmente divisa per una scala unica (calcolata sul solo training) ---
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
    
    # loader su TUTTI gli eventi, senza shuffle: serve per i grafici finali
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
# ------------------------------- (OPTIONAL) PEAK DETECTION --------------------------------
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
