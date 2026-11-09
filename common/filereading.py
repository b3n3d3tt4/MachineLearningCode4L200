"""
------------------------------------------------------------------------------------------
------ I strongly believe that alla of these data loading processes would have been ------
------ much easier with Polars, but due to lack of experience with LEGEND data I did -----
------ not know such method. Could be interesting to implement all the code with it ------
------------------------------------------------------------------------------------------

Functions to extract "clean" waveforms and energies (quality cuts + energy cut)
from LEGEND lh5 files, at three increasing levels of granularity:

    extract_run           -> a single run from a single period
    extract_period        -> all runs from a single period
    extract_all_periods   -> all runs from multiple periods

Each level returns 7 arrays:
    wf, energy, AoE, risetime              -> quality cuts + energy cut (NO A/E cut)
    wf_AoE, energy_AoE, risetime_AoE       -> quality cuts + energy cut + A/E low-side cut

The rise time is computed as tp_90 - tp_10, both read from the psp tier
(ch.../dsp/tp_10 and ch.../dsp/tp_90). It is masked with exactly the same
mask used for waveforms and energies, so it is aligned event by event.

`extract_period` and `extract_all_periods` support disk checkpointing:
if you pass `output_dir`, each period is saved as .npy files and, if you call the
function a second time, the previously saved period is reloaded instead
of being recomputed (unless `force_recompute=True`).
`extract_all_periods` also writes a single file risetime_{first}-{last}.npz in
`output_dir`, containing ONLY the rise times (keys "risetime" and "risetime_AoE").

On a machine without the lh5 files, `period_from_checkpoint` and
`partition_from_checkpoint` reload the saved checkpoints. The rise times are taken
from risetime_{period}.npy if present, otherwise they are cut out of the single
risetime_{first}-{last}.npz file (see `load_risetime`).
"""

import os
import glob
import re
import json
import random

import numpy as np
import matplotlib.pyplot as plt

# lh5 is only needed to READ the original LEGEND files (extract_* functions).
# On a machine that only has the checkpoints (.npy / .npz) it can be missing:
# the *_from_checkpoint functions and load_risetime work without it.
try:
    import lh5
except ImportError:
    lh5 = None


# Names of the rise-time fields inside the psp tier (ch.../<group>/<field>).
# Change them here if your production uses different names.
PSP_GROUP = "dsp"
TP10_FIELD = "tp_10"
TP90_FIELD = "tp_90"


# -----------------------------------------------------------------------------
# ----------------------------- Low level Helpers -----------------------------
# -----------------------------------------------------------------------------

def build_timestamp_dict(file_list):
    """
    Timestamp map -> filepath, extraxted from the name of the file
    (e.g. l200-pXX-rYYY-cal-YYYYMMDDThhmmssZ-tier_raw.lh5 -> "YYYYMMDDThhmmssZ").
    """
    d = {}
    for f in file_list:
        match = re.search(r'\d{8}T\d{6}Z', os.path.basename(f))
        if match:
            d[match.group()] = f
    return d

def get_runs(period, base_path):
    """
    Returns the list of the runs (es. ['r000', 'r001', ...]) found in base_path/period/.
    """
    period_path = os.path.join(base_path, period)
    run_dirs = sorted(glob.glob(os.path.join(period_path, "r*")))
    return [os.path.basename(os.path.normpath(r)) for r in run_dirs if os.path.isdir(r)]


def get_periods(base_path, pattern="p*"):
    """
    Returns the list of the periods (es. ['p03', 'p04', ...]) found in base_path.
    """
    period_dirs = sorted(glob.glob(os.path.join(base_path, pattern)))
    return [os.path.basename(os.path.normpath(p)) for p in period_dirs if os.path.isdir(p)]


def _empty_outputs(n=7):
    return tuple(np.array([]) for _ in range(n))


# ------------------------------------------------------------------------------------------------
# ----------------------------- Level 1: a single run (file by file) -----------------------------
# ------------------------------------------------------------------------------------------------

def extract_clean_data(rawWF_folder_path, par_folder_path, psp_folder_path, chosen_detector, lmeta,
                       e_min=1550, e_max=1650):
    """
    Reads all lh5 files of a SINGLE run, applies the quality cuts + energy cut,
    and returns seven lists (one entry per file):

        run_waveforms      -> waveforms passing quality + energy cuts (no A/E cut)
        run_energy         -> energies passing quality + energy cuts (no A/E cut)
        run_AoE            -> A/E classifier values passing quality + energy cuts (no A/E cut)
        run_risetime       -> rise time (tp_90 - tp_10) passing quality + energy cuts (no A/E cut)
        run_waveforms_AoE  -> waveforms passing quality + energy cuts + A/E low-side cut
        run_energy_AoE     -> energies passing quality + energy cuts + A/E low-side cut
        run_risetime_AoE   -> rise time passing quality + energy cuts + A/E low-side cut

    The files from the raw, pht and psp folders are matched by TIMESTAMP (not via a
    zip based on alphabetical order), so a missing or extra file in any of the three
    folders can never silently misalign waveforms, energies and rise times.
    """

    if lh5 is None:
        raise ImportError("lh5 is not installed on this machine: use the *_from_checkpoint functions instead.")

    wf_file_list = sorted(glob.glob(os.path.join(rawWF_folder_path, "*.lh5")))
    energy_file_list = sorted(glob.glob(os.path.join(par_folder_path, "*.lh5")))
    psp_file_list = sorted(glob.glob(os.path.join(psp_folder_path, "*.lh5")))

    wf_dict = build_timestamp_dict(wf_file_list)
    energy_dict = build_timestamp_dict(energy_file_list)
    psp_dict = build_timestamp_dict(psp_file_list)

    common_timestamps = sorted(set(wf_dict) & set(energy_dict) & set(psp_dict))

    # Segnala i timestamp che esistono in raw/pht ma non in psp (verrebbero persi in silenzio)
    missing_psp = sorted((set(wf_dict) & set(energy_dict)) - set(psp_dict))
    if missing_psp:
        print(f"{len(missing_psp)} timestamp(s) present in raw/pht but missing in psp "
              f"-> skipped: {missing_psp}")

    run_waveforms = []
    run_energy = []
    run_AoE = []
    run_risetime = []
    run_waveforms_AoE = []
    run_energy_AoE = []
    run_risetime_AoE = []

    for timestamp in common_timestamps:
        wf_file = wf_dict[timestamp]
        energy_file = energy_dict[timestamp]
        psp_file = psp_dict[timestamp]
        filename = os.path.basename(wf_file)

        try:
            # Channel map ESATTA per questo timestamp
            channel_map = lmeta.channelmap(timestamp)

            # Il detector era presente/attivo in questo file?
            if chosen_detector not in channel_map:
                print(f"Detector {chosen_detector} not present in {filename}. Skipping.")
                continue

            daq_channel = f"ch{channel_map[chosen_detector]['daq']['rawid']}"

            internal_wf_path = f"{daq_channel}/raw/waveform_windowed"
            internal_baseline_pileup_path = f"{daq_channel}/hit/bl_pileup_cut"
            internal_tail_pileup_path = f"{daq_channel}/hit/tail_pileup_cut"
            internal_valid_path = f"{daq_channel}/hit/is_valid_cal"
            internal_saturated_path = f"{daq_channel}/hit/is_saturated"
            internal_discharge_path = f"{daq_channel}/hit/is_discharge"
            internal_energy_path = f"{daq_channel}/hit/daqenergy_cal"
            internal_AoE_path = f"{daq_channel}/hit/AoE_Classifier"
            internal_cuts_AoE_path = f"{daq_channel}/hit/AoE_Low_Cut"
            internal_tp10_path = f"{daq_channel}/{PSP_GROUP}/{TP10_FIELD}"
            internal_tp90_path = f"{daq_channel}/{PSP_GROUP}/{TP90_FIELD}"

            data_wf = lh5.read(internal_wf_path, wf_file)
            energy_wf = lh5.read(internal_energy_path, energy_file)
            AoE_wf = lh5.read(internal_AoE_path, energy_file)
            tp10_wf = lh5.read(internal_tp10_path, psp_file)
            tp90_wf = lh5.read(internal_tp90_path, psp_file)

            if data_wf is None or energy_wf is None or tp10_wf is None or tp90_wf is None:
                continue

            baseline_pileup = lh5.read(internal_baseline_pileup_path, energy_file).nda.astype(bool)
            tail_pileup = lh5.read(internal_tail_pileup_path, energy_file).nda.astype(bool)
            valid = lh5.read(internal_valid_path, energy_file).nda.astype(bool)
            saturated = lh5.read(internal_saturated_path, energy_file).nda.astype(bool)
            discharge = lh5.read(internal_discharge_path, energy_file).nda.astype(bool)
            AoE = lh5.read(internal_cuts_AoE_path, energy_file).nda.astype(bool)

            wf_array = data_wf['values'].nda
            energy_array = energy_wf.nda
            AoE_array = AoE_wf.nda
            tp10_array = tp10_wf.nda
            tp90_array = tp90_wf.nda

            if len(wf_array) != len(energy_array):
                print(f"Mismatch lunghezza wf/energy in {filename}: "
                      f"{len(wf_array)} vs {len(energy_array)}. Skipping.")
                continue

            if len(tp10_array) != len(energy_array) or len(tp90_array) != len(energy_array):
                print(f"Mismatch lunghezza psp/energy in {filename}: "
                      f"tp10={len(tp10_array)}, tp90={len(tp90_array)} vs {len(energy_array)}. Skipping.")
                continue

            risetime_array = tp90_array - tp10_array

            energy_cut = (energy_array >= e_min) & (energy_array <= e_max)

            mask = baseline_pileup & tail_pileup & valid & (~saturated) & (~discharge) & energy_cut
            mask_AoE = mask & AoE

            run_waveforms.append(wf_array[mask])
            run_energy.append(energy_array[mask])
            run_AoE.append(AoE_array[mask])
            run_risetime.append(risetime_array[mask])
            run_waveforms_AoE.append(wf_array[mask_AoE])
            run_energy_AoE.append(energy_array[mask_AoE])
            run_risetime_AoE.append(risetime_array[mask_AoE])

        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping file.")
            continue

    return (run_waveforms, run_energy, run_AoE, run_risetime,
            run_waveforms_AoE, run_energy_AoE, run_risetime_AoE)


def extract_run(period, run, base_rawWF_path, base_par_path, base_psp_path, chosen_detector, lmeta,
                 e_min=1550, e_max=1650, verbose=True, plot=False):
    """
    Extract and concatenate all the clean events of ONE run.

    Returns (wf_run, energy_run, AoE_run, risetime_run,
             wf_run_AoE, energy_run_AoE, risetime_run_AoE) as concatenated arrays
    (empty arrays if no event survives or if the run does not exist).
    """

    rawWF_folder_path = os.path.join(base_rawWF_path, period, run) + "/"
    par_folder_path = os.path.join(base_par_path, period, run) + "/"
    psp_folder_path = os.path.join(base_psp_path, period, run) + "/"

    if not os.path.isdir(par_folder_path):
        if verbose:
            print(f"Parameter folder missing for {period}/{run}. Skipping run.")
        return _empty_outputs()

    if not os.path.isdir(psp_folder_path):
        if verbose:
            print(f"psp folder missing for {period}/{run}. Skipping run.")
        return _empty_outputs()

    if verbose:
        print(f"Processing {period}/{run} ...")

    lists = extract_clean_data(
        rawWF_folder_path, par_folder_path, psp_folder_path,
        chosen_detector, lmeta, e_min=e_min, e_max=e_max
    )

    if lists[0]:
        (wf_run, energy_run, AoE_run, risetime_run,
         wf_run_AoE, energy_run_AoE, risetime_run_AoE) = (np.concatenate(l, axis=0) for l in lists)
    else:
        (wf_run, energy_run, AoE_run, risetime_run,
         wf_run_AoE, energy_run_AoE, risetime_run_AoE) = _empty_outputs()

    if verbose:
        print(f"  -> {wf_run.shape[0]} events survived the cuts in {period}/{run} "
              f"({wf_run_AoE.shape[0]} after the A/E cut)")

    if plot:
        if wf_run.shape[0] > 0:
            print(f'Here is an example of a waveform of {period}/{run}')
            # Generate a random integer between 0 and the number of waveforms
            idx = np.random.randint(wf_run.shape[0])

            plt.plot(wf_run[idx])
            plt.title(fr"Charge raw waveforms - example from $\mathit{{{period}/{run}}}$", fontsize=12)
            plt.xlabel("Time [ns]", fontsize=10)
            plt.ylabel("Charge [ADC counts]", fontsize=10)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()

    return wf_run, energy_run, AoE_run, risetime_run, wf_run_AoE, energy_run_AoE, risetime_run_AoE


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_KEYS = ["wf", "energy", "AoE", "risetime", "wf_AoE", "energy_AoE", "risetime_AoE"]


def _checkpoint_paths(output_dir, period):
    """
    Returns (list of the 7 .npy paths, in the same order as _CHECKPOINT_KEYS, meta_path).
    """
    npy_paths = [os.path.join(output_dir, f"{key}_{period}.npy") for key in _CHECKPOINT_KEYS]
    meta_path = os.path.join(output_dir, f"meta_{period}.json")
    return npy_paths, meta_path


def _checkpoint_matches(meta_path, chosen_detector, e_min, e_max):
    """
    Returns True only if the existing checkpoiint was generated with the very same detector and very same energy window
    """
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    return (
        meta.get("chosen_detector") == chosen_detector
        and meta.get("e_min") == e_min
        and meta.get("e_max") == e_max
    )


# -----------------------------------------------------------------------------------------------
# ----------------------------- Level 2: a whole period (every run) -----------------------------
# -----------------------------------------------------------------------------------------------

def extract_period(period, base_rawWF_path, base_par_path, base_psp_path, chosen_detector, lmeta,
                    e_min=1550, e_max=1650, output_dir=None, force_recompute=False, verbose=True, plot=False):
    """
    Extract and concatenate all the clean events of all the runs of ONE period.

    If `output_dir` is specified:
      - if a checkpoint already exists (wf_, energy_, AoE_, risetime_, wf_AoE_, energy_AoE_,
        risetime_AoE_{period}.npy) and force_recompute=False, it reloads this
        instead of computing it all over again;
      - otherwise it computes and then saves the checkpoint for the next time.
    Old checkpoints without the risetime files are treated as missing and recomputed.

    Returns (wf_period, energy_period, AoE_period, risetime_period,
             wf_AoE_period, energy_AoE_period, risetime_AoE_period).
    """

    if output_dir is not None:
        npy_paths, meta_path = _checkpoint_paths(output_dir, period)
        checkpoint_exists = all(os.path.exists(p) for p in npy_paths)

        if checkpoint_exists and not force_recompute:
            if _checkpoint_matches(meta_path, chosen_detector, e_min, e_max):
                if verbose:
                    print(f"Period {period}: checkpoint found (same detector/energy window), loading from disk.")
                return tuple(np.load(p) for p in npy_paths)
            elif verbose:
                print(f"Period {period}: checkpoint found but detector/energy window changed "
                      f"(or metadata missing) -> recomputing.")

    run_list = get_runs(period, base_rawWF_path)
    if verbose:
        print(f"Period {period}: found {len(run_list)} runs -> {run_list}")

    period_lists = [[] for _ in _CHECKPOINT_KEYS]

    for run in run_list:
        run_out = extract_run(
            period, run, base_rawWF_path, base_par_path, base_psp_path,
            chosen_detector, lmeta, e_min=e_min, e_max=e_max, verbose=verbose, plot=plot
        )
        if run_out[0].shape[0] > 0:
            for lst, arr in zip(period_lists, run_out):
                lst.append(arr)

    if period_lists[0]:
        period_out = tuple(np.concatenate(l, axis=0) for l in period_lists)
    else:
        period_out = _empty_outputs()

    (wf_period, energy_period, AoE_period, risetime_period,
     wf_AoE_period, energy_AoE_period, risetime_AoE_period) = period_out

    if verbose:
        print(f"Period {period} total: {wf_period.shape[0]} events "
              f"({wf_AoE_period.shape[0]} after the A/E cut)")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        npy_paths, meta_path = _checkpoint_paths(output_dir, period)
        for path, arr in zip(npy_paths, period_out):
            np.save(path, arr)
        with open(meta_path, "w") as f:
            json.dump({"chosen_detector": chosen_detector, "e_min": e_min, "e_max": e_max,
                       "fields": _CHECKPOINT_KEYS}, f)
        if verbose:
            print(f"  -> saved checkpoint to {output_dir}")

    return period_out


# -----------------------------------------------------------------------------------------
# ----------------------------- Level 3: more than one period -----------------------------
# -----------------------------------------------------------------------------------------

def extract_all_periods(periods, base_rawWF_path, base_par_path, base_psp_path, chosen_detector, lmeta,
                        e_min=1550, e_max=1650, output_dir=None, force_recompute=False, verbose=True):
    """
    Extract and concatenate all the clean events of all the runs of more than one period.
    Internally recalls extract_period() for every period (inherits automatic checkpoint/resume if output_dir is specified).

    Returns (wf_all, energy_all, AoE_all, risetime_all, wf_AoE_all, energy_AoE_all, risetime_AoE_all).
    """

    all_lists = [[] for _ in _CHECKPOINT_KEYS]

    for period in periods:
        if verbose:
            print(f"\n=== Period {period} ===")
        period_out = extract_period(
            period, base_rawWF_path, base_par_path, base_psp_path, chosen_detector, lmeta,
            e_min=e_min, e_max=e_max, output_dir=output_dir,
            force_recompute=force_recompute, verbose=verbose
        )
        if period_out[0].shape[0] > 0:
            for lst, arr in zip(all_lists, period_out):
                lst.append(arr)

    if all_lists[0]:
        all_out = tuple(np.concatenate(l, axis=0) for l in all_lists)
    else:
        all_out = _empty_outputs()

    if verbose:
        print(f"\nTotal across {len(periods)} periods: {all_out[0].shape[0]} events "
              f"({all_out[4].shape[0]} after the A/E cut)")

    # Single file with ONLY the rise times of all the periods (nothing else inside),
    # so it can be downloaded on its own.
    if output_dir is not None and len(periods) > 0:
        os.makedirs(output_dir, exist_ok=True)
        risetime_path = os.path.join(output_dir, f"risetime_{periods[0]}-{periods[-1]}.npz")
        np.savez(risetime_path, risetime=all_out[3], risetime_AoE=all_out[6])
        if verbose:
            print(f"  -> saved rise times ({all_out[3].shape[0]} events, "
                  f"{all_out[6].shape[0]} after A/E cut) to {risetime_path}")

    return all_out


# -----------------------------------------------------------------------------------------
# ----------------------------- Reloading from checkpoints --------------------------------
# -----------------------------------------------------------------------------------------
# These functions do NOT need lh5: they only read the .npy / .npz files saved by
# extract_period / extract_all_periods (e.g. after copying them to another machine).

_RISETIME_FILE_RE = re.compile(r"risetime_p(\d+)-p(\d+)\.npz$")


def _find_risetime_file(output_dir, risetime_file=None):
    """
    Returns the path of the single risetime_pXX-pYY.npz file.
    If `risetime_file` is given it is used as is (absolute path or name inside output_dir).
    """
    if risetime_file is not None:
        path = risetime_file if os.path.isabs(risetime_file) else os.path.join(output_dir, risetime_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rise time file not found: {path}")
        return path

    candidates = sorted(f for f in glob.glob(os.path.join(output_dir, "risetime_p*-p*.npz"))
                        if _RISETIME_FILE_RE.search(os.path.basename(f)))
    if not candidates:
        raise FileNotFoundError(
            f"No risetime_pXX-pYY.npz file in {output_dir} (and no risetime_{{period}}.npy either)."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"More than one rise time file in {output_dir}: "
            f"{[os.path.basename(c) for c in candidates]}. Choose one with risetime_file=..."
        )
    return candidates[0]


def _n_events(path):
    """Number of events in a saved .npy without loading it all in memory."""
    return np.load(path, mmap_mode="r").shape[0]


def load_risetime(periods, output_dir, risetime_file=None, verbose=True):
    """
    Returns (risetime, risetime_AoE) for the requested `periods`, concatenated in the
    same order, so they are aligned event by event with the waveforms/energies
    returned by partition_from_checkpoint for the same periods.

    For every period:
      - if risetime_{period}.npy and risetime_AoE_{period}.npy exist in output_dir, they are used;
      - otherwise the period is cut out of the single risetime_pXX-pYY.npz file.
        The position of each period inside that file is reconstructed from the number of
        events in the local energy_{period}.npy / energy_AoE_{period}.npy files, so those
        must exist for every period of the file up to the last one requested.

    If the local energy checkpoints do not match the rise time file (e.g. the checkpoints
    on this machine are older than the force_recompute on the server) it raises an error
    instead of returning misaligned rise times.
    """
    periods = list(periods)

    local = {p: (os.path.join(output_dir, f"risetime_{p}.npy"),
                 os.path.join(output_dir, f"risetime_AoE_{p}.npy")) for p in periods}
    need_npz = [p for p in periods if not all(os.path.exists(x) for x in local[p])]

    rt_parts = {}
    rt_AoE_parts = {}

    for p in periods:
        if p not in need_npz:
            rt_parts[p] = np.load(local[p][0])
            rt_AoE_parts[p] = np.load(local[p][1])

    if need_npz:
        npz_path = _find_risetime_file(output_dir, risetime_file)
        m = _RISETIME_FILE_RE.search(os.path.basename(npz_path))
        if m is None:
            raise ValueError(f"Cannot read the period range from the file name {os.path.basename(npz_path)} "
                             f"(expected risetime_pXX-pYY.npz).")
        first, last = int(m.group(1)), int(m.group(2))
        file_periods = [f"p{n:02d}" for n in range(first, last + 1)]

        outside = [p for p in need_npz if p not in file_periods]
        if outside:
            raise ValueError(f"Periods {outside} are not contained in {os.path.basename(npz_path)} "
                             f"({file_periods[0]}-{file_periods[-1]}).")

        with np.load(npz_path) as d:
            rt_file = d["risetime"]
            rt_AoE_file = d["risetime_AoE"]

        # Offsets of every period inside the file, from the local energy checkpoints
        last_needed = max(file_periods.index(p) for p in need_npz)
        offsets = {}
        start, start_AoE = 0, 0
        all_counted = True
        for i, p in enumerate(file_periods):
            e_path = os.path.join(output_dir, f"energy_{p}.npy")
            eA_path = os.path.join(output_dir, f"energy_AoE_{p}.npy")
            if not (os.path.exists(e_path) and os.path.exists(eA_path)):
                if i <= last_needed:
                    raise FileNotFoundError(
                        f"energy_{p}.npy / energy_AoE_{p}.npy missing in {output_dir}: they are needed "
                        f"to know where each period starts inside {os.path.basename(npz_path)}."
                    )
                all_counted = False
                break
            n, n_AoE = _n_events(e_path), _n_events(eA_path)
            offsets[p] = (start, start + n, start_AoE, start_AoE + n_AoE)
            start += n
            start_AoE += n_AoE

        # Consistency check between local checkpoints and the rise time file
        if all_counted:
            if start != len(rt_file) or start_AoE != len(rt_AoE_file):
                raise ValueError(
                    f"The local energy checkpoints ({start} events, {start_AoE} after A/E) do not match "
                    f"{os.path.basename(npz_path)} ({len(rt_file)} events, {len(rt_AoE_file)} after A/E).\n"
                    f"The checkpoints on this machine are probably older than the recompute on the server: "
                    f"download again the wf_/energy_/AoE_ files together with the rise time file."
                )
        elif start > len(rt_file) or start_AoE > len(rt_AoE_file):
            raise ValueError(f"The local energy checkpoints have more events than {os.path.basename(npz_path)}: "
                             f"download them again from the server.")

        for p in need_npz:
            a, b, aA, bA = offsets[p]
            rt_parts[p] = rt_file[a:b]
            rt_AoE_parts[p] = rt_AoE_file[aA:bA]

        if verbose:
            print(f"Rise times of {need_npz} taken from {os.path.basename(npz_path)}")

    risetime = np.concatenate([rt_parts[p] for p in periods]) if periods else np.array([])
    risetime_AoE = np.concatenate([rt_AoE_parts[p] for p in periods]) if periods else np.array([])
    return risetime, risetime_AoE


def _load_base_checkpoint(period, chosen_detector, e_min, e_max, output_dir, verbose):
    """
    Loads the 5 "classic" files of one period (wf, energy, AoE, wf_AoE, energy_AoE),
    after checking that the meta file matches detector and energy window.
    """
    npy_paths, meta_path = _checkpoint_paths(output_dir, period)
    paths = dict(zip(_CHECKPOINT_KEYS, npy_paths))

    if chosen_detector is not None or e_min is not None or e_max is not None:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"missing meta_{period}.json in {output_dir} path"
            )

        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("chosen_detector") != chosen_detector
            or meta.get("e_min") != e_min
            or meta.get("e_max") != e_max):
            raise ValueError(
                f"The checkpoint for {period} has been saved for \n\tdetector {meta.get('chosen_detector')!r} "
                f"\n\te_min={meta.get('e_min')}, \n\te_max={meta.get('e_max')} "
                f"\nIt doesn't match the request (detector={chosen_detector!r}, e_min={e_min}, e_max={e_max})."
            )

    out = tuple(np.load(paths[k]) for k in ["wf", "energy", "AoE", "wf_AoE", "energy_AoE"])

    if verbose:
        print(f"Period {period}: checkpoint loaded from {output_dir}")

    return out


def period_from_checkpoint(period, chosen_detector, e_min=1550, e_max=1650, output_dir=None,
                           risetime_file=None, verbose=True):
    """
    Reloads ONE period from the checkpoints in output_dir (no lh5 needed).

    Returns (wf, energy, AoE, risetime, wf_AoE, energy_AoE, risetime_AoE),
    same order as extract_period.
    """
    wf, energy, AoE, wf_AoE, energy_AoE = _load_base_checkpoint(
        period, chosen_detector, e_min, e_max, output_dir, verbose
    )
    risetime, risetime_AoE = load_risetime([period], output_dir, risetime_file=risetime_file, verbose=verbose)

    if len(risetime) != len(energy) or len(risetime_AoE) != len(energy_AoE):
        raise ValueError(f"Period {period}: rise time and energy have different lengths "
                         f"({len(risetime)} vs {len(energy)}, {len(risetime_AoE)} vs {len(energy_AoE)} after A/E).")

    return wf, energy, AoE, risetime, wf_AoE, energy_AoE, risetime_AoE


def partition_from_checkpoint(periods, chosen_detector, e_min=1550, e_max=1650, output_dir=None,
                              risetime_file=None, verbose=True):
    """
    Reloads MORE periods from the checkpoints in output_dir (no lh5 needed) and concatenates them.

    Returns (wf_all, energy_all, AoE_all, risetime_all, wf_AoE_all, energy_AoE_all, risetime_AoE_all),
    same order as extract_all_periods.
    """
    periods = list(periods)
    all_wf, all_energy, all_AoE, all_wf_AoE, all_energy_AoE = [], [], [], [], []

    for period in periods:
        wf, energy, AoE, wf_AoE, energy_AoE = _load_base_checkpoint(
            period, chosen_detector, e_min, e_max, output_dir, verbose
        )
        if wf.shape[0] > 0:
            all_wf.append(wf)
            all_energy.append(energy)
            all_AoE.append(AoE)
            all_wf_AoE.append(wf_AoE)
            all_energy_AoE.append(energy_AoE)

    wf_all = np.concatenate(all_wf, axis=0) if all_wf else np.array([])
    energy_all = np.concatenate(all_energy, axis=0) if all_energy else np.array([])
    AoE_all = np.concatenate(all_AoE, axis=0) if all_AoE else np.array([])
    wf_AoE_all = np.concatenate(all_wf_AoE, axis=0) if all_wf_AoE else np.array([])
    energy_AoE_all = np.concatenate(all_energy_AoE, axis=0) if all_energy_AoE else np.array([])

    risetime_all, risetime_AoE_all = load_risetime(periods, output_dir, risetime_file=risetime_file, verbose=verbose)

    if len(risetime_all) != len(energy_all) or len(risetime_AoE_all) != len(energy_AoE_all):
        raise ValueError(f"Rise time and energy have different lengths "
                         f"({len(risetime_all)} vs {len(energy_all)}, "
                         f"{len(risetime_AoE_all)} vs {len(energy_AoE_all)} after A/E).")

    if verbose:
        print(f"Totale su {len(periods)} periodi: {wf_all.shape[0]} eventi "
              f"({wf_AoE_all.shape[0]} dopo il taglio A/E)")

    return wf_all, energy_all, AoE_all, risetime_all, wf_AoE_all, energy_AoE_all, risetime_AoE_all


# -----------------------------------------------------------------------------------------
# ----------------------------- 2nuBB physics data (physreading) --------------------------
# -----------------------------------------------------------------------------------------
# Reads the per-period checkpoints written by physreading.extract_all_periods (2nuBB selection
# wf_{period}{suffix}.npy, energy_{period}{suffix}.npy, AoE_{period}{suffix}.npy, meta_{period}{suffix}.json.
# suffix=""       -> files without suffix (e.g. clean_2nudata_V08682A_afterLAr: energy_p03.npy)
# suffix="_noLAr" -> files with suffix    (e.g. clean_2nudata_V08682A:          energy_p03_noLAr.npy)
# Only the files with EXACTLY that suffix are read (e.g. "energy_p03_noLAr (1).npy" is ignored).
# No lh5 needed.
 
_2NU_ENERGY_RE = re.compile(r"energy_(p\d+)\.npy$")
 
 
def load_2nu_checkpoint(output_dir, periods=None, chosen_detector="V08682A", e_min=560, e_max=2039,
                        verbose=True, suffix=""):
    """
    Loads the 2nuBB checkpoints of `periods` from output_dir and concatenates them.
    periods=None -> every period found in the folder (some periods can be missing
    in the physics data, so the list is taken from the files instead of being fixed).
 
    The meta file of every period must match chosen_detector, e_min and e_max, so that
    calibration checkpoints (same file names, different energy window) can never be read by mistake.
 
    suffix : "" for files like energy_p03.npy, "_noLAr" for files like energy_p03_noLAr.npy.

    Returns a dictionary with wf (raw charge, N x 1400), energy, AoE and periods.
    """
    if periods is None:
        energy_re = re.compile(rf"^energy_(p\d+){re.escape(suffix)}\.npy$")   # suffix-aware version of _2NU_ENERGY_RE
        periods = sorted(m.group(1) for f in glob.glob(os.path.join(output_dir, f"energy_p*{suffix}.npy"))
                         if (m := energy_re.search(os.path.basename(f))))
        if not periods:
            raise FileNotFoundError(f"No energy_pXX{suffix}.npy file in {output_dir}")
    periods = list(periods)
 
    wfs, energies, aoes = [], [], []
    for p in periods:
        paths = {k: os.path.join(output_dir, f"{k}_{p}{suffix}.npy") for k in ("wf", "energy", "AoE")}
        meta_path = os.path.join(output_dir, f"meta_{p}{suffix}.json")
 
        missing = [os.path.basename(x) for x in list(paths.values()) + [meta_path] if not os.path.exists(x)]
        if missing:
            raise FileNotFoundError(f"Period {p}: missing {missing} in {output_dir}")
 
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("chosen_detector") != chosen_detector
                or meta.get("e_min") != e_min or meta.get("e_max") != e_max):
            raise ValueError(
                f"Period {p}: the checkpoint was saved for detector {meta.get('chosen_detector')!r}, "
                f"e_min={meta.get('e_min')}, e_max={meta.get('e_max')}; requested detector={chosen_detector!r}, "
                f"e_min={e_min}, e_max={e_max}."
            )
 
        wf, energy, aoe = (np.load(paths[k]) for k in ("wf", "energy", "AoE"))
        if not (len(wf) == len(energy) == len(aoe)):
            raise ValueError(f"Period {p}: different lengths wf={len(wf)}, energy={len(energy)}, AoE={len(aoe)}")
 
        if verbose:
            print(f"Period {p}: {len(energy)} events")
        if len(energy) > 0:
            wfs.append(wf); energies.append(energy); aoes.append(aoe)
 
    wf_all = np.concatenate(wfs, axis=0) if wfs else np.array([])
    energy_all = np.concatenate(energies) if energies else np.array([])
    aoe_all = np.concatenate(aoes) if aoes else np.array([])
 
    if verbose:
        print(f"2nuBB total over {periods}: {len(energy_all)} events | wf shape: {wf_all.shape}")
 
    return dict(wf=wf_all, energy=energy_all, AoE=aoe_all, periods=periods)