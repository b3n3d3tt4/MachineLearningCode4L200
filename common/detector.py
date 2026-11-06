"""
Detector of the analysis and all the folders of the repository, shared by every notebook.

Usage, at the top of every notebook (BEFORE importing the other modules of the repository):

    import detector
    detector.set_detector("V08682A")      # or "V08682B"

Folders
-------
All the folders are relative to the root of the repository, except the data:
    data      -> $L200_DATA/clean_data_<DETECTOR>               (calibration checkpoints, NOT in the repository)
                 $L200_DATA/clean_2nudata_<DETECTOR>[_afterLAr]  (2nuBB checkpoints, NOT in the repository)
                 default L200_DATA = <repo>/data
    models    -> <repo>/algorithms/DTC/models/<DETECTOR>       (DTC weights, ignored by git)
                 <repo>/algorithms/VAE/results/<DETECTOR>      (VAE weights and predictions, ignored by git)
                 <repo>/algorithms/SOM/models                  (SOM weights, ignored by git)
    plots     -> <repo>/plots/<DETECTOR>/...                     (ignored by git)

Detector name on the plots
--------------------------
Importing this module also writes the name of the detector, in bold, in the title of EVERY matplotlib figure
(FigureCanvasBase.print_figure is wrapped, so it works for plt.savefig / fig.savefig and for the figures shown
in the notebook, with no change in the plotting functions):
  - figure with a main title (suptitle)        -> " - <DETECTOR>" appended to it;
  - otherwise, exactly one panel with a title  -> appended to that title;
  - otherwise (no title, or several panel titles and no main title) -> the name alone, centered above the figure.
Titles that already contain the name (e.g. f"... - {detector.BOLD_NAME}") are left as they are.
Set detector.LABEL_ENABLED = False to switch it off.
"""

import os

import matplotlib
from matplotlib.backend_bases import FigureCanvasBase

DETECTORS = ("V08682A", "V08682B")

REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("L200_DATA", os.path.join(REPO, "data"))

ALGO_DIR  = os.path.join(REPO, "algorithms")
PLOTS_ROOT = os.path.join(REPO, "plots")


def set_detector(name):
    """Sets the detector and every folder that depends on it."""
    if name not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, not {name!r}")
    g = globals()
    g['DETECTOR']   = name
    g['BOLD_NAME']  = rf"$\mathbf{{{name}}}$"     # detector name in bold, to append to the titles

    # data (not in the repository)
    g['DATA_DIR']          = os.path.join(DATA_ROOT, f"clean_data_{name}")           # calibration checkpoints
    g['DATA_2NU_DIR']      = os.path.join(DATA_ROOT, f"clean_2nudata_{name}")        # 2nuBB, before the LAr cut
    g['DATA_2NU_LAR_DIR']  = os.path.join(DATA_ROOT, f"clean_2nudata_{name}_afterLAr")  # 2nuBB, after the LAr cut

    # models (ignored by git)
    g['DTC_MODELS'] = os.path.join(ALGO_DIR, "DTC", "models", name)
    g['VAE_MODELS'] = os.path.join(ALGO_DIR, "VAE", "results", name)
    g['SOM_MODELS'] = os.path.join(ALGO_DIR, "SOM", "models")

    # plots (ignored by git)
    g['PLOTS_DIR']   = os.path.join(PLOTS_ROOT, name)
    g['DTC_PLOTS']   = os.path.join(PLOTS_ROOT, name, "DTC")
    g['VAE_PLOTS']   = os.path.join(PLOTS_ROOT, name, "VAE")
    g['SOM_PLOTS']   = os.path.join(PLOTS_ROOT, name, "SOM")
    g['DATA_PLOTS']  = os.path.join(PLOTS_ROOT, name, "data_processing")
    g['PLOTS_2NU']   = os.path.join(PLOTS_ROOT, "2nubb")       # <PLOTS_2NU>/<method>/<detector>
    return name


# default (kept when this module is reloaded): $L200_DETECTOR, otherwise V08682A
set_detector(globals().get('DETECTOR', os.environ.get("L200_DETECTOR", "V08682A")))


# ------------------------------------------------------------------------------------------
# ------------------------------- detector name on every plot ------------------------------
# ------------------------------------------------------------------------------------------
LABEL_ENABLED = globals().get('LABEL_ENABLED', True)     # False: no name added (kept when this module is reloaded)
LABEL_KWARGS  = dict(fontsize=14)                          # name alone above the figure (bold from BOLD_NAME)
LABEL_GAP_IN  = 0.06      # inches between that name and the top of the content of the figure


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig._get_renderer()


def _append(text):
    text.set_text(f"{text.get_text()} - {BOLD_NAME}")


def add_label(fig):
    """Detector name in the title of fig (called again: nothing added twice)."""
    old = getattr(fig, '_detector_label', None)
    if old is not None:
        old.remove()
        fig._detector_label = None

    suptitle = fig._suptitle
    if suptitle is not None and suptitle.get_text():
        if DETECTOR not in suptitle.get_text():
            _append(suptitle)
        return
    titles = [t for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
    if any(DETECTOR in t.get_text() for t in titles):
        return
    if len(titles) == 1:
        _append(titles[0])
        return

    # no main title: the name alone, centered above everything drawn in the figure
    bbox = fig.get_tightbbox(_renderer(fig))                  # inches
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig._detector_label = fig.text((bbox.x0 + bbox.x1) / 2 / w, (bbox.y1 + LABEL_GAP_IN) / h, BOLD_NAME,
                                   ha='center', va='bottom', **LABEL_KWARGS)


def _print_figure_with_label(self, *args, **kwargs):
    if LABEL_ENABLED:
        add_label(self.figure)
    return FigureCanvasBase._print_figure_without_label(self, *args, **kwargs)


# the original method is kept on the class: wrapped only once, also if this module is reloaded
if not hasattr(FigureCanvasBase, '_print_figure_without_label'):
    FigureCanvasBase._print_figure_without_label = FigureCanvasBase.print_figure
FigureCanvasBase.print_figure = _print_figure_with_label

matplotlib.rcParams['savefig.bbox'] = 'tight'     # the centered name is above the original figure area
