"""Repo-relative paths shared by the analysis scripts.

Every path is derived from this file's location, so the scripts work
wherever the repository is cloned.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repository root
ANALYSIS = ROOT / "analysis"
RESULTS = ANALYSIS / "results"
FIGURES = ANALYSIS / "figures"
HPARAMS = ANALYSIS / "hparams.sh"
DATA = ROOT / "src" / "data" / "LastFM.txt"
OUTPUT = ROOT / "src" / "output"             # training logs and checkpoints
PREDICTIONS = ANALYSIS / "predictions"       # per-seed top-20 predictions from --do_eval runs
