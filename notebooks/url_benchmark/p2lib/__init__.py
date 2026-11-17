"""Code of the URL-benchmark walkthrough notebooks. Each module holds the steps of one notebook:

    common      features, model families, metrics, the temporal split (shared by every step)
    core        notebook 01: benchmark and temporal protocol
    transfer    notebook 02: transfer across corpora
    robustness  notebook 03: tuning, stacking, forecastability, label noise
    cnn, dedup  notebook 04: character CNN, deduplicated corpus, no-www corpora
    derived     the numbers the paper computes from the result tables
    setup       downloads and checks the data; checks compares each result with the paper's
"""
from . import setup, checks, common, core, transfer, robustness, cnn, dedup, derived  # noqa: F401
