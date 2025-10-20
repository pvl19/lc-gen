"""Data processing utilities"""

from .preprocessing import (
    normalize_ps,
    normalize_acf,
    denormalize_ps,
    denormalize_acf,
    preprocess_lightcurve_for_spectral,
    prepare_spectra_for_model,
    preprocess_lightcurve_multimodal,
    batch_preprocess_multimodal,
)
from .masking import block_predefined_mask, block_predefined_mask_lc, block_mask
from .datasets import FluxDataset, PowerSpectrumDataset
from .spectral import (
    compute_multitaper_psd,
    compute_fstatistic,
    compute_frequency_grid,
    fstatistic_to_pvalue,
    psd_to_acf,
    extract_acf_timescale,
    resample_to_uniform_grid,
    resample_acf_to_bins,
    batch_compute_spectra,
)

__all__ = [
    # Normalization
    "normalize_ps",
    "normalize_acf",
    "denormalize_ps",
    "denormalize_acf",
    # Preprocessing
    "preprocess_lightcurve_for_spectral",
    "prepare_spectra_for_model",
    "preprocess_lightcurve_multimodal",
    "batch_preprocess_multimodal",
    # Masking
    "block_predefined_mask",
    "block_predefined_mask_lc",
    "block_mask",
    # Datasets
    "FluxDataset",
    "PowerSpectrumDataset",
    # Spectral analysis
    "compute_multitaper_psd",
    "compute_fstatistic",
    "compute_frequency_grid",
    "fstatistic_to_pvalue",
    "psd_to_acf",
    "extract_acf_timescale",
    "resample_to_uniform_grid",
    "resample_acf_to_bins",
    "batch_compute_spectra",
]
