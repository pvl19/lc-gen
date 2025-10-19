"""Data preprocessing functions for power spectra, ACF, and light curves"""

import numpy as np
from typing import Optional, Dict, Tuple
import warnings


def normalize_ps(data, scale_factor=50):
    """
    Normalize power spectrum using arcsinh transformation.

    Args:
        data: Power spectrum array
        scale_factor: Scaling factor for arcsinh (default: 50)

    Returns:
        Normalized power spectrum (z-scored after arcsinh transform)
    """
    asinh = np.arcsinh(np.log10(data) / scale_factor)
    scaled = (asinh - np.mean(asinh)) / np.std(asinh)
    return scaled


def normalize_acf(data, scale_factor=0.001):
    """
    Normalize autocorrelation function using arcsinh transformation.

    Args:
        data: ACF array
        scale_factor: Scaling factor for arcsinh (default: 0.001)

    Returns:
        Normalized ACF (z-scored after arcsinh transform)
    """
    asinh = np.arcsinh(data / scale_factor)
    scaled = (asinh - np.mean(asinh)) / np.std(asinh)
    return scaled


def denormalize_ps(normalized_data, original_data, scale_factor=50):
    """
    Reverse the power spectrum normalization.

    Args:
        normalized_data: Normalized power spectrum
        original_data: Original power spectrum (for statistics)
        scale_factor: Same scale_factor used in normalization

    Returns:
        Denormalized power spectrum
    """
    # Compute statistics from original data
    asinh_orig = np.arcsinh(np.log10(original_data) / scale_factor)
    mean_orig = np.mean(asinh_orig)
    std_orig = np.std(asinh_orig)

    # Reverse z-score
    asinh_reconstructed = normalized_data * std_orig + mean_orig

    # Reverse arcsinh
    log10_ps = np.sinh(asinh_reconstructed) * scale_factor
    ps = 10 ** log10_ps

    return ps


def denormalize_acf(normalized_data, original_data, scale_factor=0.001):
    """
    Reverse the ACF normalization.

    Args:
        normalized_data: Normalized ACF
        original_data: Original ACF (for statistics)
        scale_factor: Same scale_factor used in normalization

    Returns:
        Denormalized ACF
    """
    # Compute statistics from original data
    asinh_orig = np.arcsinh(original_data / scale_factor)
    mean_orig = np.mean(asinh_orig)
    std_orig = np.std(asinh_orig)

    # Reverse z-score
    asinh_reconstructed = normalized_data * std_orig + mean_orig

    # Reverse arcsinh
    acf = np.sinh(asinh_reconstructed) * scale_factor

    return acf


def preprocess_lightcurve_for_spectral(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    remove_mean: bool = True,
    remove_outliers: bool = True,
    outlier_sigma: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess a light curve before spectral analysis.

    Performs standard preprocessing steps:
    - Remove NaN values
    - Remove outliers (optional)
    - Remove mean (optional)
    - Sort by time

    Parameters
    ----------
    time : np.ndarray
        Time values
    flux : np.ndarray
        Flux values
    flux_err : np.ndarray, optional
        Flux uncertainties
    remove_mean : bool, default=True
        Subtract mean flux
    remove_outliers : bool, default=True
        Remove outliers beyond outlier_sigma standard deviations
    outlier_sigma : float, default=5.0
        Number of standard deviations for outlier rejection

    Returns
    -------
    time_clean : np.ndarray
        Cleaned time array
    flux_clean : np.ndarray
        Cleaned flux array
    flux_err_clean : np.ndarray or None
        Cleaned flux error array (if provided)
    """
    # Remove NaN values
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        valid_mask &= np.isfinite(flux_err)

    time_clean = time[valid_mask]
    flux_clean = flux[valid_mask]
    flux_err_clean = flux_err[valid_mask] if flux_err is not None else None

    # Remove outliers
    if remove_outliers and len(flux_clean) > 10:
        median = np.median(flux_clean)
        std = np.std(flux_clean)
        outlier_mask = np.abs(flux_clean - median) < outlier_sigma * std

        time_clean = time_clean[outlier_mask]
        flux_clean = flux_clean[outlier_mask]
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean[outlier_mask]

    # Sort by time
    sort_idx = np.argsort(time_clean)
    time_clean = time_clean[sort_idx]
    flux_clean = flux_clean[sort_idx]
    if flux_err_clean is not None:
        flux_err_clean = flux_err_clean[sort_idx]

    # Remove mean
    if remove_mean:
        flux_clean = flux_clean - np.mean(flux_clean)

    return time_clean, flux_clean, flux_err_clean


def prepare_spectra_for_model(
    psd: Optional[np.ndarray] = None,
    acf: Optional[np.ndarray] = None,
    fstat: Optional[np.ndarray] = None,
    normalize_psd: bool = True,
    normalize_acf: bool = True,
    psd_scale_factor: float = 50.0,
    acf_scale_factor: float = 0.001
) -> Dict[str, np.ndarray]:
    """
    Prepare spectral features (PSD, ACF, F-stat) for model input.

    Applies normalization and ensures proper format for neural network input.

    Parameters
    ----------
    psd : np.ndarray, optional
        Power spectral density array, shape (n_samples, n_bins) or (n_bins,)
    acf : np.ndarray, optional
        Autocorrelation function array, shape (n_samples, n_bins) or (n_bins,)
    fstat : np.ndarray, optional
        F-statistic array, shape (n_samples, n_bins) or (n_bins,)
    normalize_psd : bool, default=True
        Apply arcsinh normalization to PSD
    normalize_acf : bool, default=True
        Apply arcsinh normalization to ACF
    psd_scale_factor : float, default=50.0
        Scale factor for PSD normalization
    acf_scale_factor : float, default=0.001
        Scale factor for ACF normalization

    Returns
    -------
    features : dict
        Dictionary with keys 'psd', 'acf', 'fstat' containing normalized features
        (only includes keys for non-None inputs)

    Examples
    --------
    >>> from lcgen.data import compute_multitaper_psd, prepare_spectra_for_model
    >>>
    >>> # Compute PSD from light curve
    >>> result = compute_multitaper_psd(time, flux, return_fstat=True)
    >>>
    >>> # Prepare for model
    >>> features = prepare_spectra_for_model(
    ...     psd=result['psd'],
    ...     fstat=result['fstat']
    ... )
    >>> model_input = features['psd']
    """
    features = {}

    if psd is not None:
        if normalize_psd:
            # Handle both single and batch inputs
            if psd.ndim == 1:
                features['psd'] = normalize_ps(psd, scale_factor=psd_scale_factor)
            else:
                features['psd'] = np.array([
                    normalize_ps(psd[i], scale_factor=psd_scale_factor)
                    for i in range(len(psd))
                ])
        else:
            features['psd'] = psd

    if acf is not None:
        if normalize_acf:
            if acf.ndim == 1:
                features['acf'] = normalize_acf(acf, scale_factor=acf_scale_factor)
            else:
                features['acf'] = np.array([
                    normalize_acf(acf[i], scale_factor=acf_scale_factor)
                    for i in range(len(acf))
                ])
        else:
            features['acf'] = acf

    if fstat is not None:
        # F-statistic typically doesn't need normalization (already in reasonable range)
        # but we can standardize it
        if fstat.ndim == 1:
            features['fstat'] = (fstat - np.mean(fstat)) / (np.std(fstat) + 1e-8)
        else:
            features['fstat'] = np.array([
                (fstat[i] - np.mean(fstat[i])) / (np.std(fstat[i]) + 1e-8)
                for i in range(len(fstat))
            ])

    return features
