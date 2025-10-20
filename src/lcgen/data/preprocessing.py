"""Data preprocessing functions for power spectra, ACF, and light curves"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
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
    standardize: bool = True,
    remove_outliers: bool = True,
    outlier_sigma: float = 5.0,
    use_robust: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict]]:
    """
    Preprocess a light curve before spectral analysis.

    Performs standard preprocessing steps:
    - Remove NaN values
    - Remove outliers (optional)
    - Standardize flux (optional, using median/IQR for robustness)
    - Sort by time

    Parameters
    ----------
    time : np.ndarray
        Time values
    flux : np.ndarray
        Flux values
    flux_err : np.ndarray, optional
        Flux uncertainties
    standardize : bool, default=True
        Standardize flux using robust statistics (median/IQR)
    remove_outliers : bool, default=True
        Remove outliers beyond outlier_sigma standard deviations (or IQR if robust)
    outlier_sigma : float, default=5.0
        Number of standard deviations (or IQR multiples) for outlier rejection
    use_robust : bool, default=True
        Use robust statistics (median/IQR) instead of mean/std

    Returns
    -------
    time_clean : np.ndarray
        Cleaned time array
    flux_clean : np.ndarray
        Cleaned and standardized flux array
    flux_err_clean : np.ndarray or None
        Cleaned flux error array (if provided)
    params : dict or None
        Standardization parameters if standardize=True:
        {'median': float, 'iqr': float} or {'mean': float, 'std': float}
        None if standardize=False
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
        if use_robust:
            # Use median absolute deviation (MAD) for robust outlier detection
            median = np.median(flux_clean)
            q75, q25 = np.percentile(flux_clean, [75, 25])
            iqr = q75 - q25
            outlier_mask = np.abs(flux_clean - median) < outlier_sigma * iqr
        else:
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

    # Standardize
    params = None
    if standardize:
        if use_robust:
            # Robust standardization using median and IQR
            median_flux = np.median(flux_clean)
            q75, q25 = np.percentile(flux_clean, [75, 25])
            iqr_flux = q75 - q25
            if iqr_flux == 0:
                iqr_flux = 1.0  # Avoid division by zero
            flux_clean = (flux_clean - median_flux) / iqr_flux
            params = {'median': median_flux, 'iqr': iqr_flux}
        else:
            # Standard standardization using mean and std
            mean_flux = np.mean(flux_clean)
            std_flux = np.std(flux_clean)
            if std_flux == 0:
                std_flux = 1.0
            flux_clean = (flux_clean - mean_flux) / std_flux
            params = {'mean': mean_flux, 'std': std_flux}

    return time_clean, flux_clean, flux_err_clean, params


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


def preprocess_lightcurve_multimodal(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    # Spectral analysis parameters
    NW: float = 4.0,
    K: Optional[int] = None,
    n_bins: int = 1024,
    freq_grid: Optional[np.ndarray] = None,
    freq_spacing: str = 'log',
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    # Normalization options
    compute_pvalues: bool = True,
    acf_mode: str = 'correlation',
    # Preprocessing options
    remove_outliers: bool = True,
    outlier_sigma: float = 5.0,
    # Feature extraction
    include_lc_features: bool = True,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline: raw light curve → 5-channel multi-modal input.

    This function performs the full workflow from raw irregularly-sampled light curves
    to normalized multi-modal features ready for input to the encoder network.

    Workflow:
    1. Clean data (NaN removal, outlier rejection, time sorting)
    2. Standardize using median/IQR (robust to outliers)
    3. Compute multitaper PSD and F-statistic via tapify
    4. Compute p-values from F-statistic
    5. Derive ACF from PSD via Wiener-Khinchin theorem
    6. Resample PSD/F-stat/ACF to pre-specified 1024-bin grid
    7. Extract auxiliary features (integral, peak, variance, timescale)
    8. Normalize spectral features (arcsinh for PSD/ACF, z-score for F-stat)
    9. Prepare time series input (standardized flux, scaled errors)
    10. Assemble tabular auxiliary features

    Parameters
    ----------
    time : np.ndarray
        Time array (irregular sampling allowed), shape (N,)
    flux : np.ndarray
        Flux measurements, shape (N,)
    flux_err : np.ndarray
        Flux uncertainties, shape (N,)
    NW : float, default=4.0
        Time-bandwidth parameter for multitaper analysis
    K : int, optional
        Number of tapers. If None, defaults to int(2*NW - 1)
    n_bins : int, default=1024
        Number of bins for output grids (PSD, ACF, F-stat)
    freq_grid : np.ndarray, optional
        Pre-specified frequency grid. If None, computed automatically
    freq_spacing : str, default='log'
        Frequency grid spacing: 'log' or 'linear'.
        Log spacing provides better resolution at low frequencies
    freq_min, freq_max : float, optional
        Frequency bounds. If None, computed from light curve
    compute_pvalues : bool, default=True
        Compute p-values from F-statistics
    acf_mode : str, default='correlation'
        ACF mode: 'correlation' (normalized) or 'covariance' (preserves amplitude)
    remove_outliers : bool, default=True
        Remove outliers during preprocessing
    outlier_sigma : float, default=5.0
        Outlier rejection threshold (in units of IQR)
    include_lc_features : bool, default=True
        Include light curve characteristics in tabular features

    Returns
    -------
    result : dict
        Dictionary containing:

        **5 Input Channels**:
        - 'acf' : (n_bins,) - Autocorrelation function (arcsinh normalized)
        - 'psd' : (n_bins,) - Power spectral density (arcsinh normalized)
        - 'fstat' : (n_bins,) - F-statistic (z-score normalized)
        - 'timeseries' : (seq_len, 2) - [flux_standardized, flux_err_scaled]
        - 'time' : (seq_len,) - Time array for positional encoding
        - 'tabular' : (n_features,) - Auxiliary features

        **Additional Outputs**:
        - 'pvalues' : (n_bins,) - p-values from F-test (if compute_pvalues=True)
        - 'frequency' : (n_bins,) - Frequency grid used

        **Metadata** (for reconstruction):
        - 'metadata' : dict with keys:
            - 'median_flux' : float - Position parameter
            - 'iqr_flux' : float - Scale parameter
            - 'psd_integral' : float - Total power
            - 'psd_peak' : float - Peak power
            - 'acf_variance' : float - Variance of standardized flux
            - 'acf_timescale' : float - Characteristic timescale
            - 'lc_duration' : float - Observation duration
            - 'n_observations' : int - Number of data points
            - 'median_cadence' : float - Median time between observations
            - 'NW', 'K' : Multitaper parameters

    Examples
    --------
    >>> from lcgen.data import preprocess_lightcurve_multimodal
    >>>
    >>> # Preprocess a single light curve
    >>> result = preprocess_lightcurve_multimodal(time, flux, flux_err)
    >>>
    >>> # Extract 5 channels for model input
    >>> acf_input = result['acf']
    >>> psd_input = result['psd']
    >>> fstat_input = result['fstat']
    >>> timeseries_input = result['timeseries']
    >>> tabular_input = result['tabular']
    >>>
    >>> # Check for significant periods
    >>> sig_peaks = result['pvalues'] < 0.01
    >>> print(f"Significant frequencies: {result['frequency'][sig_peaks]}")
    """
    # Import spectral analysis functions (avoid circular imports)
    from .spectral import (
        compute_multitaper_psd,
        compute_frequency_grid,
        fstatistic_to_pvalue,
        psd_to_acf,
        resample_to_uniform_grid,
        extract_acf_timescale,
        resample_acf_to_bins,
    )

    # Step 1: Clean and standardize
    time_clean, flux_std, flux_err_clean, std_params = preprocess_lightcurve_for_spectral(
        time, flux, flux_err,
        standardize=True,
        remove_outliers=remove_outliers,
        outlier_sigma=outlier_sigma,
        use_robust=True  # Use median/IQR
    )

    # Extract standardization parameters
    median_flux = std_params['median']
    iqr_flux = std_params['iqr']

    # Step 2: Compute frequency grid (if not provided)
    if freq_grid is None:
        freq_grid = compute_frequency_grid(
            time_clean,
            n_bins=n_bins,
            spacing=freq_spacing,
            freq_min=freq_min,
            freq_max=freq_max
        )

    # Step 3: Compute multitaper PSD and F-statistic
    psd_result = compute_multitaper_psd(
        time_clean,
        flux_std,  # Use standardized flux
        flux_err=None,  # Not used by tapify
        NW=NW,
        K=K,
        freq=freq_grid,  # Use pre-specified grid
        return_fstat=True
    )

    psd_raw = psd_result['psd']
    fstat_raw = psd_result['fstat']
    frequency = psd_result['frequency']
    K_used = psd_result['K']

    # Step 4: Compute p-values from F-statistic
    pvalues = None
    if compute_pvalues:
        pvalues = fstatistic_to_pvalue(fstat_raw, K_used)

    # Step 5: Derive ACF from PSD via Wiener-Khinchin
    lags, acf_raw = psd_to_acf(
        frequency,
        psd_raw,
        normalize=(acf_mode == 'correlation')
    )

    # Step 6: Resample to uniform 1024-bin grids
    # PSD and F-stat are already on freq_grid if we passed it
    # But ensure they're exactly n_bins length
    if len(frequency) != n_bins:
        freq_grid_final, psd_1024 = resample_to_uniform_grid(
            frequency, psd_raw, n_bins=n_bins, method=freq_spacing
        )
        _, fstat_1024 = resample_to_uniform_grid(
            frequency, fstat_raw, n_bins=n_bins, method=freq_spacing
        )
    else:
        freq_grid_final = frequency
        psd_1024 = psd_raw
        fstat_1024 = fstat_raw

    # Resample ACF
    acf_1024 = resample_acf_to_bins(lags, acf_raw, n_bins=n_bins)

    # Step 7: Extract auxiliary features
    # PSD features
    psd_integral = np.trapz(psd_1024, freq_grid_final)
    psd_peak = np.max(psd_1024)

    # ACF features
    if acf_mode == 'correlation':
        # Variance of standardized flux
        # Since we standardized with IQR, variance != 1
        acf_variance = np.var(flux_std)
    else:
        # ACF(0) gives variance for covariance mode
        acf_variance = acf_1024[0] if len(acf_1024) > 0 else 1.0

    acf_timescale = extract_acf_timescale(lags, acf_raw, method='efolding')
    if np.isnan(acf_timescale):
        acf_timescale = 0.0  # Default if can't be computed

    # Light curve characteristics
    lc_duration = time_clean[-1] - time_clean[0]
    n_observations = len(time_clean)
    cadences = np.diff(time_clean)
    median_cadence = np.median(cadences) if len(cadences) > 0 else 1.0

    # Step 8: Normalize spectral features
    # Arcsinh normalization for PSD and ACF
    psd_normalized = normalize_ps(psd_1024, scale_factor=50.0)
    acf_normalized = normalize_acf(acf_1024, scale_factor=0.001)

    # Z-score normalization for F-statistic
    fstat_mean = np.mean(fstat_1024)
    fstat_std = np.std(fstat_1024)
    fstat_normalized = (fstat_1024 - fstat_mean) / (fstat_std + 1e-8)

    # Step 9: Prepare time series input
    # Standardized flux + scaled uncertainties
    flux_err_scaled = flux_err_clean / iqr_flux
    timeseries = np.stack([flux_std, flux_err_scaled], axis=-1)  # Shape: (seq_len, 2)

    # Step 10: Assemble tabular features
    tabular_features = [
        median_flux,
        iqr_flux,
        psd_integral,
        psd_peak,
        acf_variance,
        acf_timescale,
    ]

    if include_lc_features:
        tabular_features.extend([
            lc_duration,
            float(n_observations),
            median_cadence,
        ])

    tabular = np.array(tabular_features, dtype=np.float32)

    # Assemble output dictionary
    result = {
        # 5 input channels
        'acf': acf_normalized.astype(np.float32),
        'psd': psd_normalized.astype(np.float32),
        'fstat': fstat_normalized.astype(np.float32),
        'timeseries': timeseries.astype(np.float32),
        'time': time_clean.astype(np.float32),
        'tabular': tabular,

        # Additional outputs
        'frequency': freq_grid_final.astype(np.float32),

        # Metadata for reconstruction/denormalization
        'metadata': {
            'median_flux': float(median_flux),
            'iqr_flux': float(iqr_flux),
            'psd_integral': float(psd_integral),
            'psd_peak': float(psd_peak),
            'acf_variance': float(acf_variance),
            'acf_timescale': float(acf_timescale),
            'lc_duration': float(lc_duration),
            'n_observations': int(n_observations),
            'median_cadence': float(median_cadence),
            'NW': float(NW),
            'K': int(K_used),
            'freq_spacing': freq_spacing,
            'acf_mode': acf_mode,
        }
    }

    # Add p-values if computed
    if pvalues is not None:
        if len(pvalues) != n_bins:
            # Resample p-values to match grid
            _, pvalues_resampled = resample_to_uniform_grid(
                frequency, pvalues, n_bins=n_bins, method=freq_spacing
            )
            result['pvalues'] = pvalues_resampled.astype(np.float32)
        else:
            result['pvalues'] = pvalues.astype(np.float32)

    return result


def batch_preprocess_multimodal(
    times: list,
    fluxes: list,
    flux_errs: list,
    verbose: bool = True,
    **kwargs  # Pass through to preprocess_lightcurve_multimodal
) -> Dict[str, np.ndarray]:
    """
    Batch process multiple light curves into multi-modal features.

    Parameters
    ----------
    times : list of np.ndarray
        List of time arrays, one per light curve
    fluxes : list of np.ndarray
        List of flux arrays, one per light curve
    flux_errs : list of np.ndarray
        List of flux error arrays, one per light curve
    verbose : bool, default=True
        Print progress information
    **kwargs
        Additional arguments passed to preprocess_lightcurve_multimodal()

    Returns
    -------
    batch_result : dict
        Dictionary containing batched arrays:
        - 'acf' : (n_samples, n_bins)
        - 'psd' : (n_samples, n_bins)
        - 'fstat' : (n_samples, n_bins)
        - 'pvalues' : (n_samples, n_bins) if compute_pvalues=True
        - 'timeseries' : list of (seq_len, 2) arrays (variable length)
        - 'time' : list of (seq_len,) arrays (variable length)
        - 'tabular' : (n_samples, n_features)
        - 'frequency' : (n_bins,) - common frequency grid
        - 'metadata' : list of metadata dicts

    Examples
    --------
    >>> from lcgen.data import batch_preprocess_multimodal
    >>>
    >>> # Batch process multiple light curves
    >>> results = batch_preprocess_multimodal(
    ...     times=[t1, t2, t3],
    ...     fluxes=[f1, f2, f3],
    ...     flux_errs=[e1, e2, e3],
    ...     NW=4.0,
    ...     verbose=True
    ... )
    >>>
    >>> # Extract batched arrays
    >>> psd_batch = results['psd']  # Shape: (3, 1024)
    >>> tabular_batch = results['tabular']  # Shape: (3, 9)
    """
    n_samples = len(times)

    if verbose:
        print(f"Batch preprocessing {n_samples} light curves...")
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="Processing")
        except ImportError:
            iterator = range(n_samples)
    else:
        iterator = range(n_samples)

    # Process first light curve to determine dimensions
    first_result = preprocess_lightcurve_multimodal(
        times[0], fluxes[0], flux_errs[0], **kwargs
    )

    n_bins = len(first_result['acf'])
    n_tabular = len(first_result['tabular'])
    compute_pvalues = 'pvalues' in first_result

    # Initialize output arrays
    acf_batch = np.zeros((n_samples, n_bins), dtype=np.float32)
    psd_batch = np.zeros((n_samples, n_bins), dtype=np.float32)
    fstat_batch = np.zeros((n_samples, n_bins), dtype=np.float32)
    tabular_batch = np.zeros((n_samples, n_tabular), dtype=np.float32)

    if compute_pvalues:
        pvalues_batch = np.zeros((n_samples, n_bins), dtype=np.float32)

    # Variable-length arrays (stored as lists)
    timeseries_list = []
    time_list = []
    metadata_list = []

    # Store first result
    acf_batch[0] = first_result['acf']
    psd_batch[0] = first_result['psd']
    fstat_batch[0] = first_result['fstat']
    tabular_batch[0] = first_result['tabular']
    timeseries_list.append(first_result['timeseries'])
    time_list.append(first_result['time'])
    metadata_list.append(first_result['metadata'])

    if compute_pvalues:
        pvalues_batch[0] = first_result['pvalues']

    common_freq = first_result['frequency']

    # Process remaining light curves
    for i in range(1, n_samples):

        try:
            result = preprocess_lightcurve_multimodal(
                times[i], fluxes[i], flux_errs[i], **kwargs
            )

            acf_batch[i] = result['acf']
            psd_batch[i] = result['psd']
            fstat_batch[i] = result['fstat']
            tabular_batch[i] = result['tabular']
            timeseries_list.append(result['timeseries'])
            time_list.append(result['time'])
            metadata_list.append(result['metadata'])

            if compute_pvalues:
                pvalues_batch[i] = result['pvalues']

        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to process light curve {i}: {e}")
            # Fill with NaN
            acf_batch[i] = np.nan
            psd_batch[i] = np.nan
            fstat_batch[i] = np.nan
            tabular_batch[i] = np.nan
            if compute_pvalues:
                pvalues_batch[i] = np.nan
            timeseries_list.append(np.array([[np.nan, np.nan]]))
            time_list.append(np.array([np.nan]))
            metadata_list.append({})

    # Assemble batch output
    batch_result = {
        'acf': acf_batch,
        'psd': psd_batch,
        'fstat': fstat_batch,
        'timeseries': timeseries_list,  # List of variable-length arrays
        'time': time_list,  # List of variable-length arrays
        'tabular': tabular_batch,
        'frequency': common_freq,
        'metadata': metadata_list,
    }

    if compute_pvalues:
        batch_result['pvalues'] = pvalues_batch

    if verbose:
        print(f"Batch preprocessing complete: {n_samples} light curves processed")

    return batch_result
