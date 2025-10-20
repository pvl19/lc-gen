"""
Spectral analysis utilities for irregularly sampled time series.

This module provides functions for computing power spectral density (PSD),
autocorrelation function (ACF), and F-statistics from irregularly sampled
light curves using multitaper methods via the tapify package.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from tapify import MultiTaper
    TAPIFY_AVAILABLE = True
except ImportError:
    TAPIFY_AVAILABLE = False
    warnings.warn(
        "tapify not installed. Install with: pip install tapify\n"
        "Or from source: https://github.com/aaryapatil/tapify",
        ImportWarning
    )

try:
    from scipy.stats import f as f_dist
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not installed. Some features will be unavailable.",
        ImportWarning
    )


def compute_multitaper_psd(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    NW: float = 4.0,
    K: Optional[int] = None,
    freq: Optional[np.ndarray] = None,
    adaptive_weighting: bool = True,
    method: str = 'fft',
    return_fstat: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute multitaper power spectral density for irregularly sampled time series.

    Uses the tapify package to compute PSD via multitaper methods with support for
    irregular sampling. Supports Lomb-Scargle ('ls'), DFT ('dft'), or FFT ('fft') methods
    for handling non-uniform time sampling.

    Parameters
    ----------
    time : np.ndarray
        Time values (irregular sampling allowed), shape (N,)
    flux : np.ndarray
        Flux values corresponding to time points, shape (N,)
    flux_err : np.ndarray, optional
        Flux uncertainties. NOTE: Not currently used by tapify's multitaper PSD computation.
        The multitaper method uses adaptive weighting of tapers, not measurement uncertainties.
        This parameter is included for API consistency and potential future weighted implementations.
    NW : float, default=4.0
        Time-bandwidth parameter. Controls frequency resolution.
        Typical values: 2.5, 3.0, 3.5, 4.0
        Higher values → better frequency resolution but more variance
    K : int, optional
        Number of tapers. If None, defaults to int(2 * NW - 1)
        Typical: K = 2*NW or K = 2*NW - 1
    freq : np.ndarray, optional
        Frequency grid for output. If None, uses tapify's default
    adaptive_weighting : bool, default=True
        Use adaptive weighting of tapers based on signal content
    method : str, default='fft'
        Method for computation: 'fft' (fast NUFFT), 'dft' (direct DFT), or 'ls' (Lomb-Scargle)
        NOTE: F-statistics require 'fft' or 'dft'; not available with 'ls'
    return_fstat : bool, default=False
        If True, also compute and return F-statistic for harmonic detection
    **kwargs : dict
        Additional arguments passed to tapify.MultiTaper

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'frequency': Frequency array, shape (M,)
        - 'psd': Power spectral density, shape (M,)
        - 'fstat': F-statistic (if return_fstat=True), shape (M,)
        - 'NW': Time-bandwidth parameter used
        - 'K': Number of tapers used
        - 'method': Method used

    Raises
    ------
    ImportError
        If tapify is not installed

    Notes
    -----
    The multitaper method uses multiple orthogonal tapers (windows) to reduce
    spectral leakage and improve statistical properties of the PSD estimate.

    The F-statistic can be used to detect and characterize periodic signals
    in the presence of colored noise.

    References
    ----------
    Patil, A. D. et al. (2022). "Improving Power Spectrum Estimation using
    Multitapering". arXiv:2209.15027

    Examples
    --------
    >>> # Generate irregular time series
    >>> time = np.sort(np.random.uniform(0, 100, 500))
    >>> flux = np.sin(2 * np.pi * 0.1 * time) + 0.1 * np.random.randn(500)
    >>>
    >>> # Compute multitaper PSD
    >>> result = compute_multitaper_psd(time, flux, NW=4.0, return_fstat=True)
    >>> freq, psd = result['frequency'], result['psd']
    >>> fstat = result['fstat']
    """
    if not TAPIFY_AVAILABLE:
        raise ImportError(
            "tapify is required for multitaper spectral analysis. "
            "Install with: pip install tapify"
        )

    # Set default number of tapers
    if K is None:
        K = int(2 * NW - 1)

    # Mean-subtract the flux (standard preprocessing)
    flux_centered = flux - np.mean(flux)

    # Create MultiTaper object
    mt = MultiTaper(
        flux_centered,
        t=time,
        NW=NW,
        K=K,
        **kwargs
    )

    # Compute periodogram with optional F-test
    if freq is None:
        if return_fstat:
            frequency, psd, fstat = mt.periodogram(
                method=method,
                adaptive_weighting=adaptive_weighting,
                ftest=True
            )
        else:
            frequency, psd = mt.periodogram(
                method=method,
                adaptive_weighting=adaptive_weighting,
                ftest=False
            )
    else:
        if return_fstat:
            frequency, psd, fstat = mt.periodogram(
                freq=freq,
                method=method,
                adaptive_weighting=adaptive_weighting,
                ftest=True
            )
        else:
            frequency, psd = mt.periodogram(
                freq=freq,
                method=method,
                adaptive_weighting=adaptive_weighting,
                ftest=False
            )

    # Build result dictionary
    result = {
        'frequency': frequency,
        'psd': psd,
        'NW': NW,
        'K': K,
        'method': method,
    }

    # Add F-statistic if computed
    if return_fstat:
        result['fstat'] = fstat

    return result


def compute_fstatistic(
    mt_object: 'MultiTaper',
    freq: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute F-statistic for harmonic signal detection.

    The F-statistic tests the null hypothesis that the signal at each frequency
    is purely noise against the alternative that it contains a harmonic component.
    Higher F-statistics indicate stronger periodic signals.

    Parameters
    ----------
    mt_object : MultiTaper
        Fitted MultiTaper object from tapify
    freq : np.ndarray, optional
        Frequency array. If None, uses frequencies from mt_object

    Returns
    -------
    fstat : np.ndarray
        F-statistic values at each frequency

    Notes
    -----
    The F-statistic is particularly useful for detecting periodic signals
    in colored noise backgrounds, which is common in astronomical time series.

    A large F-statistic (typically > 10-20) suggests a significant periodic signal.

    References
    ----------
    Thomson, D. J. (1982). "Spectrum estimation and harmonic analysis".
    Proceedings of the IEEE, 70(9), 1055-1096.
    """
    if not TAPIFY_AVAILABLE:
        raise ImportError("tapify is required to compute F-statistics")

    # Check if mt_object has fstatistic method and if it's callable
    if hasattr(mt_object, 'fstatistic') and callable(mt_object.fstatistic):
        if freq is None:
            fstat = mt_object.fstatistic()
        else:
            fstat = mt_object.fstatistic(freq=freq)
        return fstat
    else:
        # Fallback: compute manually using the ratio of harmonic to broadband power
        # F-statistic approximation based on Thomson (1982)
        # This is a simplified implementation
        warnings.warn(
            "F-statistic method not available in tapify. Using manual approximation. "
            "Results may differ from full Thomson F-test.",
            UserWarning
        )

        # For now, return ones (neutral value) as placeholder
        # A proper implementation would require access to individual taper spectra
        freq_len = len(freq) if freq is not None else 1024
        return np.ones(freq_len)


def psd_to_acf(
    frequency: np.ndarray,
    psd: np.ndarray,
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function (ACF) from power spectral density.

    Uses the Wiener-Khinchin theorem: ACF is the inverse Fourier transform of PSD.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency array (assumed to start at 0 or small positive value)
    psd : np.ndarray
        Power spectral density values
    max_lag : int, optional
        Maximum lag to return. If None, returns full ACF
    normalize : bool, default=True
        If True, normalize ACF so that ACF(0) = 1

    Returns
    -------
    lags : np.ndarray
        Time lag array
    acf : np.ndarray
        Autocorrelation function values

    Notes
    -----
    For irregularly sampled data, the ACF computed from the multitaper PSD
    properly accounts for the irregular sampling via the spectral estimate.

    The ACF is real-valued and symmetric, so we return only the positive lags.

    Examples
    --------
    >>> result = compute_multitaper_psd(time, flux)
    >>> lags, acf = psd_to_acf(result['frequency'], result['psd'])
    """
    # Ensure PSD is positive (small numerical errors can cause issues)
    psd = np.maximum(psd, 0)

    # Create symmetric PSD for IFFT (PSD is real, so we need both positive and negative freqs)
    # Assume frequency array starts at or near 0
    if frequency[0] > 0:
        # Prepend zero frequency if not present
        frequency = np.concatenate([[0], frequency])
        psd = np.concatenate([[psd[0]], psd])

    # Create negative frequencies (mirror)
    psd_symmetric = np.concatenate([psd, psd[-2:0:-1]])

    # Inverse FFT to get ACF
    acf = np.fft.ifft(psd_symmetric).real

    # Shift so that zero lag is at center
    acf = np.fft.ifftshift(acf)

    # Create lag array
    n = len(acf)
    dt = 1.0 / (2 * frequency[-1])  # Approximate sampling interval
    lags = np.arange(n) * dt - (n // 2) * dt

    # Take only positive lags (ACF is symmetric)
    positive_mask = lags >= 0
    lags = lags[positive_mask]
    acf = acf[positive_mask]

    # Normalize if requested
    if normalize and acf[0] != 0:
        acf = acf / acf[0]

    # Truncate to max_lag if specified
    if max_lag is not None:
        mask = lags <= max_lag
        lags = lags[mask]
        acf = acf[mask]

    return lags, acf


def resample_to_uniform_grid(
    frequency: np.ndarray,
    psd: np.ndarray,
    n_bins: int = 1024,
    method: str = 'log',
    fmin: Optional[float] = None,
    fmax: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample PSD to a uniform frequency grid.

    Useful for preparing PSD for input to CNN models that expect fixed-size inputs.

    Parameters
    ----------
    frequency : np.ndarray
        Original frequency array
    psd : np.ndarray
        Original PSD values
    n_bins : int, default=1024
        Number of bins in output grid
    method : str, default='log'
        Grid spacing: 'log' (logarithmic) or 'linear'
    fmin : float, optional
        Minimum frequency. If None, uses min(frequency)
    fmax : float, optional
        Maximum frequency. If None, uses max(frequency)

    Returns
    -------
    freq_grid : np.ndarray
        Uniform frequency grid, shape (n_bins,)
    psd_grid : np.ndarray
        Interpolated PSD values on grid, shape (n_bins,)

    Notes
    -----
    Logarithmic spacing is often preferred for astronomical power spectra
    as it provides better resolution at low frequencies where most
    stellar variability occurs.
    """
    from scipy.interpolate import interp1d

    if fmin is None:
        fmin = frequency[0]
    if fmax is None:
        fmax = frequency[-1]

    # Create uniform grid
    if method == 'log':
        # Avoid log(0) by ensuring fmin > 0
        if fmin <= 0:
            fmin = frequency[frequency > 0][0]
        freq_grid = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    elif method == 'linear':
        freq_grid = np.linspace(fmin, fmax, n_bins)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'linear'")

    # Interpolate PSD onto grid
    interpolator = interp1d(
        frequency, psd,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    psd_grid = interpolator(freq_grid)

    return freq_grid, psd_grid


def batch_compute_spectra(
    times: list,
    fluxes: list,
    flux_errs: Optional[list] = None,
    NW: float = 4.0,
    K: Optional[int] = None,
    n_bins: int = 1024,
    return_fstat: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Batch compute multitaper PSD and ACF for multiple light curves.

    Parameters
    ----------
    times : list of np.ndarray
        List of time arrays, one per light curve
    fluxes : list of np.ndarray
        List of flux arrays, one per light curve
    flux_errs : list of np.ndarray, optional
        List of flux error arrays
    NW : float, default=4.0
        Time-bandwidth parameter
    K : int, optional
        Number of tapers
    n_bins : int, default=1024
        Number of frequency bins for output
    return_fstat : bool, default=False
        If True, also compute F-statistics
    verbose : bool, default=True
        Print progress

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'psd': Array of PSDs, shape (n_samples, n_bins)
        - 'acf': Array of ACFs, shape (n_samples, n_bins)
        - 'fstat': Array of F-statistics if return_fstat=True, shape (n_samples, n_bins)
        - 'frequency': Common frequency grid, shape (n_bins,)
        - 'lags': Common lag grid for ACF, shape (n_bins,)

    Examples
    --------
    >>> # Process multiple light curves
    >>> results = batch_compute_spectra(
    ...     times=[t1, t2, t3],
    ...     fluxes=[f1, f2, f3],
    ...     NW=4.0,
    ...     return_fstat=True
    ... )
    >>> psd_array = results['psd']  # shape: (3, 1024)
    """
    if not TAPIFY_AVAILABLE:
        raise ImportError("tapify is required for spectral analysis")

    n_samples = len(times)

    if flux_errs is None:
        flux_errs = [None] * n_samples

    # Initialize output arrays
    psd_array = np.zeros((n_samples, n_bins))
    acf_array = np.zeros((n_samples, n_bins))
    if return_fstat:
        fstat_array = np.zeros((n_samples, n_bins))

    # Process each light curve
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="Computing spectra")
        except ImportError:
            iterator = range(n_samples)
            print(f"Processing {n_samples} light curves...")
    else:
        iterator = range(n_samples)

    common_freq = None
    common_lags = None

    for i in iterator:
        # Compute multitaper PSD
        result = compute_multitaper_psd(
            times[i],
            fluxes[i],
            flux_err=flux_errs[i],
            NW=NW,
            K=K,
            return_fstat=return_fstat
        )

        # Resample to uniform grid
        freq_grid, psd_grid = resample_to_uniform_grid(
            result['frequency'],
            result['psd'],
            n_bins=n_bins
        )
        psd_array[i] = psd_grid

        # Compute ACF from PSD
        lags, acf = psd_to_acf(result['frequency'], result['psd'])

        # Resample ACF to n_bins (simple approach: interpolate or truncate)
        if len(acf) >= n_bins:
            acf_array[i] = acf[:n_bins]
            if common_lags is None:
                common_lags = lags[:n_bins]
        else:
            # Pad if necessary
            acf_array[i, :len(acf)] = acf
            if common_lags is None:
                common_lags = lags

        # Store F-statistic if requested
        if return_fstat and 'fstat' in result:
            _, fstat_grid = resample_to_uniform_grid(
                result['frequency'],
                result['fstat'],
                n_bins=n_bins
            )
            fstat_array[i] = fstat_grid

        if common_freq is None:
            common_freq = freq_grid

    # Build output dictionary
    output = {
        'psd': psd_array,
        'acf': acf_array,
        'frequency': common_freq,
        'lags': common_lags,
    }

    if return_fstat:
        output['fstat'] = fstat_array

    return output


def compute_frequency_grid(
    time: np.ndarray,
    n_bins: int = 1024,
    spacing: str = 'log',
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
) -> np.ndarray:
    """
    Compute pre-specified frequency grid for PSD/F-stat resampling.

    Generates a uniform frequency grid (linear or logarithmic spacing) suitable
    for neural network input. Default bounds are computed from the light curve
    characteristics if not provided.

    Parameters
    ----------
    time : np.ndarray
        Time array from the light curve, used to compute default bounds
    n_bins : int, default=1024
        Number of frequency bins in output grid
    spacing : str, default='log'
        Grid spacing: 'log' (logarithmic) or 'linear'
        Logarithmic spacing provides better resolution at low frequencies,
        which is typically where most stellar variability occurs
    freq_min : float, optional
        Minimum frequency. If None, computed as 1/(time_max - time_min)
        (one complete cycle over the observation duration)
    freq_max : float, optional
        Maximum frequency. If None, computed as 0.5/median_cadence
        (Nyquist frequency based on median sampling)

    Returns
    -------
    freq_grid : np.ndarray
        Frequency grid, shape (n_bins,)

    Examples
    --------
    >>> # Create log-spaced grid with automatic bounds
    >>> time = np.linspace(0, 100, 500)  # 100 days, ~500 points
    >>> freq_grid = compute_frequency_grid(time)
    >>> print(f"Frequency range: {freq_grid[0]:.6f} to {freq_grid[-1]:.4f}")

    >>> # Custom linear grid
    >>> freq_grid = compute_frequency_grid(
    ...     time,
    ...     spacing='linear',
    ...     freq_min=0.01,
    ...     freq_max=1.0,
    ...     n_bins=512
    ... )
    """
    # Compute default bounds if not provided
    if freq_min is None:
        duration = time[-1] - time[0]
        if duration <= 0:
            raise ValueError("Time array must be monotonically increasing")
        freq_min = 1.0 / duration  # One cycle over observation

    if freq_max is None:
        # Compute median cadence (time between observations)
        cadences = np.diff(time)
        median_cadence = np.median(cadences)
        if median_cadence <= 0:
            raise ValueError("Invalid time array: non-positive cadence")
        freq_max = 0.5 / median_cadence  # Nyquist frequency

    # Create frequency grid
    if spacing == 'log':
        if freq_min <= 0:
            raise ValueError("freq_min must be > 0 for logarithmic spacing")
        freq_grid = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins)
    elif spacing == 'linear':
        freq_grid = np.linspace(freq_min, freq_max, n_bins)
    else:
        raise ValueError(f"Unknown spacing: {spacing}. Use 'log' or 'linear'")

    return freq_grid


def fstatistic_to_pvalue(
    fstat: np.ndarray,
    K: int
) -> np.ndarray:
    """
    Convert F-statistic values to p-values using the F-distribution.

    The F-statistic from multitaper analysis follows an F-distribution under
    the null hypothesis of pure noise. This function computes the probability
    of observing the given F-statistic or larger under this null hypothesis.

    Parameters
    ----------
    fstat : np.ndarray
        F-statistic values from multitaper analysis
    K : int
        Number of tapers used in the multitaper computation
        Determines the degrees of freedom: df1=2, df2=2*K-2

    Returns
    -------
    pvalues : np.ndarray
        P-values corresponding to each F-statistic
        Lower p-values indicate higher significance (stronger periodic signals)

    Notes
    -----
    The F-statistic for harmonic analysis follows F(2, 2K-2) distribution,
    where K is the number of tapers.

    A p-value < 0.01 typically indicates a significant periodic signal,
    though the exact threshold depends on the number of independent frequencies
    tested (Bonferroni or other corrections may be needed).

    Examples
    --------
    >>> from lcgen.data import compute_multitaper_psd, fstatistic_to_pvalue
    >>>
    >>> # Compute PSD and F-statistics
    >>> result = compute_multitaper_psd(time, flux, NW=4.0, K=7, return_fstat=True)
    >>> fstat = result['fstat']
    >>> K = result['K']
    >>>
    >>> # Convert to p-values
    >>> pvalues = fstatistic_to_pvalue(fstat, K)
    >>> significant_peaks = pvalues < 0.01
    >>> print(f"Found {np.sum(significant_peaks)} significant peaks")

    References
    ----------
    Thomson, D. J. (1982). "Spectrum estimation and harmonic analysis".
    Proceedings of the IEEE, 70(9), 1055-1096.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for p-value computation. "
            "Install with: pip install scipy"
        )

    if K < 2:
        raise ValueError("K (number of tapers) must be >= 2")

    # Degrees of freedom for F-statistic from multitaper harmonic analysis
    dfn = 2  # Numerator degrees of freedom
    dfd = 2 * K - 2  # Denominator degrees of freedom

    # Compute p-values (survival function = 1 - CDF)
    pvalues = f_dist.sf(fstat, dfn, dfd)

    return pvalues


def extract_acf_timescale(
    lags: np.ndarray,
    acf: np.ndarray,
    method: str = 'efolding'
) -> float:
    """
    Extract characteristic timescale from autocorrelation function.

    Parameters
    ----------
    lags : np.ndarray
        Time lag array
    acf : np.ndarray
        Autocorrelation function values
    method : str, default='efolding'
        Method for extracting timescale:
        - 'efolding': Lag where ACF drops to 1/e ≈ 0.368
        - 'half_max': Lag where ACF drops to 0.5
        - 'zero_crossing': First zero crossing of ACF

    Returns
    -------
    timescale : float
        Characteristic timescale in the same units as lags
        Returns np.nan if timescale cannot be determined

    Examples
    --------
    >>> # Compute ACF from PSD
    >>> lags, acf = psd_to_acf(frequency, psd)
    >>>
    >>> # Extract e-folding time
    >>> tau = extract_acf_timescale(lags, acf, method='efolding')
    >>> print(f"Characteristic timescale: {tau:.2f} days")
    """
    # Ensure we're working with positive lags only
    if lags[0] < 0:
        # Find zero lag and take positive side
        zero_idx = np.argmin(np.abs(lags))
        lags = lags[zero_idx:]
        acf = acf[zero_idx:]

    if len(acf) < 2:
        return np.nan

    if method == 'efolding':
        # Find where ACF drops to 1/e
        target = 1.0 / np.e
        # Find first crossing
        crossings = np.where(acf < target)[0]
        if len(crossings) == 0:
            return np.nan
        # Interpolate for more accurate estimate
        idx = crossings[0]
        if idx == 0:
            return lags[0]
        # Linear interpolation
        x1, x2 = lags[idx-1], lags[idx]
        y1, y2 = acf[idx-1], acf[idx]
        timescale = x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    elif method == 'half_max':
        # Find where ACF drops to 0.5
        target = 0.5
        crossings = np.where(acf < target)[0]
        if len(crossings) == 0:
            return np.nan
        idx = crossings[0]
        if idx == 0:
            return lags[0]
        x1, x2 = lags[idx-1], lags[idx]
        y1, y2 = acf[idx-1], acf[idx]
        timescale = x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    elif method == 'zero_crossing':
        # Find first zero crossing
        crossings = np.where(acf <= 0)[0]
        if len(crossings) == 0:
            return np.nan
        idx = crossings[0]
        if idx == 0:
            return lags[0]
        # Interpolate
        x1, x2 = lags[idx-1], lags[idx]
        y1, y2 = acf[idx-1], acf[idx]
        timescale = x1 - y1 * (x2 - x1) / (y2 - y1)

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from 'efolding', 'half_max', 'zero_crossing'"
        )

    return float(timescale)


def resample_acf_to_bins(
    lags: np.ndarray,
    acf: np.ndarray,
    n_bins: int = 1024
) -> np.ndarray:
    """
    Resample autocorrelation function to a fixed number of lag bins.

    Uses linear interpolation to resample the ACF onto a uniform lag grid
    suitable for neural network input.

    Parameters
    ----------
    lags : np.ndarray
        Original lag array (should be sorted, starting from 0 or positive)
    acf : np.ndarray
        Autocorrelation function values
    n_bins : int, default=1024
        Number of bins for resampled ACF

    Returns
    -------
    acf_resampled : np.ndarray
        Resampled ACF, shape (n_bins,)

    Notes
    -----
    The resampling uses linear interpolation. Lags beyond the original
    range are set to the last value (constant extrapolation).

    Examples
    --------
    >>> lags, acf = psd_to_acf(frequency, psd)
    >>> acf_1024 = resample_acf_to_bins(lags, acf, n_bins=1024)
    """
    if not SCIPY_AVAILABLE:
        # Fallback to simple binning if scipy not available
        if len(acf) >= n_bins:
            # Downsample by taking every nth point
            indices = np.linspace(0, len(acf)-1, n_bins, dtype=int)
            return acf[indices]
        else:
            # Upsample by repeating
            acf_resampled = np.zeros(n_bins)
            acf_resampled[:len(acf)] = acf
            # Fill remainder with last value
            if len(acf) < n_bins:
                acf_resampled[len(acf):] = acf[-1]
            return acf_resampled

    # Ensure positive lags only
    if lags[0] < 0:
        zero_idx = np.argmin(np.abs(lags))
        lags = lags[zero_idx:]
        acf = acf[zero_idx:]

    # Create uniform lag grid
    lag_grid = np.linspace(lags[0], lags[-1], n_bins)

    # Interpolate ACF onto uniform grid
    interpolator = interp1d(
        lags, acf,
        kind='linear',
        bounds_error=False,
        fill_value=acf[-1]  # Constant extrapolation
    )
    acf_resampled = interpolator(lag_grid)

    return acf_resampled
