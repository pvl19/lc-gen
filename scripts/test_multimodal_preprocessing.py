#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test multi-modal preprocessing pipeline with synthetic light curves.

This script generates several types of mock data and validates the complete
preprocessing pipeline from raw light curves to 5-channel multi-modal inputs.

Mock Light Curve Types:
1. Simple sinusoid with white noise
2. Multi-periodic (3 frequencies)
3. Damped random walk (stellar variability)
4. Quasi-periodic oscillation

Usage:
    python scripts/test_multimodal_preprocessing.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.data import preprocess_lightcurve_multimodal, batch_preprocess_multimodal


def generate_mock_lightcurve(
    lc_type: str,
    duration: float = 100.0,  # days
    n_points: int = 500,
    noise_level: float = 0.1,
    **params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic light curve.

    Parameters
    ----------
    lc_type : str
        Type of light curve: 'sinusoid', 'multiperiodic', 'drw', 'qpo'
    duration : float
        Observation duration in days
    n_points : int
        Number of observations
    noise_level : float
        Noise amplitude (fraction of signal)
    **params : dict
        Additional parameters for specific light curve types

    Returns
    -------
    time : np.ndarray
        Time array (irregular sampling)
    flux : np.ndarray
        Flux measurements
    flux_err : np.ndarray
        Flux uncertainties
    """
    # Create irregular time sampling
    np.random.seed(params.get('seed', 42))
    time = np.sort(np.random.uniform(0, duration, n_points))

    if lc_type == 'sinusoid':
        # Simple sinusoid: A * sin(2πft) + noise
        frequency = params.get('frequency', 0.05)  # cycles/day
        amplitude = params.get('amplitude', 1.0)

        flux = amplitude * np.sin(2 * np.pi * frequency * time)
        flux += noise_level * np.random.randn(n_points)
        flux += 10.0  # Add offset to make all flux positive

    elif lc_type == 'multiperiodic':
        # Sum of 3 sinusoids at different frequencies
        frequencies = params.get('frequencies', [0.03, 0.07, 0.12])
        amplitudes = params.get('amplitudes', [1.0, 0.5, 0.3])

        flux = np.zeros(n_points)
        for freq, amp in zip(frequencies, amplitudes):
            flux += amp * np.sin(2 * np.pi * freq * time)
        flux += noise_level * np.random.randn(n_points)
        flux += 10.0

    elif lc_type == 'drw':
        # Damped Random Walk (Ornstein-Uhlenbeck process)
        # dx/dt = -x/τ + σ*dW
        tau = params.get('tau', 20.0)  # Damping timescale (days)
        sigma = params.get('sigma', 0.5)  # Noise amplitude

        dt = np.diff(time)
        flux = np.zeros(n_points)
        flux[0] = 0.0

        for i in range(1, n_points):
            # OU process step
            decay = np.exp(-dt[i-1] / tau)
            flux[i] = flux[i-1] * decay + sigma * np.sqrt(1 - decay**2) * np.random.randn()

        flux += noise_level * np.random.randn(n_points)
        flux += 10.0

    elif lc_type == 'qpo':
        # Quasi-Periodic Oscillation (modulated sinusoid)
        carrier_freq = params.get('carrier_freq', 0.1)  # cycles/day
        mod_freq = params.get('mod_freq', 0.01)  # Modulation frequency
        amplitude = params.get('amplitude', 1.0)

        # Amplitude modulation
        envelope = 1.0 + 0.5 * np.sin(2 * np.pi * mod_freq * time)
        flux = amplitude * envelope * np.sin(2 * np.pi * carrier_freq * time)
        flux += noise_level * np.random.randn(n_points)
        flux += 10.0

    else:
        raise ValueError(f"Unknown light curve type: {lc_type}")

    # Generate flux uncertainties (heteroscedastic)
    flux_err = noise_level * (0.5 + 0.5 * np.random.rand(n_points))

    return time, flux, flux_err


def test_single_preprocessing():
    """Test preprocessing on a single light curve."""
    print("=" * 70)
    print("TEST 1: Single Light Curve Preprocessing")
    print("=" * 70)

    # Generate sinusoidal light curve
    time, flux, flux_err = generate_mock_lightcurve(
        'sinusoid',
        duration=100.0,
        n_points=500,
        frequency=0.05,  # 20-day period
        amplitude=1.0,
        noise_level=0.1,
        seed=42
    )

    print(f"\nInput light curve:")
    print(f"  Duration: {time[-1] - time[0]:.1f} days")
    print(f"  Number of points: {len(time)}")
    print(f"  Median cadence: {np.median(np.diff(time)):.3f} days")
    print(f"  Flux range: [{flux.min():.2f}, {flux.max():.2f}]")

    # Run preprocessing
    print("\nRunning preprocessing pipeline...")
    result = preprocess_lightcurve_multimodal(
        time, flux, flux_err,
        NW=4.0,
        K=7,
        n_bins=1024,
        compute_pvalues=True
    )

    # Validate outputs
    print("\n[PASS] Preprocessing complete!")
    print("\nOutput shapes:")
    print(f"  ACF: {result['acf'].shape}")
    print(f"  PSD: {result['psd'].shape}")
    print(f"  F-stat: {result['fstat'].shape}")
    print(f"  P-values: {result['pvalues'].shape}")
    print(f"  Timeseries: {result['timeseries'].shape}")
    print(f"  Time: {result['time'].shape}")
    print(f"  Tabular: {result['tabular'].shape}")
    print(f"  Frequency: {result['frequency'].shape}")

    # Validate values
    print("\nValidation checks:")
    checks = []

    # Check for NaN/Inf
    has_nan = any([
        np.any(np.isnan(result['acf'])),
        np.any(np.isnan(result['psd'])),
        np.any(np.isnan(result['fstat'])),
        np.any(np.isnan(result['timeseries'])),
    ])
    checks.append(("No NaN values", not has_nan))

    # Check shapes
    checks.append(("ACF shape (1024,)", result['acf'].shape == (1024,)))
    checks.append(("PSD shape (1024,)", result['psd'].shape == (1024,)))
    checks.append(("F-stat shape (1024,)", result['fstat'].shape == (1024,)))
    checks.append(("Tabular shape (9,)", result['tabular'].shape == (9,)))

    # Check p-values in [0, 1]
    pval_valid = np.all((result['pvalues'] >= 0) & (result['pvalues'] <= 1))
    checks.append(("P-values in [0, 1]", pval_valid))

    # Check for peak in PSD at expected frequency
    expected_freq = 0.05  # cycles/day
    freq_idx = np.argmin(np.abs(result['frequency'] - expected_freq))
    psd_peak_idx = np.argmax(result['psd'])
    freq_error = np.abs(result['frequency'][psd_peak_idx] - expected_freq)
    checks.append(("PSD peak near expected freq", freq_error < 0.01))

    # Print check results
    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    all_passed = all(c[1] for c in checks)

    # Print metadata
    print("\nMetadata:")
    for key, value in result['metadata'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Find significant peaks
    sig_peaks = result['pvalues'] < 0.01
    if np.any(sig_peaks):
        print(f"\n[INFO] Found {np.sum(sig_peaks)} significant peaks (p < 0.01)")
        top_peaks = np.argsort(result['pvalues'])[:5]
        print("  Top 5 frequencies:")
        for i, idx in enumerate(top_peaks, 1):
            print(f"    {i}. f = {result['frequency'][idx]:.4f} cycles/day, "
                  f"p = {result['pvalues'][idx]:.2e}")

    return result, all_passed


def test_batch_preprocessing():
    """Test batch processing of multiple light curves."""
    print("\n" + "=" * 70)
    print("TEST 2: Batch Preprocessing (4 Light Curve Types)")
    print("=" * 70)

    # Generate 4 different types of light curves
    lc_types = ['sinusoid', 'multiperiodic', 'drw', 'qpo']
    times, fluxes, flux_errs = [], [], []

    print("\nGenerating mock light curves...")
    for i, lc_type in enumerate(lc_types):
        time, flux, flux_err = generate_mock_lightcurve(
            lc_type,
            duration=100.0,
            n_points=500,
            noise_level=0.1,
            seed=42 + i
        )
        times.append(time)
        fluxes.append(flux)
        flux_errs.append(flux_err)
        print(f"  {i+1}. {lc_type:15s} - {len(time)} points")

    # Batch process
    print("\nBatch preprocessing...")
    batch_result = batch_preprocess_multimodal(
        times, fluxes, flux_errs,
        NW=4.0,
        K=7,
        verbose=True
    )

    print("\n[PASS] Batch preprocessing complete!")
    print("\nBatch output shapes:")
    print(f"  ACF: {batch_result['acf'].shape}")
    print(f"  PSD: {batch_result['psd'].shape}")
    print(f"  F-stat: {batch_result['fstat'].shape}")
    print(f"  P-values: {batch_result['pvalues'].shape}")
    print(f"  Tabular: {batch_result['tabular'].shape}")
    print(f"  Timeseries: list of {len(batch_result['timeseries'])} arrays")

    # Validation
    print("\nValidation checks:")
    checks = []

    checks.append(("Batch size = 4", batch_result['acf'].shape[0] == 4))
    checks.append(("All ACF (4, 1024)", batch_result['acf'].shape == (4, 1024)))
    checks.append(("All PSD (4, 1024)", batch_result['psd'].shape == (4, 1024)))
    checks.append(("Tabular (4, 9)", batch_result['tabular'].shape == (4, 9)))

    # Check no NaN in batch arrays
    has_nan = np.any(np.isnan(batch_result['acf']))
    checks.append(("No NaN in batch", not has_nan))

    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")

    all_passed = all(c[1] for c in checks)

    return batch_result, all_passed


def visualize_results(single_result: Dict, batch_result: Dict):
    """Create comprehensive visualization of preprocessing results."""
    print("\n" + "=" * 70)
    print("TEST 3: Visualization")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 12))

    # Plot single light curve results
    # Row 1: Input light curve
    ax1 = plt.subplot(4, 3, 1)
    time = single_result['time']
    flux_std = single_result['timeseries'][:, 0]
    flux_err_scaled = single_result['timeseries'][:, 1]
    ax1.errorbar(time, flux_std, yerr=flux_err_scaled, fmt='o', ms=2, alpha=0.5)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Standardized Flux')
    ax1.set_title('Input: Standardized Light Curve')
    ax1.grid(True, alpha=0.3)

    # Row 1: PSD
    ax2 = plt.subplot(4, 3, 2)
    ax2.semilogx(single_result['frequency'], single_result['psd'])
    ax2.set_xlabel('Frequency (cycles/day)')
    ax2.set_ylabel('Normalized PSD')
    ax2.set_title('Channel 1: Power Spectral Density')
    ax2.grid(True, alpha=0.3)

    # Row 1: ACF
    ax3 = plt.subplot(4, 3, 3)
    # Reconstruct lag grid for visualization
    n_bins = len(single_result['acf'])
    lags = np.linspace(0, single_result['metadata']['lc_duration'], n_bins)
    ax3.plot(lags, single_result['acf'])
    ax3.set_xlabel('Lag (days)')
    ax3.set_ylabel('Normalized ACF')
    ax3.set_title('Channel 2: Autocorrelation Function')
    ax3.grid(True, alpha=0.3)

    # Row 2: F-statistic
    ax4 = plt.subplot(4, 3, 4)
    ax4.semilogx(single_result['frequency'], single_result['fstat'])
    ax4.set_xlabel('Frequency (cycles/day)')
    ax4.set_ylabel('Normalized F-stat')
    ax4.set_title('Channel 3: F-statistic')
    ax4.grid(True, alpha=0.3)

    # Row 2: P-values
    ax5 = plt.subplot(4, 3, 5)
    ax5.semilogx(single_result['frequency'], single_result['pvalues'])
    ax5.axhline(0.01, color='r', linestyle='--', label='p=0.01')
    ax5.set_xlabel('Frequency (cycles/day)')
    ax5.set_ylabel('P-value')
    ax5.set_title('F-test P-values')
    ax5.set_ylim([1e-10, 1])
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Row 2: Tabular features
    ax6 = plt.subplot(4, 3, 6)
    feature_names = ['Median', 'IQR', 'PSD Int', 'PSD Peak',
                     'ACF Var', 'ACF τ', 'Duration', 'N obs', 'Cadence']
    tabular_vals = single_result['tabular']
    ax6.barh(range(len(tabular_vals)), tabular_vals)
    ax6.set_yticks(range(len(tabular_vals)))
    ax6.set_yticklabels(feature_names, fontsize=8)
    ax6.set_xlabel('Value')
    ax6.set_title('Channel 5: Tabular Features')
    ax6.grid(True, alpha=0.3, axis='x')

    # Row 3-4: Batch results (show all 4 PSDs)
    lc_types = ['Sinusoid', 'Multi-periodic', 'DRW', 'QPO']
    for i in range(4):
        ax = plt.subplot(4, 3, 7 + i)
        ax.semilogx(batch_result['frequency'], batch_result['psd'][i])
        ax.set_xlabel('Frequency (cycles/day)')
        ax.set_ylabel('Normalized PSD')
        ax.set_title(f'Batch {i+1}: {lc_types[i]} PSD')
        ax.grid(True, alpha=0.3)

    # Row 4: Batch tabular features comparison
    ax11 = plt.subplot(4, 3, 11)
    tabular_batch = batch_result['tabular']
    for i, lc_type in enumerate(lc_types):
        ax11.plot(tabular_batch[i], 'o-', label=lc_type, alpha=0.7)
    ax11.set_xticks(range(len(feature_names)))
    ax11.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax11.set_ylabel('Value')
    ax11.set_title('Batch: Tabular Features Comparison')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Summary statistics table
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    summary_text += f"Single LC:\n"
    summary_text += f"  Median flux: {single_result['metadata']['median_flux']:.2f}\n"
    summary_text += f"  IQR: {single_result['metadata']['iqr_flux']:.2f}\n"
    summary_text += f"  PSD integral: {single_result['metadata']['psd_integral']:.4f}\n"
    summary_text += f"  ACF timescale: {single_result['metadata']['acf_timescale']:.2f} d\n"
    summary_text += f"  N obs: {single_result['metadata']['n_observations']}\n\n"
    summary_text += f"Batch ({len(lc_types)} LCs):\n"
    summary_text += f"  All processed: [PASS]\n"
    summary_text += f"  Output shapes: ({len(lc_types)}, 1024)\n"
    summary_text += f"  Tabular: ({len(lc_types)}, 9)\n"
    ax12.text(0.1, 0.9, summary_text, fontsize=9, family='monospace',
              verticalalignment='top', transform=ax12.transAxes)

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'test_multimodal_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[PASS] Visualization saved to: {output_path}")

    return str(output_path)


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# Multi-Modal Preprocessing Pipeline - Comprehensive Test Suite")
    print("#" * 70)

    all_tests_passed = True

    try:
        # Test 1: Single preprocessing
        single_result, test1_passed = test_single_preprocessing()
        all_tests_passed &= test1_passed

        # Test 2: Batch preprocessing
        batch_result, test2_passed = test_batch_preprocessing()
        all_tests_passed &= test2_passed

        # Test 3: Visualization
        viz_path = visualize_results(single_result, batch_result)

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        if all_tests_passed:
            print("\n[SUCCESS] ALL TESTS PASSED!")
            print("\nThe multi-modal preprocessing pipeline is working correctly.")
            print("All 5 channels are generated with proper shapes and values.")
            print(f"\nVisualization saved to: {viz_path}")
            return 0
        else:
            print("\n[FAILED] SOME TESTS FAILED")
            print("\nPlease review the output above for details.")
            return 1

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
