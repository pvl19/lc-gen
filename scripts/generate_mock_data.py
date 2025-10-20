"""
Generate large-scale mock light curve dataset for testing MLP autoencoder.

Creates 50,000+ light curves with:
- 4 light curve types: sinusoid, multi-periodic, DRW, QPO
- 4 sampling strategies: regular, random gaps, variable cadence, astronomical
- Diverse properties: duration, n_obs, periods, SNR
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import h5py


def generate_time_series(lc_type, duration, n_points, sampling_strategy, **params):
    """
    Generate a single mock light curve with specified properties.

    Parameters
    ----------
    lc_type : str
        Type of light curve: 'sinusoid', 'multiperiodic', 'drw', 'qpo'
    duration : float
        Total duration in days
    n_points : int
        Number of observations
    sampling_strategy : str
        Sampling pattern: 'regular', 'random_gaps', 'variable_cadence', 'astronomical'
    **params : dict
        Light curve specific parameters

    Returns
    -------
    time : np.ndarray
        Observation times
    flux : np.ndarray
        Flux measurements
    flux_err : np.ndarray
        Flux uncertainties
    metadata : dict
        Light curve metadata
    """

    # Step 1: Generate time sampling
    if sampling_strategy == 'regular':
        # Evenly-spaced observations
        time = np.linspace(0, duration, n_points)

    elif sampling_strategy == 'random_gaps':
        # Random gaps removing 10-30% of data
        time = np.linspace(0, duration, n_points)
        gap_fraction = np.random.uniform(0.1, 0.3)
        n_gaps = int(n_points * gap_fraction / 20)  # Average gap size ~20 points

        keep_mask = np.ones(n_points, dtype=bool)
        for _ in range(n_gaps):
            gap_start = np.random.randint(0, n_points - 20)
            gap_size = np.random.randint(5, 30)
            keep_mask[gap_start:min(gap_start + gap_size, n_points)] = False

        time = time[keep_mask]
        n_points = len(time)

    elif sampling_strategy == 'variable_cadence':
        # Variable sampling rate (2-3x variation)
        # Divide into 3-5 segments with different cadences
        n_segments = np.random.randint(3, 6)
        segment_boundaries = np.sort(np.random.uniform(0, duration, n_segments - 1))
        segment_boundaries = np.concatenate([[0], segment_boundaries, [duration]])

        time_list = []
        for i in range(n_segments):
            seg_start = segment_boundaries[i]
            seg_end = segment_boundaries[i + 1]
            seg_duration = seg_end - seg_start

            # Variable number of points per segment
            seg_n_points = int(n_points * seg_duration / duration * np.random.uniform(0.5, 2.0))
            seg_n_points = max(10, seg_n_points)

            seg_time = np.linspace(seg_start, seg_end, seg_n_points)
            time_list.append(seg_time)

        time = np.concatenate(time_list)
        time = np.sort(time)
        n_points = len(time)

    elif sampling_strategy == 'astronomical':
        # Realistic astronomical sampling with season gaps
        # Assume ~6 month observing season, ~6 month gap
        n_seasons = max(1, int(duration / 180))

        time_list = []
        for season in range(n_seasons):
            # Each season: 150-180 days of observation
            season_duration = np.random.uniform(150, 180)
            season_start = season * 365  # One season per year

            if season_start + season_duration > duration:
                season_duration = duration - season_start

            if season_duration <= 0:
                break

            # Points in this season
            season_n_points = int(n_points * season_duration / duration)
            season_n_points = max(20, season_n_points)

            # Add weather gaps (10-20% of nights lost)
            season_time = np.linspace(season_start, season_start + season_duration,
                                     int(season_n_points * 1.3))

            # Remove random nights for weather
            keep_prob = np.random.uniform(0.8, 0.9)
            keep_mask = np.random.rand(len(season_time)) < keep_prob
            season_time = season_time[keep_mask]

            time_list.append(season_time)

        time = np.concatenate(time_list)
        time = np.sort(time)
        # Truncate to requested duration
        time = time[time <= duration]
        n_points = len(time)

    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    # Step 2: Generate flux signal based on light curve type
    if lc_type == 'sinusoid':
        # Single periodic signal
        period = params.get('period', np.random.uniform(2, 100))
        amplitude = params.get('amplitude', np.random.uniform(0.5, 2.0))
        phase = params.get('phase', np.random.uniform(0, 2 * np.pi))

        flux = amplitude * np.sin(2 * np.pi * time / period + phase)
        true_params = {'period': period, 'amplitude': amplitude, 'phase': phase}

    elif lc_type == 'multiperiodic':
        # Multiple periodic components
        n_periods = params.get('n_periods', np.random.randint(2, 5))
        periods = params.get('periods',
                           np.random.uniform(5, 100, n_periods))
        amplitudes = params.get('amplitudes',
                               np.random.uniform(0.3, 1.5, n_periods))
        # Sort by amplitude (strongest first)
        sort_idx = np.argsort(amplitudes)[::-1]
        periods = periods[sort_idx]
        amplitudes = amplitudes[sort_idx]

        phases = np.random.uniform(0, 2 * np.pi, n_periods)

        flux = np.zeros_like(time)
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            flux += amplitude * np.sin(2 * np.pi * time / period + phase)

        true_params = {'periods': periods, 'amplitudes': amplitudes, 'phases': phases}

    elif lc_type == 'drw':
        # Damped Random Walk (Ornstein-Uhlenbeck process)
        tau = params.get('tau', np.random.uniform(5, 50))  # Damping timescale
        sigma = params.get('sigma', np.random.uniform(0.5, 2.0))  # Variance

        # Generate via discretized OU process
        dt = np.diff(time)
        dt = np.concatenate([[np.median(dt)], dt])

        flux = np.zeros(n_points)
        for i in range(1, n_points):
            # OU process: dX = -X/tau * dt + sigma * sqrt(dt) * dW
            flux[i] = flux[i-1] * np.exp(-dt[i] / tau) + \
                     sigma * np.sqrt(1 - np.exp(-2 * dt[i] / tau)) * np.random.randn()

        true_params = {'tau': tau, 'sigma': sigma}

    elif lc_type == 'qpo':
        # Quasi-Periodic Oscillation (modulated sinusoid)
        central_period = params.get('central_period', np.random.uniform(10, 50))
        coherence = params.get('coherence', np.random.uniform(5, 20))  # Q-factor
        amplitude = params.get('amplitude', np.random.uniform(0.5, 2.0))

        # QPO = amplitude * sin(omega*t + random_walk_phase)
        omega = 2 * np.pi / central_period

        # Phase random walk
        dt = np.diff(time)
        dt = np.concatenate([[np.median(dt)], dt])

        phase = np.zeros(n_points)
        phase_diffusion = np.sqrt(2 * np.pi / (coherence * central_period))

        for i in range(1, n_points):
            phase[i] = phase[i-1] + phase_diffusion * np.sqrt(dt[i]) * np.random.randn()

        flux = amplitude * np.sin(omega * time + phase)

        true_params = {'central_period': central_period, 'coherence': coherence,
                      'amplitude': amplitude}

    else:
        raise ValueError(f"Unknown lc_type: {lc_type}")

    # Step 3: Add noise
    snr = params.get('snr', np.random.uniform(5, 30))
    signal_std = np.std(flux)
    noise_std = signal_std / snr

    # Add Gaussian noise
    noise = np.random.randn(n_points) * noise_std
    flux_clean = flux.copy()
    flux = flux + noise

    # Generate realistic uncertainties (varying per point)
    base_err = noise_std
    flux_err = base_err * np.random.uniform(0.8, 1.2, n_points)

    # Step 4: Add constant offset and rescale
    offset = params.get('offset', np.random.uniform(8, 12))
    flux = flux + offset

    # Metadata
    metadata = {
        'lc_type': lc_type,
        'sampling_strategy': sampling_strategy,
        'duration': duration,
        'n_points': n_points,
        'snr': snr,
        'offset': offset,
        'noise_std': noise_std,
        **true_params
    }

    return time, flux, flux_err, metadata


def generate_dataset(n_samples, output_dir, seed=42):
    """
    Generate full dataset of mock light curves.

    Parameters
    ----------
    n_samples : int
        Number of light curves to generate
    output_dir : Path
        Output directory for saving data
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribution of light curve types
    lc_type_probs = {
        'sinusoid': 0.30,
        'multiperiodic': 0.25,
        'drw': 0.25,
        'qpo': 0.20
    }

    # Distribution of sampling strategies
    sampling_probs = {
        'regular': 0.40,
        'random_gaps': 0.20,
        'variable_cadence': 0.20,
        'astronomical': 0.20
    }

    print(f"Generating {n_samples} mock light curves...")
    print(f"Output directory: {output_dir}")
    print(f"\nLight curve type distribution:")
    for lc_type, prob in lc_type_probs.items():
        print(f"  {lc_type}: {prob * 100:.0f}%")
    print(f"\nSampling strategy distribution:")
    for strategy, prob in sampling_probs.items():
        print(f"  {strategy}: {prob * 100:.0f}%")

    # Storage
    times = []
    fluxes = []
    flux_errs = []
    metadatas = []

    for i in tqdm(range(n_samples), desc="Generating"):
        # Sample light curve type
        lc_type = np.random.choice(
            list(lc_type_probs.keys()),
            p=list(lc_type_probs.values())
        )

        # Sample sampling strategy
        sampling_strategy = np.random.choice(
            list(sampling_probs.keys()),
            p=list(sampling_probs.values())
        )

        # Sample properties
        duration = np.random.uniform(50, 500)
        n_points = np.random.randint(200, 1000)

        # Generate light curve
        time, flux, flux_err, metadata = generate_time_series(
            lc_type, duration, n_points, sampling_strategy
        )

        times.append(time)
        fluxes.append(flux)
        flux_errs.append(flux_err)
        metadatas.append(metadata)

    # Save to pickle (preserves variable-length arrays)
    print(f"\nSaving to {output_dir / 'mock_lightcurves.pkl'}...")
    with open(output_dir / 'mock_lightcurves.pkl', 'wb') as f:
        pickle.dump({
            'times': times,
            'fluxes': fluxes,
            'flux_errs': flux_errs,
            'metadatas': metadatas
        }, f)

    print("Done!")
    print(f"\nDataset statistics:")
    print(f"  Total light curves: {len(times)}")
    print(f"  Avg points per LC: {np.mean([len(t) for t in times]):.0f}")
    print(f"  Avg duration: {np.mean([m['duration'] for m in metadatas]):.1f} days")

    # Print distribution verification
    lc_types_generated = [m['lc_type'] for m in metadatas]
    sampling_generated = [m['sampling_strategy'] for m in metadatas]

    print(f"\nGenerated distribution (LC types):")
    for lc_type in lc_type_probs.keys():
        count = lc_types_generated.count(lc_type)
        print(f"  {lc_type}: {count} ({count/n_samples*100:.1f}%)")

    print(f"\nGenerated distribution (Sampling):")
    for strategy in sampling_probs.keys():
        count = sampling_generated.count(strategy)
        print(f"  {strategy}: {count} ({count/n_samples*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mock light curve dataset"
    )
    parser.add_argument(
        '--n_samples', type=int, default=50000,
        help='Number of light curves to generate (default: 50000)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/mock_lightcurves',
        help='Output directory (default: data/mock_lightcurves)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    generate_dataset(args.n_samples, args.output_dir, args.seed)
