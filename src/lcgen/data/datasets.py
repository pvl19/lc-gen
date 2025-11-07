"""PyTorch dataset classes for different data modalities"""

import torch
from torch.utils.data import Dataset
import numpy as np


class FluxDataset(Dataset):
    """
    Dataset for stellar light curves with flux, uncertainty, and timestamps.

    Args:
        flux_array: Array of flux values, shape (N, seq_len)
        t_array: Array of timestamps, shape (N, seq_len)
        flux_err_array: Array of flux uncertainties, shape (N, seq_len), optional
    """

    def __init__(self, flux_array, t_array, flux_err_array=None):
        self.flux = torch.tensor(flux_array, dtype=torch.float32)
        self.flux_err = None
        if flux_err_array is not None:
            self.flux_err = torch.tensor(flux_err_array, dtype=torch.float32)
        self.t = torch.tensor(t_array, dtype=torch.float32)

    def __len__(self):
        return self.flux.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            x: Combined [flux, flux_err] tensor of shape (seq_len, 2) or (seq_len, 1)
            t: Timestamps of shape (seq_len, 1)
            flux: Flux values of shape (seq_len, 1) (target for reconstruction)
            flux_err: Flux uncertainties of shape (seq_len, 1)
        """
        x_flux = self.flux[idx].unsqueeze(-1)  # [seq_len, 1]
        x_t = self.t[idx].unsqueeze(-1)        # [seq_len, 1]

        if self.flux_err is not None:
            x_err = self.flux_err[idx].unsqueeze(-1)  # [seq_len, 1]
            x = torch.cat([x_flux, x_err], dim=-1)    # [seq_len, 2]
        else:
            x = x_flux

        # Input is both flux+error, but target is flux only
        return x, x_t, x_flux, x_err if self.flux_err is not None else x_flux


class PowerSpectrumDataset(Dataset):
    """
    Dataset for power spectra or autocorrelation functions.

    Args:
        data: Tensor or array of shape (N, seq_len) or (N, channels, seq_len)
        labels: Optional labels for supervised learning
        transform: Optional transform to apply to data
    """

    def __init__(self, data, labels=None, transform=None):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            x: Power spectrum/ACF of shape (seq_len,) or (channels, seq_len)
            label: Optional label (if provided)
        """
        x = self.data[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.labels is not None:
            return x, self.labels[idx]
        return x


class MultiModalDataset(Dataset):
    """
    Dataset combining multiple modalities (ACF, PSD, F-stat, timeseries, tabular).

    This is a scaffold for future multi-modal architecture.

    Args:
        acf_data: ACF data, shape (N, seq_len)
        psd_data: PSD data, shape (N, seq_len)
        fstat_data: F-statistic data, shape (N, seq_len), optional
        timeseries_data: Light curve data (FluxDataset), optional
        tabular_data: Auxiliary tabular features, shape (N, n_features), optional
        targets: Target light curves for reconstruction, shape (N, target_len)
    """

    def __init__(
        self,
        acf_data=None,
        psd_data=None,
        fstat_data=None,
        timeseries_data=None,
        tabular_data=None,
        targets=None,
    ):
        self.acf_data = self._to_tensor(acf_data)
        self.psd_data = self._to_tensor(psd_data)
        self.fstat_data = self._to_tensor(fstat_data)
        self.timeseries_data = timeseries_data  # FluxDataset instance
        self.tabular_data = self._to_tensor(tabular_data)
        self.targets = self._to_tensor(targets)

        # Determine dataset length
        self.length = self._get_length()

    def _to_tensor(self, data):
        """Convert data to tensor if not None"""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data.float()

    def _get_length(self):
        """Determine dataset length from available data"""
        for data in [self.acf_data, self.psd_data, self.fstat_data, self.tabular_data, self.targets]:
            if data is not None:
                return len(data)
        if self.timeseries_data is not None:
            return len(self.timeseries_data)
        raise ValueError("At least one data modality must be provided")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with available modalities:
            {
                'acf': ACF tensor,
                'psd': PSD tensor,
                'fstat': F-statistic tensor,
                'timeseries': (flux, time, flux_err) tuple,
                'tabular': Tabular features,
                'target': Target light curve
            }
        """
        sample = {}

        if self.acf_data is not None:
            sample['acf'] = self.acf_data[idx]

        if self.psd_data is not None:
            sample['psd'] = self.psd_data[idx]

        if self.fstat_data is not None:
            sample['fstat'] = self.fstat_data[idx]

        if self.timeseries_data is not None:
            sample['timeseries'] = self.timeseries_data[idx]

        if self.tabular_data is not None:
            sample['tabular'] = self.tabular_data[idx]

        if self.targets is not None:
            sample['target'] = self.targets[idx]

        return sample
