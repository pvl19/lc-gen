#!/bin/bash
# Compute multi-taper power spectra, f-statistics, and ACFs for all light
# curves in the pretrain H5 files.  Results are written to separate H5 files
# (*_spectra.h5) to avoid any risk of corrupting the original data.

NW=4
K=7
MAX_FREQ=50
FREQ_RES=0.02

echo "=== Computing power spectra for pretrain H5 ==="
python scripts/compute_power_spectra.py \
    --h5_path final_pretrain/timeseries_pretrain.h5 \
    --output_h5 final_pretrain/timeseries_pretrain_spectra.h5 \
    --nw ${NW} \
    --k ${K} \
    --max_freq ${MAX_FREQ} \
    --freq_res ${FREQ_RES} \
    --overwrite

echo ""
echo "=== Computing power spectra for exoplanet hosts H5 ==="
python scripts/compute_power_spectra.py \
    --h5_path final_pretrain/timeseries_exop_hosts.h5 \
    --output_h5 final_pretrain/timeseries_exop_hosts_spectra.h5 \
    --nw ${NW} \
    --k ${K} \
    --max_freq ${MAX_FREQ} \
    --freq_res ${FREQ_RES} \
    --overwrite

echo ""
echo "Done."
