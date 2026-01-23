import h5py
import numpy as np

def extract_data(input_file,random_seed,keys=['flux','flux_err','time'],max_length=512,num_samples=None):

    trunc_data = {}
    with h5py.File(input_file, 'r') as hf:
        for k in keys:
            data = hf[k]
            trunc_data[k] = data[:,:max_length]
        # Also load lengths if available (capped at max_length)
        if 'length' in hf:
            lengths = hf['length'][:]
            trunc_data['length'] = np.minimum(lengths, max_length)
        else:
            # If no lengths stored, assume all positions are valid
            trunc_data['length'] = np.full(trunc_data[keys[0]].shape[0], max_length, dtype=np.int32)
    num_samples_original = trunc_data[keys[0]].shape[0]
    num_samples_keep = num_samples_original
    if num_samples is not None:
        num_samples_keep = min(num_samples, num_samples_original)  # Cap at available samples
    np.random.seed(random_seed)
    keep_idx = np.random.choice(num_samples_original, num_samples_keep, replace=False)
    for k in keys:
        trunc_data[k] = trunc_data[k][keep_idx]
    # Also subset lengths
    trunc_data['length'] = trunc_data['length'][keep_idx]
    # Store the indices so callers can load matching data from other sources
    trunc_data['indices'] = keep_idx
    print(f'Truncated data to max_length={max_length}, kept {num_samples_keep}/{num_samples_original} samples.')
    return trunc_data