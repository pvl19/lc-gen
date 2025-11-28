import h5py

def trunc_data(input_file,keys=['flux','flux_err','time'],max_length=512):

    trunc_data = {}
    with h5py.File(input_file, 'r') as hf:
        for k in keys:
            data = hf[k]
            trunc_data[k] = data[:,:max_length]
    return trunc_data