# Latent-space probe summary

Pearson r and MAE in transformed space (asinh for flux moments, log10 for Prot). Baseline columns: `none` = real RNN latents (BiDirectionalMinGRU multiscale-pooled); `MLP` = pooled MLP-baseline features (k-averaged, mean+std); `gaussian` = iid N(0,1) noise of the same shape; `shuffle` = real latents permuted row-wise. The MLP cache covers all sequences in the H5 files, the RNN cache covers a subset, hence the per-baseline `n` columns differ.

| probe | transform | n (none) | n (MLP) | n (gaussian) | n (shuffle) | r (none) | r (MLP) | r (gaussian) | r (shuffle) | MAE (none) | MAE (MLP) | MAE (gaussian) | MAE (shuffle) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flux_kurt | asinh | 31912 | 60631 | 31912 | 31912 | +0.999 | +0.952 | -0.009 | +0.020 | 0.0569 | 0.3150 | 1.2273 | 1.2196 |
| flux_skew | asinh | 31912 | 60631 | 31912 | 31912 | +0.983 | +0.971 | -0.004 | +0.007 | 0.0349 | 0.0937 | 0.4211 | 0.4124 |
| lit_prot | log10 | 9431 | 17841 | 9431 | 9431 | +0.868 | +0.618 | +0.009 | +0.009 | 0.1796 | 0.5074 | 0.4980 | 0.4984 |
| num_flares | log10 | 13257 | 13346 | 13257 | 13257 | +0.812 | +0.753 | -0.016 | +0.021 | 0.2983 | 0.3439 | 0.5427 | 0.5416 |
| tars_prot | log10 | 21096 | 27002 | 21096 | 21096 | +0.924 | +0.916 | -0.011 | +0.015 | 0.1127 | 0.1279 | 0.4283 | 0.4278 |
| total_ed | log10 | 13257 | 13346 | 13257 | 13257 | +0.844 | +0.822 | -0.051 | +0.016 | 0.4608 | 0.4876 | 0.8970 | 0.8946 |
