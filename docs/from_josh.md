
So the current architecture and training scheme includes what I call dynamic hierarchical masking where data modalities and portions of data are removed during training, um, I believe based on a variable threshold, also variable sizing, and on a per object basis, um, so that it varies within even a batch. And I want to pull out lots of details related to the implementation so that I can summarize and hand off a thorough description to another student who might be able to implement something similar. I also want this to include examples of code snippets for how we solve particular problems such as random number generation and selection.


Ran 6 commands, read 3 files
Here's a thorough write-up of the dynamic hierarchical masking implementation in DOROTHY, organized for handoff. The code lives in dorothy/data/augmentation.py and is invoked from dorothy/training/trainer.py inside the per-batch training loops.

Dynamic Hierarchical Masking — Implementation Notes
1. Conceptual overview
There are two masking augmentations applied at training time only, on each batch independently:

Class	Levels	Purpose
DynamicInputMasking	(1) survey, (2) wavelength block	Simulates missing surveys and gaps/bad-pixel regions in input spectra
DynamicLabelMasking	(1) labelset, (2) individual label	Simulates missing label sources (APOGEE/GALAH) and missing individual parameters
The "hierarchy" is: first decide which top-level groups (surveys / labelsets) to keep, then within each kept group decide which sub-elements (blocks / labels) to keep.

Every batch resamples fresh:

the keep probability at each level (uniform over a configurable range, e.g. [0.3, 1.0])
the block size for input masking (log-uniform over [1/N, f_max])
the block boundary offset (uniform integer in [0, block_size))
per-sample Bernoulli draws of which groups / sub-elements to keep
Each sample in the batch ends up with its own keep pattern — within a single batch different stars see different masking realizations.

Crucially, all of this is vectorized over the batch dimension on the GPU; there are no Python loops over samples.

2. Data shapes & conventions
Inputs and labels are stored as 3-channel tensors:

X[survey]:    (batch, 3, N_wavelengths)   channels = [flux, error, mask]
y[labelset]:  (batch, 3, n_params)        channels = [value, uncertainty, mask]
Channel 2 is the "natural" mask: 1 = valid (pixel/label is real), 0 = missing in the data itself.
Augmentation modifies channel 2 (turning 1s into 0s), and zeros channels 0 and 1 wherever it sets channel 2 to 0.
The heteroscedastic loss (dorothy/losses/heteroscedastic.py:163) multiplies the per-element loss by target[:, 2, :], so a masked entry contributes nothing to the loss.
This means masking is implemented entirely through the mask channel — the loss already handles "ignore where mask=0" correctly, so the augmentation is just a mask-mutation operation.

3. Configuration surface
From dorothy/config/schema.py:

# Multi-survey + block-level
input_masking:
  enabled: true
  p_survey_min: 0.3       # uniform draw bounds per-batch
  p_survey_max: 1.0
  f_min_override: null    # null => 1/N (single pixel) is min block size
  f_max: 0.5              # max block size as fraction of spectrum
  p_block_min: 0.3
  p_block_max: 1.0

# Multi-labelset + individual-label
label_masking:
  enabled: true
  p_labelset_min: 0.3
  p_labelset_max: 1.0
  p_label_min: 0.3
  p_label_max: 1.0
The constructors validate 0 <= min <= max <= 1 and 0 < f_min <= f_max <= 1.

4. The core vectorized pattern
The same six-step pattern is used at every level (survey, block, labelset, label). Understanding this pattern once explains the entire file:

# (1) Determine which entries are "available" (have any real data)
available = mask.any(dim=-1)             # bool, shape (batch, n_groups)

# (2) Per-sample short-circuit: if a sample has nothing available,
#     leave it untouched at the end.
any_available = available.any(dim=1)     # (batch,)

# (3) Generate one random number per (sample, group)
rand = torch.rand(batch, n_groups, device=device, dtype=dtype)

# (4) "Guaranteed keeper" — pick the available group with the LARGEST
#     random value as the one that must be kept.
rand_for_guaranteed = rand.clone()
rand_for_guaranteed[~available] = -float("inf")
guaranteed_idx = rand_for_guaranteed.argmax(dim=1)   # (batch,)

# (5) Independent Bernoulli over all (sample, group) pairs
keep = (rand < p_keep) & available

# (6) Force-keep the guaranteed index per sample
batch_idx = torch.arange(batch, device=device)
keep[batch_idx, guaranteed_idx] = True

# (7) Don't touch samples that had nothing available to begin with
keep = keep | ~any_available.unsqueeze(1)
Two tricks worth highlighting for the student:

Trick A — uniform random choice via argmax over random values. To pick one element uniformly at random from the available set, fill the unavailable positions with -inf and take argmax of the random tensor. This is a vectorized stand-in for np.random.choice per row, and it composes naturally with the Bernoulli draw because both reuse the same rand tensor.

Trick B — fancy indexing with torch.arange(batch). keep[torch.arange(batch), guaranteed_idx] = True sets exactly one element per row, with the column chosen by guaranteed_idx[i]. This is the idiomatic way to "scatter one True per row" without a Python loop.

5. DynamicLabelMasking — full walkthrough
The class has two paths: a single-labelset path (_apply_single) and a hierarchical path (_apply_hierarchical). The hierarchical version calls the single version internally.

Per-batch probability resampling
def __call__(self, y):
    p_keep_labelset = np.random.uniform(self.p_labelset_min, self.p_labelset_max)
    p_keep_label    = np.random.uniform(self.p_label_min,    self.p_label_max)
    ...
Note: probabilities are sampled per batch (not per sample). All samples in the batch share the same p_keep, but they each get independent Bernoulli draws against it.

Label-level masking (vectorized core)
def _apply_single(self, y, p_keep):
    batch_size, _, n_params = y.shape
    mask = y[:, 2, :]                         # natural mask
    available = mask > 0

    rand = torch.rand(batch_size, n_params, device=y.device, dtype=y.dtype)

    # Guaranteed keeper per sample
    rand_for_guaranteed = rand.clone()
    rand_for_guaranteed[~available] = -float("inf")
    guaranteed_idx = rand_for_guaranteed.argmax(dim=1)

    keep = (rand < p_keep) & available
    batch_indices = torch.arange(batch_size, device=y.device)
    keep[batch_indices, guaranteed_idx] = True

    # Apply to all three channels
    y_out = y.clone()
    keep_float = keep.float()
    y_out[:, 2, :] = mask * keep_float
    y_out[:, 0, :] = y[:, 0, :] * keep_float
    y_out[:, 1, :] = y[:, 1, :] * keep_float
    return y_out
Hierarchical (labelset → label)
For a dict {"apogee": tensor, "galah": tensor}:

Build a (batch, n_labelsets) availability matrix where entry (i, k) is True if sample i has at least one valid label in labelset k.
Run the 6-step pattern at the labelset level to get keep_labelset.
For each kept labelset, run _apply_single (label-level masking) on its tensor.
For samples where the labelset is not kept, multiply the whole tensor by zero.
Code excerpt (dorothy/data/augmentation.py:220):

for i, name in enumerate(labelsets):
    y_tensor = y_dict[name]
    labelset_kept = keep_labelset[:, i]                  # (batch,)

    if not labelset_kept.any():
        y_out[name] = torch.zeros_like(y_tensor)
        continue

    y_masked = self._apply_single(y_tensor, p_keep_label)
    labelset_mask = labelset_kept.float().view(-1, 1, 1)  # broadcastable
    y_out[name] = y_masked * labelset_mask
6. DynamicInputMasking — the interesting bits
The survey level is structurally identical to the labelset level. The block level is where the more interesting tricks live.

Step 1 — log-uniform block size
f_min = self.f_min_override if self.f_min_override is not None else (1.0 / N)
f_min = max(f_min, 1.0 / N)            # guarantee at least one pixel

log_f = np.random.uniform(np.log(f_min), np.log(self.f_max))
f = np.exp(log_f)
block_size = max(1, int(np.ceil(f * N)))
Log-uniform sampling means the model is equally likely to see "tiny" masks (single pixels) and "huge" masks (half the spectrum) — over training it explores all scales.

Step 2 — random offset to break fixed boundaries
Without an offset, blocks always align to positions 0, block_size, 2*block_size, .... The model can then learn positional shortcuts. Fix:

offset = np.random.randint(0, block_size)   # int in [0, block_size)
We virtually shift the spectrum to the right by offset. The first and last blocks become "partial":

N=100, block_size=50, offset=20:
  Without offset:  [0:50] [50:100]                  -> 2 blocks
  With offset=20:  [0:30] [30:80] [80:100]          -> 3 blocks
Step 3 — implement the offset with padding + reshape
Instead of writing a custom loop, the code pads the mask with zeros on the left (by offset) and on the right (to round up to a multiple of block_size), then reshapes:

total_length = offset + N
n_blocks     = int(np.ceil(total_length / block_size))
post_pad     = n_blocks * block_size - total_length

mask_padded  = F.pad(mask, (offset, post_pad), value=0)     # (batch, padded_len)
mask_blocks  = mask_padded.view(batch_size, n_blocks, block_size)
block_available = mask_blocks.any(dim=2)                    # (batch, n_blocks)
Two important consequences:

The padded positions are zeros, so the partial blocks naturally inherit the correct "available" status of their real pixels.
Once reshaped to (batch, n_blocks, block_size), all block-level operations are just per-block reductions — no Python loops.
Step 4 — apply the 6-step pattern at block level
rand_block = torch.rand(batch_size, n_blocks, device=device, dtype=dtype)

rand_for_guaranteed = rand_block.clone()
rand_for_guaranteed[~block_available] = -float("inf")
guaranteed_block_idx = rand_for_guaranteed.argmax(dim=1)

keep_block = (rand_block < p_keep_block) & block_available
batch_indices = torch.arange(batch_size, device=device)
keep_block[batch_indices, guaranteed_block_idx] = True
keep_block = keep_block | ~any_block_available.unsqueeze(1)
Step 5 — expand back from blocks to wavelengths
The block decision needs to be broadcast back to every pixel inside the block, then cropped to remove the offset padding:

# (batch, n_blocks) -> (batch, n_blocks, block_size) -> (batch, padded_len)
wavelength_keep_padded = keep_block.unsqueeze(2).expand(-1, -1, block_size)
wavelength_keep_padded = wavelength_keep_padded.reshape(batch_size, -1)

# Crop out the offset region at the front, keep N pixels
wavelength_keep = wavelength_keep_padded[:, offset : offset + N]
.expand is a zero-copy broadcast; .reshape materializes the view. The crop drops the synthetic left-padding.

Step 6 — apply to all three channels
X_out = X.clone()
keep_float = wavelength_keep.float()
X_out[:, 2, :] = mask * keep_float          # update mask
X_out[:, 0, :] = X[:, 0, :] * keep_float    # zero flux
X_out[:, 1, :] = X[:, 1, :] * keep_float    # zero error
7. Integration with the training loop
Both maskings are applied once per batch, after moving to device, before the forward pass. Example from _train_epoch_multi_survey (dorothy/training/trainer.py:1547):

X_batch = {s: arr[batch_idx].to(self.device) for s, arr in X.items()}
y_batch = y[batch_idx].to(self.device)

if self._input_masking is not None:
    X_batch = self._input_masking(X_batch, self._survey_wavelengths)
if self._label_masking is not None:
    y_batch = self._label_masking(y_batch)

output = self.model.forward(X_batch, has_data=has_data_batch)
loss   = self.loss_fn(output, y_batch)   # loss reads y_batch[:, 2, :] as mask
For the single-survey trainer, the tensor is briefly wrapped in {"default": X_batch} so the same DynamicInputMasking class handles both single- and multi-survey cases with p_survey_min = p_survey_max = 1.0 to disable survey-level masking.

Masking is never applied at validation/test time: self.model.eval() plus an if self._input_masking is not None guarded only on the training side.

8. Random number generation — the mixed numpy / torch approach
The code uses two RNG families on purpose:

np.random.uniform / np.random.randint for the per-batch scalars (block size, offset, keep-probabilities). These are tiny Python-level draws; staying on CPU is cheaper than launching a GPU kernel.
torch.rand(..., device=device, dtype=dtype) for the per-(sample, group) Bernoulli matrices. These need to be on the same device as the data, and torch.rand reuses PyTorch's RNG so seeded runs behave consistently across CPU/GPU.
A single rng = np.random.default_rng(self.config.seed) is used for shuffling indices in the trainer (trainer.py:649), but it is not threaded into the augmentation classes — they use the global numpy / torch RNGs. If you want strict reproducibility, you'd need to plumb a torch.Generator into the augmentation classes; the current code seeds at the top of training but doesn't isolate the augmentation RNG.

9. Edge cases the implementation handles
Edge case	How it's handled
Sample has no available labels/blocks at all	any_available check; sample is left untouched (see line 141 and 478)
All samples in batch have no data	Early-return the input unchanged
Single survey/labelset	Skip the top level entirely, just apply the inner masking
Top-level group is dropped for all samples	Return a zero tensor of the right shape instead of computing inner masking
Block size larger than spectrum	f_max <= 1 enforced; block_size = max(1, ceil(f*N))
Block size of zero	max(1, ...) floor
offset = 0	F.pad(..., (0, post_pad)) is a no-op on the left; pattern still works
10. Suggested handoff summary for another student
The augmentation has two classes, each with two hierarchical levels. Every level uses the same six-step vectorized pattern: (1) compute a per-row availability mask, (2) draw a random tensor of the right shape, (3) pick one "guaranteed keeper" per row via argmax over the random values with unavailable positions set to -inf, (4) form independent Bernoulli decisions with rand < p_keep, (5) OR in the guaranteed keeper via keep[arange(batch), guaranteed_idx] = True, (6) leave totally-empty rows alone.

The only level that differs is block-level input masking: block size is log-uniform per batch, a random integer offset shifts block boundaries to prevent positional shortcuts, and the offset is implemented by left-padding the mask with zeros before reshaping into (batch, n_blocks, block_size). After masking decisions are made at the block level, they're broadcast back to wavelengths with unsqueeze(2).expand(...).reshape(...) and the offset region is cropped off.

The masking only mutates the mask channel (channel 2) and zeros the corresponding values/errors; the heteroscedastic loss already multiplies by the mask, so masked entries contribute nothing to the gradient.

Key files to point them at:

dorothy/data/augmentation.py — both classes, ~530 lines, well-commented
dorothy/config/schema.py:471-625 — config schema with validators
dorothy/training/trainer.py:864-879 — call site (single-survey)
dorothy/training/trainer.py:1547-1553 — call site (multi-survey)
dorothy/losses/heteroscedastic.py:140-172 — how the loss consumes the mask
examples/variant5_all_surveys_masked.yaml — fully worked example config
tests/test_augmentation.py — many examples of expected behavior in unit-test form

