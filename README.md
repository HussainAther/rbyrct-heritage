# rbyrct-heritage

A small synthetic research probe for **adaptive ray-by-ray tomography in cultural heritage imaging**.

## Question

Can a steerable/adaptive acquisition policy recover hidden internal structures
inside a sealed artifact with fewer measurements than conventional sparse-view CT?

## V0 phantom

The synthetic object contains:

- an outer ceramic-like vessel wall,
- heterogeneous attenuation,
- a hidden high-density insert,
- a small internal cavity,
- a crack-like feature.

## Compared acquisition strategies

1. `dense_reference` — full 180-view reference acquisition.
2. `uniform_sparse` — evenly spaced sparse projection angles.
3. `random_sparse` — random sparse projection angles.
4. `adaptive` — sequential next-angle selection based only on the current reconstruction.

The adaptive method does **not** inspect unmeasured projection values when choosing
the next view. It forward-projects the current reconstruction and favors candidate
angles with higher predicted structural information while retaining angular diversity.

## Metrics

- MSE
- SSIM
- hidden-feature IoU
- acquired angle count
- approximate ray count
- runtime

## Minimum convincing experiment

Run each strategy at 5%, 10%, 20%, 40%, and 100% of the 180-angle acquisition.
The branch is promising if adaptive acquisition recovers hidden structure or global
image quality faster as a function of acquired rays than the sparse baselines.

## Run

```bash
python -m pip install -r requirements.txt
python experiments/experiment_001.py
```

Outputs are written to `results/` and `figures/`.

## V0 caveat

This repository tests the acquisition idea, not archaeological realism. Positive
results should be followed by realistic attenuation/noise models, material-specific
phantoms, scanner geometry, and eventually real heritage CT data.
