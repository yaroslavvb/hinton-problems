# 4-2-4 encoder

Boltzmann-machine reproduction of the experiment from Ackley, Hinton &
Sejnowski, *"A learning algorithm for Boltzmann machines"*, Cognitive Science 9
(1985).

![4-2-4 encoder animation](encoder.gif)

## Problem

Two groups of 4 visible binary units (`V1`, `V2`) are connected through 2
hidden binary units (`H`). Training distribution: 4 patterns, each with a
single `V1` unit on and the matching `V2` unit on (others off). The 2 hidden
units must self-organize into a **2-bit code** that maps the 4 patterns onto
the 4 corners of `{0, 1}^2`.

- **Visible**: 8 bits = `V1 (4) || V2 (4)`
- **Hidden**: 2 bits
- **Connectivity**: bipartite (visible ↔ hidden only) — `V1` and `V2`
  communicate exclusively through `H`
- **Training set**: 4 patterns

The interesting property: with only 2 hidden units, the network has *exactly*
log2(4) bits of bottleneck capacity. Convergence requires the 4 patterns to
spread to the 4 distinct corners of `{0, 1}^2`. Local minima where two
patterns share a hidden code are common.

## Files

| File | Purpose |
|---|---|
| `encoder_4_2_4.py` | Bipartite RBM trained with CD-k. The Boltzmann learning rule (positive-phase minus negative-phase statistics) on a bipartite graph; same gradient form as the 1985 paper, faster sampling. |
| `make_encoder_gif.py` | Generates `encoder.gif` (the animation at the top of this README). |
| `visualize_encoder.py` | Static training curves + final weight matrix + final hidden codes. |
| `viz/` | Output PNGs from the run below. |

## Running

```bash
python3 encoder_4_2_4.py --epochs 400 --seed 2
```

Training takes ~1 second on a laptop. Final accuracy: **100% (4/4)**.

To regenerate visualizations:

```bash
python3 visualize_encoder.py --epochs 400 --seed 2 --outdir viz
python3 make_encoder_gif.py  --epochs 400 --seed 2 --snapshot-every 5 --fps 12
```

## Results

| Metric | Value |
|---|---|
| Final accuracy | 100% (4/4) |
| Hidden codes | `(0,1), (1,1), (1,0), (0,0)` — the 4 corners of `{0,1}^2` |
| Restarts | 1 (plateau at 75% accuracy from epoch ~100, restart at epoch 201, converged by ~350) |
| Training time | ~1 sec |
| Hyperparameters | k=5, lr=0.05, momentum=0.5, batch_repeats=8, init_scale=0.1 |

## What the network actually learns

### Hidden codes

![hidden codes](viz/hidden_codes.png)

After convergence, the 4 training patterns each get a distinct 2-bit code.
Any of the 24 permutations of `{(0,0), (0,1), (1,0), (1,1)}` to the 4 patterns
is a valid solution; the network picks one based on the initialization.

### Weight matrix

![weights](viz/weights.png)

The two columns are the hidden units `H[0]` and `H[1]`. Red = positive,
blue = negative; square area is proportional to `sqrt(|w|)`. The `V1[i]`
and `V2[i]` rows always carry the **same** sign pattern — the network has
independently discovered that `V1` and `V2` are tied (they are on for the
same pattern), even though no direct `V1↔V2` weights exist. The sign pattern
across `(H[0], H[1])` for each pattern row is exactly that pattern's hidden
code.

### Training curves

![training curves](viz/training_curves.png)

The vertical red dashed line at epoch 201 marks a **restart** triggered by
the plateau detector. The network had been stuck at 75% accuracy — three
patterns had distinct codes and the fourth had collapsed onto one of them.
Reinitializing the weights and continuing training produces the correct
4-corner solution.

The four panels track:
- **Reconstruction accuracy**: clamp `V1`, sample `V2`, count `argmax` matches.
- **Hidden-code separation**: mean pairwise L2 distance between the 4 hidden
  codes — converges to ≈ 1.13, slightly above the unit-square diagonal `√2`,
  reflecting saturation past the binary corners.
- **Weight norm**: `‖W‖_F` grows roughly linearly during convergence.
- **Reconstruction MSE**: mean-squared error of the predicted `V2`.

## Deviations from the 1985 procedure

1. **Sampling** — CD-5 (Hinton 2002) instead of simulated annealing. Same
   gradient form, faster sampling, sloppier asymptotics.
2. **Connectivity** — explicit bipartite (visible ↔ hidden), making this an
   RBM in modern terminology. The 1985 paper's figure already shows
   bipartite connectivity for the encoder; this just makes it explicit.
3. **Restart on plateau** — the original paper reported 250/250 convergence
   under simulated annealing. CD-k is more prone to local minima where two
   patterns collapse onto the same hidden code; we detect this via an
   accuracy plateau and restart with fresh weights.

## Open questions / next experiments

- Does the convergence rate (in number of weight updates) under CD-k come
  close to the 250/250 ≈ 110-cycle median reported in the 1985 paper for
  full simulated annealing?
- Can we eliminate the local-minima problem entirely by switching to PCD or
  by adding a small temperature schedule to the Gibbs sampler?
- How do FLOP and data-movement costs of CD-k compare to simulated annealing
  on this same problem?
- Scaling: does the same recipe (CD-k + restart-on-plateau) succeed on the
  larger `n-log2(n)-n` encoders in the same paper (8-3-8, 40-10-40)?
