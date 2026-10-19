# pSigLU: a parametric sigmoid linear unit with a power-law gate

pSigLU is a branch-free, twice continuously differentiable (C^2) activation
function whose gate is the cumulative distribution function of a symmetric
Lomax (power-law) law with tail index p > 1. It has the closed form

```
pSigLU_p(x) = ReLU(x) - 0.5 * |x| * (1 + |x|) ** (-p)
```

For integer p the function and its derivative are rational (one reciprocal and a
few multiplications, no transcendental call); p = 2 is the default used in all
main experiments. LapLU (the exponential-kernel member of the same family,
`ReLU(x) - 0.5 * |x| * exp(-|x| / b)`) is included as a baseline in every
experiment, together with ReLU, ELU, GELU and Swish.

This repository contains the code, the raw per-seed results from which every
table of the accompanying manuscript was computed, the figures, and a symbolic
check of every closed-form expression in the manuscript.

## Repository layout

```
scripts/     experiment scripts (one per benchmark family), the fused-kernel timing
             microbenchmark, the figure scripts, and verify_closed_forms.py
results/     raw per-seed result files (CSV) produced by the scripts (27 files,
             including the three timing CSVs)
figures/     the nine manuscript figures (PNG, 600 dpi)
```

## Requirements

- Python 3.9 or newer (the reported runs used Python 3.12.10)
- PyTorch with CUDA (the reported runs used PyTorch 2.6.0 + CUDA 12.4, cuDNN 9.1)
- torchvision, numpy, pandas, scipy, matplotlib, sympy

```
pip install -r requirements.txt
```

All reported runs used a single NVIDIA GeForce RTX 4050 Laptop GPU (6 GB).

## The activation

```python
import torch, torch.nn as nn

class pSigLU(nn.Module):
    def __init__(self, p: float = 2.0):
        super().__init__()
        if p <= 1.0:
            raise ValueError("pSigLU requires p > 1")
        self.p = float(p)
    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * (1.0 + ax).pow(-self.p)
```

The constraint p > 1 is structural: at p = 1 the negative branch saturates at
-1/2 instead of returning to zero, and for p < 1 it diverges. Every script
refuses p <= 1. The activation names accepted by the scripts are
`psiglu` (p = 2), `psiglup12`, `psiglup15`, `psiglup3`, `psiglup4`, `psiglup6`,
`psiglus2` (the scaled variant with kernel (1 + |x|/2)^-2, same origin curvature
as LapLU at b = 1), `laplu` (b = 1), `laplub05` (LapLU at b = 1/2, same origin
curvature as pSigLU at p = 2; the Eikonal script also accepts `laplub0p5`),
`relu`, `elu`, `gelu`, `swish`.

## Reproducing the experiments

Every script writes `checkpoint.jsonl` into its output directory after each run
and resumes from it, so a run can be interrupted and restarted. Times are rough
estimates for the 6 GB laptop GPU above.

```
# 1. main classification comparison (100 epochs, 15 seeds)          ~30 h
python scripts/psiglu_main_MNIST_CIFAR10_CIFAR100_e100.py --data-root ./data --outdir results_psiglu_main
python scripts/psiglu_main_FASHIONMNIST_SVHN_e100.py     --data-root ./data --outdir results_psiglu_fm_svhn
python scripts/psiglu_TinyImageNet_e100.py               --data-root ./data --outdir results_psiglu_tiny

# 2. tail-index ablation on the compact networks (p = 1.2 ... 6)   ~35 h
python scripts/psiglu_p_ablation_MNIST_CIFAR10_CIFAR100_e100.py --data-root ./data --outdir results_psiglu_p
python scripts/psiglu_p_ablation_FASHIONMNIST_SVHN_e100.py      --data-root ./data --outdir results_psiglu_p_fm

# 3. ResNet-18 (60 epochs, 15 seeds), six activations, then p = 1.2 and p = 1.5
python scripts/pSigLU_ResNet18_CIFAR.py --outdir results_psiglu_resnet
python scripts/pSigLU_ResNet18_CIFAR.py --activations psiglup12 --outdir results_resnet_p12
python scripts/pSigLU_ResNet18_CIFAR.py --activations psiglup15 --datasets cifar100 --outdir results_resnet_p15

# 4. Sobolev gradient matching (20 seeds) at five loss weights, then the p sweep at lambda = 1
for L in 0.1 0.3 1 3 10; do python scripts/pSigLU_DoubleBackprop_Sobolev.py --lam $L --outdir results_sobolev_lam$L; done
python scripts/pSigLU_DoubleBackprop_Sobolev.py --lam 1 --activations psiglup12 psiglup15 psiglup3 psiglup4 psiglup6 --outdir results_sobolev_lam1_psweep

# 5. Eikonal physics-informed network (10 seeds): circle, rounded box, and the p sweep on the circle
python scripts/pSigLU_Eikonal_PINN.py --shape circle     --outdir results_eikonal_circle
python scripts/pSigLU_Eikonal_PINN.py --shape roundedbox --outdir results_eikonal_roundedbox
python scripts/pSigLU_Eikonal_PINN.py --shape circle --activations psiglup12 psiglup15 psiglup3 psiglup4 psiglup6 --outdir results_eikonal_psweep

# 5b. scaled variant psiglus2 (power-law tail, origin curvature of LapLU at b = 1)
python scripts/pSigLU_Eikonal_PINN.py --shape circle     --activations psiglus2 laplu psiglu --outdir results_eikonal_scaled
python scripts/pSigLU_Eikonal_PINN.py --shape roundedbox --activations psiglus2 laplu psiglu --outdir results_eikonal_scaled_roundedbox
python scripts/pSigLU_ResNet18_CIFAR.py --activations psiglus2 --outdir results_resnet_scaled
#     (the two eikonal_*_scaled CSVs from this step are superseded by the four-activation job of step 5c,
#      which reproduces them bit for bit; they are kept for provenance)

# 5c. two-by-two design: tail (power law / exponential) x origin curvature (1 / 2)
#     the four activations in one job on the Eikonal network and on the compact network; the mirror LapLU (b = 1/2) on ResNet-18
python scripts/pSigLU_Eikonal_PINN.py --shape circle     --activations laplub0p5 laplu psiglu psiglus2 --outdir results_eikonal_mirror
python scripts/pSigLU_Eikonal_PINN.py --shape roundedbox --activations laplub0p5 laplu psiglu psiglus2 --outdir results_eikonal_mirror_roundedbox
python scripts/psiglu_p_ablation_MNIST_CIFAR10_CIFAR100_e100.py --data-root ./data --datasets cifar100 cifar10 --activations psiglus2 laplub05 laplu psiglu --outdir results_compact_curvature
python scripts/pSigLU_ResNet18_CIFAR.py --activations laplub05 --outdir results_resnet_mirror

# 6. fused-kernel timing (GPU otherwise idle); three configurations
python scripts/psiglu_timing_fused.py --outdir results_timing
python scripts/psiglu_timing_fused.py --iters 1000 --outdir results_timing_1000
python scripts/psiglu_timing_fused.py --shape 128 64 32 32 --iters 500 --outdir results_timing_big
```

The timing script checks every fused variant against the eager implementation
(forward values and input gradients, TF32 disabled for the check) before timing it.

## Results files

Each CSV in `results/` is the `checkpoint.jsonl` of one experiment with the
per-epoch history removed: one row per (dataset or task, activation, seed).
Classification files report `final_acc` and `best_acc`; Sobolev files report
`val_mse`, `grad_mse` and `total_loss`; Eikonal files (keyed by `shape`) report
`sdf_mae`, `eikonal_res` and `surface_err`, plus the sign-free SDF error
`sdf_mae_signfree` and a `flipped` flag that are not used in the manuscript;
every file reports the wall-clock `seconds` of the run. The three `timing_fused_*.csv` files hold the median and
interquartile range (ms) of each kernel variant.

Statistics in the manuscript are computed from these files with scipy:
two-sided paired t-tests matched by seed within a table for accuracies,
two-sided Mann-Whitney U tests for errors, Holm-Bonferroni correction within
each table, Welch's t-test for the ResNet-18 tail-index rows (separate jobs).

## Figures

```
python scripts/make_psiglu_figures_all.py <results_root> <extra_root>
```
`results_root` holds the main result folders (results_psiglu_*), `extra_root` the
tail-index sweeps (results_2026-10-08_*), as named in the script docstring. The script
reads the original run folders (checkpoint.jsonl and, for the learning-curve and
convergence panels, history_long.csv), not the trimmed CSVs in results/; the
per-epoch histories are not shipped, so those two panels need a re-run. The nine
figures are written to figures/ as Figure*_<name>.png (600 dpi) with matching PDFs.
`scripts/plot_curves.py` plots learning curves from a run folder and also needs
history_long.csv.

## Symbolic check

```
python scripts/verify_closed_forms.py
```

verifies with SymPy the derivative jumps of the family theorem up to order
seven for a generic kernel, the closed-form derivatives and shape constants of
pSigLU, the uniform distance to ReLU, and the third-derivative jumps of LapLU
and pSigLU.

## Related repository

The LapLU baseline implementation and its original per-seed files:
https://github.com/thawatchai2799/LapLU

## License

MIT (see LICENSE).
