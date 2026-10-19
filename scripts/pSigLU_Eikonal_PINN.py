#!/usr/bin/env python3
"""
============================================================================
  Eikonal PINN  --  activation comparison on a first-order PDE objective
============================================================================
  WHY THIS EXPERIMENT
  -------------------
  The Sobolev experiment shows that a continuous second derivative matters
  when the training objective differentiates through the gradient. This
  script tests the same claim on a genuine partial differential equation
  rather than a synthetic target.

  We solve the Eikonal equation

        || grad u(x) ||  =  1      for x in Omega
                  u(x)  =  0      for x on the boundary dOmega

  whose solution is the signed distance function (SDF) of the shape. This
  is a FIRST-ORDER PDE: the residual needs grad u, so backpropagating the
  loss differentiates through the activation's SECOND derivative exactly
  once. That is the regime where a C-squared activation is expected to
  help and where ReLU (second derivative zero almost everywhere) should
  degrade.

  Note that second-order PDEs (Poisson, Burgers) are deliberately NOT used
  here: their residual already contains the second derivative, so training
  would differentiate the THIRD derivative, which is discontinuous for
  LapLU. Those tasks are outside the range this activation targets.

  GROUND TRUTH
  ------------
  The default shape is the unit circle, whose exact SDF is ||x|| - 1. We
  therefore report the true SDF error on a dense grid, not merely the PDE
  residual, so a network cannot look good by satisfying the residual with
  the wrong solution.

  Activations: laplu (b=1), relu, gelu, swish, elu
  Metrics per run:
      sdf_mae        mean |u - u_true| on a dense grid   (lower is better)
      eikonal_res    mean (||grad u|| - 1)^2 on the grid (lower is better)
      surface_err    mean |u| on the boundary            (lower is better)
      seconds        wall-clock time for the run

  USAGE
      python pSigLU_Eikonal_PINN.py                    # circle, 10 seeds
      python pSigLU_Eikonal_PINN.py --shape roundedbox
      python pSigLU_Eikonal_PINN.py --seeds 10 --steps 5000
      python pSigLU_Eikonal_PINN.py --outdir results_eikonal

  Writes <outdir>/raw_results.csv, one row per (shape, activation, seed),
  plus checkpoint.jsonl so an interrupted run can be resumed.
============================================================================
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
class LapLU(nn.Module):
    """LapLU(x) = ReLU(x) - 0.5 * |x| * exp(-|x| / b).  C-squared everywhere."""

    def __init__(self, b: float = 1.0):
        super().__init__()
        self.b = float(b)

    def forward(self, x):
        return torch.relu(x) - 0.5 * x.abs() * torch.exp(-x.abs() / self.b)



class pSigLU(nn.Module):
    """
    pSigLU (Parametric Sigmoid Linear Unit):
        pSigLU(x) = x * sigma_p(x),  where sigma_p is the CDF of a symmetric
        power-law (Lomax-type) density with tail index p.

    Branch-free closed form:
        pSigLU(x) = relu(x) - 0.5 * |x| * (1 + |x|) ** (-p)
    Equivalent piecewise form:
        = 0.5 * x * (1 - x) ** (-p)          for x <= 0
        = x * (1 - 0.5 * (1 + x) ** (-p))    for x >= 0

    Notes
    -----
    * p > 1 is required. At p = 1 the activation saturates to an ELU-like
      floor; for p < 1 the negative branch diverges.
    * The density has a corner at the origin, so the gate itself is only C1,
      but the corner coincides with the zero of the gating factor x, which
      lifts the activation to C2 (twice continuously differentiable).
    * The tail decays polynomially, i.e. more slowly than the exponential
      tail of LapLU or the Gaussian tail of GELU.
    """

    def __init__(self, p: float = 2.0, learnable: bool = False):
        super().__init__()
        if p <= 1.0:
            raise ValueError("pSigLU requires p > 1 (got %r)" % p)
        self.learnable = learnable
        if learnable:
            self.raw_p = nn.Parameter(torch.tensor(float(p) - 1.0).log())
        else:
            self.p = float(p)

    def _tail(self):
        # Keep p strictly greater than 1 under optimization. A bare 1 + exp(raw_p)
        # underflows to exactly 1.0 once raw_p goes strongly negative, which is the
        # one value __init__ rejects, so clamp the increment away from zero.
        if self.learnable:
            return 1.0 + self.raw_p.exp().clamp_min(1e-4)
        return self.p

    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * (1.0 + ax).pow(-self._tail())



class pSigLUScaled(nn.Module):
    """Scaled pSigLU: relu(x) - 0.5*|x|*(1 + |x|/s)**(-p).

    With s = p the origin curvature is f''(0) = p/s = 1, the same as LapLU
    at b = 1, while the tail stays a power law |x|**(1-p). Used to separate
    the effect of the tail from the effect of the origin curvature.
    """
    def __init__(self, p: float = 2.0, s: float = 2.0):
        super().__init__()
        if p <= 1.0 or s <= 0.0:
            raise ValueError("pSigLUScaled requires p > 1 and s > 0")
        self.p = float(p)
        self.s = float(s)

    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * (1.0 + ax / self.s).pow(-self.p)


def make_activation(name: str):
    name = name.lower()
    if name == "laplu":
        return LapLU(1.0)
    if name == "laplub05":
        return LapLU(0.5)
    if name.startswith("laplub"):          # e.g. laplub2 -> b = 2
        return LapLU(float(name[6:].replace("p", ".")))
    if name in ("psiglu", "psiglup2"):
        return pSigLU(p=2.0)
    if name == "psiglus2":            # p = 2, s = 2: f''(0) = 1 like LapLU(b = 1)
        return pSigLUScaled(p=2.0, s=2.0)
    if name == "psiglup12":
        return pSigLU(p=1.2)
    if name == "psiglup15":
        return pSigLU(p=1.5)
    if name == "psiglup3":
        return pSigLU(p=3.0)
    if name == "psiglup4":
        return pSigLU(p=4.0)
    if name == "psiglup6":
        return pSigLU(p=6.0)
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"unknown activation: {name}")


# --------------------------------------------------------------------------
# Shapes: exact signed distance functions used as ground truth
# --------------------------------------------------------------------------
def sdf_circle(p, radius: float = 1.0):
    return p.norm(dim=-1) - radius


def sdf_roundedbox(p, half=(0.9, 0.6), r: float = 0.2):
    hx, hy = half
    q = p.abs() - torch.tensor([hx, hy], device=p.device, dtype=p.dtype)
    outside = torch.clamp(q, min=0.0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    return outside + inside - r


def sample_surface(shape, n, device, generator):
    """Points lying exactly on the zero level set."""
    if shape == "circle":
        t = torch.rand(n, 1, device=device, generator=generator) * (2 * math.pi)
        return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
    if shape == "roundedbox":
        # sample the domain, then project onto the surface by Newton steps on the exact SDF
        p = (torch.rand(n, 2, device=device, generator=generator) * 4.0) - 2.0
        p = p.clone()
        for _ in range(50):
            p.requires_grad_(True)
            d = sdf_roundedbox(p)
            g = torch.autograd.grad(d.sum(), p, create_graph=False)[0]
            with torch.no_grad():
                p = p - d.unsqueeze(-1) * g / (g.norm(dim=-1, keepdim=True) + 1e-12)
            p = p.detach()
        return p
    raise ValueError(shape)


def exact_sdf(shape, p):
    return sdf_circle(p) if shape == "circle" else sdf_roundedbox(p)


def surface_normal(shape, p):
    """Outward unit normal on the zero level set (grad of the exact SDF)."""
    p = p.detach().requires_grad_(True)
    d = exact_sdf(shape, p)
    n = torch.autograd.grad(d.sum(), p, create_graph=False)[0]
    return (n / (n.norm(dim=-1, keepdim=True) + 1e-12)).detach()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, act_name, width=64, depth=3):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), make_activation(act_name)]
            d_in = width
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grad_u(model, x, create_graph=True):
    """grad of u wrt input. create_graph=True during training so the loss can
    backprop through the gradient (this is what exercises f''); False at eval."""
    x = x.detach().requires_grad_(True)
    u = model(x)
    g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                            create_graph=create_graph)[0]
    return (u, g) if create_graph else (u.detach(), g.detach())


# --------------------------------------------------------------------------
# Train / evaluate one run
# --------------------------------------------------------------------------
def run_one(shape, act_name, seed, steps, lam_eik, lam_nrm, n_dom, n_surf, lr, device):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    model = MLP(act_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Precompute a large pool of surface points ONCE. Re-projecting them every
    # step is pure overhead and, for roundedbox, dominates the run time.
    surf_pool = sample_surface(shape, max(8192, n_surf), device, gen).detach()
    # Outward normals break the sign symmetry of the Eikonal equation: both
    # +d and -d satisfy ||grad u||=1 with u=0 on the boundary, so without this
    # term a run lands on either branch at random.
    norm_pool = surface_normal(shape, surf_pool)

    t0 = time.time()
    for _ in range(steps):
        x_dom = (torch.rand(n_dom, 2, device=device, generator=gen) * 4.0) - 2.0
        idx = torch.randint(0, surf_pool.shape[0], (n_surf,),
                            device=device, generator=gen)
        x_srf = surf_pool[idx]
        n_srf = norm_pool[idx]

        _, g = grad_u(model, x_dom)
        eik = ((g.norm(dim=-1) - 1.0) ** 2).mean()          # first-order PDE residual
        u_srf, g_srf = grad_u(model, x_srf)
        surf = (u_srf ** 2).mean()                           # boundary condition
        nrm = ((g_srf - n_srf) ** 2).sum(dim=-1).mean()      # orientation (fixes the sign)

        loss = surf + lam_eik * eik + lam_nrm * nrm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    seconds = time.time() - t0

    # ---- evaluation on a fixed dense grid (identical for every activation) ----
    model.eval()
    lin = torch.linspace(-2.0, 2.0, 200, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    u_pred, g_pred = grad_u(model, grid, create_graph=False)
    u_true = exact_sdf(shape, grid)
    sdf_mae = (u_pred - u_true).abs().mean().item()
    sdf_mae_flipped = (u_pred + u_true).abs().mean().item()
    flipped = int(sdf_mae_flipped < sdf_mae)
    eik_res = ((g_pred.norm(dim=-1) - 1.0) ** 2).mean().item()

    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(12345)                              # same surface points for all runs
    x_srf_eval = sample_surface(shape, 4096, device, eval_gen)
    with torch.no_grad():
        surface_err = model(x_srf_eval).abs().mean().item()

    return {
        "shape": shape, "activation": act_name, "seed": seed,
        "sdf_mae": sdf_mae, "sdf_mae_signfree": min(sdf_mae, sdf_mae_flipped),
        "flipped": flipped, "eikonal_res": eik_res,
        "surface_err": surface_err, "seconds": seconds,
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="circle", choices=["circle", "roundedbox"])
    ap.add_argument("--activations", nargs="+",
                    default=["psiglu", "laplu", "relu", "gelu", "swish", "elu"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lam-eik", type=float, default=1.0)
    ap.add_argument("--lam-nrm", type=float, default=1.0,
                    help="weight of the surface-normal term that fixes the SDF sign")
    ap.add_argument("--n-dom", type=int, default=2048)
    ap.add_argument("--n-surf", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default="results_eikonal")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "checkpoint.jsonl")

    # Schema of one result row. A checkpoint written by an older version of this
    # script has different columns; such entries are discarded and re-run rather
    # than silently mixed with new ones (which would corrupt the CSV).
    FIELDS = ["shape", "activation", "seed", "sdf_mae", "sdf_mae_signfree",
              "flipped", "eikonal_res", "surface_err", "seconds"]

    done = set()
    rows = []
    if os.path.exists(ckpt_path):
        stale = 0
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    stale += 1
                    continue
                if set(r.keys()) != set(FIELDS):
                    stale += 1
                    continue
                rows.append(r)
                done.add((r["shape"], r["activation"], r["seed"]))
        print(f"resuming: {len(done)} run(s) already finished")
        if stale:
            print(f"  ignored {stale} incompatible checkpoint entry(ies) "
                  f"from an older script version - those runs will be repeated")

    total = len(args.activations) * args.seeds
    print(f"device={device}  shape={args.shape}  runs={total}  steps={args.steps}")

    with open(ckpt_path, "a") as ck:
        for act in args.activations:
            for seed in range(args.seeds):
                key = (args.shape, act, seed)
                if key in done:
                    continue
                r = run_one(args.shape, act, seed, args.steps, args.lam_eik,
                            args.lam_nrm, args.n_dom, args.n_surf, args.lr, device)
                rows.append(r)
                ck.write(json.dumps(r) + "\n")
                ck.flush()
                print(f"  {act:6s} seed {seed:2d} | sdf_mae {r['sdf_mae']:.5f} "
                      f"| eik {r['eikonal_res']:.5f} | {r['seconds']:.1f}s")

    if not rows:
        print("no runs to write - nothing to do")
        return

    csv_path = os.path.join(args.outdir, "raw_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    # ---- summary: mean +/- std per activation ----
    print(f"\n{'activation':<10} {'sdf_mae (mean+/-std)':<26} {'eikonal_res':<20} n")
    for act in args.activations:
        v = [r["sdf_mae"] for r in rows if r["activation"] == act]
        e = [r["eikonal_res"] for r in rows if r["activation"] == act]
        if not v:
            continue
        m = sum(v) / len(v)
        s = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        me = sum(e) / len(e)
        nf = sum(r["flipped"] for r in rows if r["activation"] == act)
        print(f"{act:<10} {m:.5f} +/- {s:.5f}          {me:.5f}              {len(v)}"
              f"   flipped={nf}")


if __name__ == "__main__":
    main()
