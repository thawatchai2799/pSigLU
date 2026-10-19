"""
pSigLU on the Sobolev gradient-matching task (double backpropagation)

============================================================================
  DOUBLE-BACKPROP / C^2 DEMONSTRATION  --  Sobolev (gradient-matching) training
============================================================================
  This experiment tests the C-squared smoothness claim empirically
  claim: because pSigLU and LapLU are twice continuously differentiable (C^2), they can be
  used in training objectives that DIFFERENTIATE THROUGH THE GRADIENT (a
  "double-backward" pass), where ReLU degenerates because its second
  derivative is zero almost everywhere (plus a Dirac spike at the origin).

  Setup (a clean, controlled isolation of the second-derivative effect):
    We fit a small MLP f_theta : R^d -> R to a smooth target g, using a
    SOBOLEV loss that matches both the value AND the input-gradient:

        L = MSE(f, g)  +  lambda * MSE(d f / d x , d g / d x).

    The gradient term d f / d x is obtained with autograd (create_graph=True),
    so MINIMIZING it updates theta through the second derivative of the
    activation. For ReLU the gradient of f is piecewise-constant and its
    curvature signal is degenerate, so the gradient-matching term trains
    poorly; for the smooth activations (LapLU, GELU, Swish, ELU) it trains
    well. The expected, honest result is:

        pSigLU ~= LapLU ~= GELU ~= Swish ~= ELU   <<   ReLU   (on grad-MSE)

    i.e. the C^2 gates CLOSE THE GAP to the C-infinity gates while ReLU fails. They
    is not expected to BEAT GELU/Swish here (it only reaches C^2), and we
    report it as such.

  Output (same CSV conventions as the classification scripts, so the same
  analysis tooling works):
    raw_results.csv     : task, activation, seed, val_mse, grad_mse,
                          total_loss, seconds
    history_long.csv    : task, activation, seed, step, train_loss,
                          val_mse, grad_mse
    checkpoint.jsonl    : one finished (task, activation, seed) per line;
                          re-running resumes and SKIPS finished jobs.
    step_ckpt/*.pt      : per-step checkpoint (added for Colab Pro+, whose
                          sessions can disconnect every ~2h) so a single
                          interrupted (task, activation, seed) run resumes
                          from its last saved step instead of restarting.

  This is CHEAP (tiny MLPs, scalar targets): the full default grid
  (3 tasks * 5 activations * 5 seeds = 75 short runs) finishes well under an
  hour even on CPU, and in minutes on a GPU. Run it first.

  USAGE:
      pip install torch numpy pandas
      python pSigLU_DoubleBackprop_Sobolev.py
      # subset of activations / seeds (for chunking or quick checks):
      python pSigLU_DoubleBackprop_Sobolev.py --activations laplu relu gelu
      python pSigLU_DoubleBackprop_Sobolev.py --seed-start 5 --seeds 5 \
             --outdir results_sobolev_run2
      # ablate the gradient weight lambda or the number of steps:
      python pSigLU_DoubleBackprop_Sobolev.py --lam 1.0 --steps 8000
============================================================================
"""

import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  LapLU  (identical to the classification scripts)
# ---------------------------------------------------------------------------
class LapLU(nn.Module):
    """LapLU(x) = relu(x) - 0.5*|x|*exp(-|x|/b)  (branch-free Laplace-CDF gate)."""
    def __init__(self, b: float = 1.0, learnable: bool = False):
        super().__init__()
        t = torch.tensor(float(b))
        if learnable:
            self.b = nn.Parameter(t)
        else:
            self.register_buffer("b", t)

    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * torch.exp(-ax / self.b)


class RectLU(nn.Module):
    """
    RectLU (Rectangle Linear Unit), an earlier compact-support activation
    (v1 cubic form), included here as a legacy baseline. Re-implemented under this script's protocol.

        RectLU(x) = 0                    for x < L
                  = x * (x - L)^2 * m    for L <= x < 0   (central, cubic)
                  = x                    for x >= 0
    with L = -sqrt(1/m). The central branch is a cubic with a double root at
    L, so the FIRST derivative is continuous (RectLU is C^1) but the SECOND
    derivative is discontinuous at L and 0 (NOT C^2). Curvature is therefore
    confined to the narrow band [L, 0]; elsewhere f'' = 0. m = 10 gives
    L ~ -0.316 (the best value on CIFAR-100 in the original work).
    """
    def __init__(self, m: float = 10.0):
        super().__init__()
        import math as _m
        self.m = float(m)
        self.L = -_m.sqrt(1.0 / self.m)   # L = -sqrt(1/m)

    def forward(self, x):
        L, m = self.L, self.m
        central = x * (x - L) * (x - L) * m    # x * (x - L)^2 * m  (cubic, v1)
        return torch.where(x < L, torch.zeros_like(x),
                           torch.where(x < 0, central, x))



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


def make_activation(name: str):
    name = name.lower()
    if name in ("laplu", "laplub1"):
        return LapLU(b=1.0, learnable=False)
    if name == "laplub2":
        return LapLU(b=2.0, learnable=False)
    if name in ("psiglu", "psiglup2"):
        return pSigLU(p=2.0)
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
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name in ("rectlu", "rectlum10"):
        return RectLU(m=10.0)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


ACTIVATIONS = ['psiglu', 'laplu', 'relu', 'elu', 'gelu', 'swish']


# ---------------------------------------------------------------------------
# 2.  SMOOTH TARGET FUNCTIONS  (value + analytic gradient)
# ---------------------------------------------------------------------------
# Each task: dim, a sampler over the domain, the value g(x), and the analytic
# gradient grad g(x). All targets are smooth, so a smooth model CAN match both
# the value and the gradient; the question is whether the OPTIMIZER can drive
# the gradient term down, which is where the activation's 2nd derivative matters.
class Task:
    def __init__(self, name, dim, lo, hi, g, grad):
        self.name, self.dim, self.lo, self.hi = name, dim, lo, hi
        self.g, self.grad = g, grad

    def sample(self, n, device, gen):
        x = torch.rand(n, self.dim, generator=gen) * (self.hi - self.lo) + self.lo
        return x.to(device)


def build_tasks():
    tasks = {}
    # 1-D: g(x) = sin(3x)
    tasks["sin1d"] = Task(
        "sin1d", 1, -2.0, 2.0,
        g=lambda x: torch.sin(3.0 * x),
        grad=lambda x: 3.0 * torch.cos(3.0 * x),
    )
    # 1-D: g(x) = exp(-x^2)   (Gaussian bump)
    tasks["gauss1d"] = Task(
        "gauss1d", 1, -3.0, 3.0,
        g=lambda x: torch.exp(-(x ** 2)),
        grad=lambda x: (-2.0 * x) * torch.exp(-(x ** 2)),
    )
    # 2-D: g(x,y) = sin(2x) cos(2y)
    def g2(x):
        return (torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]))
    def grad2(x):
        gx = 2.0 * torch.cos(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2])
        gy = -2.0 * torch.sin(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2])
        return torch.cat([gx, gy], dim=1)
    tasks["sincos2d"] = Task("sincos2d", 2, -2.0, 2.0, g=g2, grad=grad2)
    return tasks


# ---------------------------------------------------------------------------
# 3.  MODEL  (small MLP; activation injected)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, act_name, in_dim, hidden=64, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), make_activation(act_name)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 4.  TRAIN ONE (task, activation, seed)
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_of(f_sum_fn, x):
    """d/dx of the scalar output, with graph kept for double-backward."""
    f = f_sum_fn(x)
    g = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    return f, g


def evaluate(model, task, x_eval, device):
    x = x_eval.clone().requires_grad_(True)
    f = model(x)
    gf = torch.autograd.grad(f.sum(), x, create_graph=False)[0]
    with torch.no_grad():
        gtrue = task.grad(x_eval)
        val_mse = torch.mean((f.detach() - task.g(x_eval)) ** 2).item()
        grad_mse = torch.mean(torch.sum((gf.detach() - gtrue) ** 2, dim=1)).item()
    return val_mse, grad_mse


def train_one(task, act_name, seed, steps, lam, lr, n_train, n_eval,
              device, log_every, step_ckpt_dir=None, ckpt_every=None):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    model = MLP(act_name, in_dim=task.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # fixed train / eval sets drawn from the domain. These are 100%
    # deterministic given (task, seed), so a resumed run can regenerate
    # them from scratch instead of needing to save them to disk.
    x_train = task.sample(n_train, device, gen)
    x_eval = task.sample(n_eval, device, torch.Generator().manual_seed(seed + 999))
    g_train = task.g(x_train).detach()
    gradg_train = task.grad(x_train).detach()

    # ---- step-level checkpoint (lets a single long run resume mid-training;
    # matters on Colab where a session can be cut off every ~2h) ----
    start_step = 1
    history = []
    ck_path = None
    if step_ckpt_dir is not None:
        os.makedirs(step_ckpt_dir, exist_ok=True)
        ck_path = os.path.join(step_ckpt_dir, f"step_{task.name}_{act_name}_seed{seed}.pt")
        if os.path.exists(ck_path):
            try:
                ck = torch.load(ck_path, map_location=device, weights_only=True)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                start_step = ck["step"] + 1; history = ck.get("history", [])
                print(f"      (resumed {task.name}/{act_name}/seed{seed} from step {start_step}/{steps})")
            except Exception as e:
                print(f"      (could not load step checkpoint, starting run over: {e})")
                start_step, history = 1, []

    ckpt_every = ckpt_every or log_every  # default: checkpoint exactly when we log
    mse = nn.MSELoss()
    for step in range(start_step, steps + 1):
        x = x_train.clone().requires_grad_(True)
        f, gf = grad_of(model, x)
        loss_val = mse(f, g_train)
        loss_grad = mse(gf, gradg_train)
        loss = loss_val + lam * loss_grad
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            v, gm = evaluate(model, task, x_eval, device)
            history.append({"step": step,
                            "train_loss": round(loss.item(), 6),
                            "val_mse": round(v, 6),
                            "grad_mse": round(gm, 6)})
        if ck_path is not None and (step % ckpt_every == 0 or step == steps):
            tmp = ck_path + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "history": history}, tmp)
            os.replace(tmp, ck_path)   # atomic on the same filesystem

    v, gm = evaluate(model, task, x_eval, device)
    if ck_path is not None and os.path.exists(ck_path):
        os.remove(ck_path)   # run finished: drop the step checkpoint
    return {"val_mse": v, "grad_mse": gm, "total_loss": v + lam * gm,
            "history": history}


# ---------------------------------------------------------------------------
# 5.  CHECKPOINT  (job-level; finished jobs are skipped on re-run)
# ---------------------------------------------------------------------------
def _job_key(task, act, seed):
    return f"{task}|{act}|{seed}"


def load_checkpoint(path):
    done, rows = set(), []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(rec)
                done.add(_job_key(rec["task"], rec["activation"], rec["seed"]))
    return done, rows


def append_checkpoint(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _fmt_hms(s):
    s = int(round(s)); return f"{s//3600:d}h{(s%3600)//60:02d}m{s%60:02d}s"


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sin1d", "gauss1d", "sincos2d"])
    ap.add_argument("--activations", nargs="+", default=ACTIVATIONS)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="index of first seed; use 0,5,10,... across runs for distinct seeds")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the gradient-matching term")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--outdir", default="./results_doublebackprop_sobolev")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "checkpoint.jsonl")
    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("(--fresh) removed existing checkpoint")

    all_tasks = build_tasks()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    jobs = [(t, a, s) for t in args.tasks for a in args.activations for s in seeds]
    done, all_rows = load_checkpoint(ckpt_path)
    todo = [j for j in jobs if _job_key(*j) not in done]
    print(f"device={device} | total jobs={len(jobs)} | already done={len(jobs)-len(todo)} | to run={len(todo)}")

    durations = [r["seconds"] for r in all_rows if "seconds" in r]
    for i, (tname, act, seed) in enumerate(todo, 1):
        task = all_tasks[tname]
        t0 = time.time()
        r = train_one(task, act, seed, args.steps, args.lam, args.lr,
                       args.n_train, args.n_eval, device, args.log_every,
                       step_ckpt_dir=os.path.join(args.outdir, "step_ckpt"))
        dt = time.time() - t0
        durations.append(dt)
        rec = {"task": tname, "activation": act, "seed": seed,
               "val_mse": round(r["val_mse"], 6), "grad_mse": round(r["grad_mse"], 6),
               "total_loss": round(r["total_loss"], 6),
               "seconds": round(dt, 2), "history": r["history"]}
        append_checkpoint(ckpt_path, rec)
        all_rows.append(rec)
        avg = sum(durations) / len(durations)
        eta = avg * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {tname:9s} {act:6s} seed={seed} -> "
              f"val_mse={r['val_mse']:.4e}  grad_mse={r['grad_mse']:.4e}  "
              f"({_fmt_hms(dt)}; ETA ~{_fmt_hms(eta)})")

    # ---- write CSV / JSON summaries ----
    try:
        import pandas as pd
        df = pd.DataFrame([{k: r[k] for k in
                            ("task", "activation", "seed", "val_mse", "grad_mse", "total_loss", "seconds")}
                           for r in all_rows])
        df.to_csv(os.path.join(args.outdir, "raw_results.csv"), index=False)
        for tname in sorted(df["task"].unique()):
            sub = df[df["task"] == tname]
            g = sub.groupby("activation")[["val_mse", "grad_mse"]].agg(["mean", "std"])
            g.to_csv(os.path.join(args.outdir, f"summary_{tname}.csv"))
    except Exception as e:
        print("pandas summary skipped:", e)

    import csv as _csv
    with open(os.path.join(args.outdir, "history_long.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["task", "activation", "seed", "step", "train_loss", "val_mse", "grad_mse"])
        for r in all_rows:
            for h in r.get("history", []):
                w.writerow([r["task"], r["activation"], r["seed"],
                            h["step"], h["train_loss"], h["val_mse"], h["grad_mse"]])

    print(f"\nDONE. Wrote raw_results.csv, history_long.csv, summary_<task>.csv to {args.outdir}/")
    print("Re-running the same command resumes and skips finished jobs; add --fresh to restart.")
    print("Send me the whole output folder (or raw_results.csv + history_long.csv) and I'll plot it.")


if __name__ == "__main__":
    main()
