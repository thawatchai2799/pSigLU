"""
Controlled wall-clock timing of pSigLU and LapLU on a GPU WITHOUT Triton (e.g. Windows).

Compares, per batch (forward + backward, CUDA-event timed, medians):
    relu, silu            fused built-in kernels (reference)
    laplu_eager           ReLU(x) - 0.5|x|exp(-|x|/b) as a composition of tensor ops,
                          backward traced by autograd (the implementation used for
                          every reported experiment)
    laplu_customgrad      same forward, but a custom autograd.Function with the
                          closed-form derivative and a single saved intermediate,
                          which removes the traced backward graph
    laplu_jit             TorchScript version: the TorchScript CUDA fuser (NNC) fuses
                          the elementwise chain into one kernel; no Triton needed
    laplu_compiled        torch.compile (inductor); skipped automatically if Triton
                          is not available
    psiglu_eager          ReLU(x) - 0.5|x|(1+|x|)^(-p) as a composition of tensor ops
    psiglu_jit            TorchScript-fused version of the same (no transcendental call)

Before timing, the script checks that every variant reproduces the eager
forward values and gradients (max abs error printed), so no variant can look
fast by being wrong.

Usage (in the training venv, on an otherwise idle GPU):
    python psiglu_timing_fused.py
    python psiglu_timing_fused.py --shape 256 3 32 32 --iters 300
Writes timing_fused_psiglu.csv to --outdir. ASCII output only.
v2: adds psiglu_jit_rational (p = 2 as r*r, Proposition 5) next to psiglu_jit_pow (powf); conv width follows --shape.
"""
import argparse, csv, os, statistics, warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit.script deprecation notice
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. eager (reference implementation used in all experiments)
# ---------------------------------------------------------------------------
class LapLUEager(nn.Module):
    def __init__(self, b=1.0):
        super().__init__(); self.b = float(b)

    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * torch.exp(-ax / self.b)


# ---------------------------------------------------------------------------
# 2. custom autograd.Function with closed-form derivative
#    f'(x) = H(x) - 0.5 sign(x) exp(-|x|/b) (1 - |x|/b)     (closed-form LapLU derivative)
# ---------------------------------------------------------------------------
class _LapLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        ax = x.abs()
        e = torch.exp(-ax / b)
        ctx.save_for_backward(x, e)
        ctx.b = b
        return torch.relu(x) - 0.5 * ax * e

    @staticmethod
    def backward(ctx, g):
        x, e = ctx.saved_tensors
        b = ctx.b
        ax = x.abs()
        d = (x > 0).to(g.dtype) - 0.5 * torch.sign(x) * e * (1.0 - ax / b)
        return g * d, None


class LapLUCustomGrad(nn.Module):
    def __init__(self, b=1.0):
        super().__init__(); self.b = float(b)

    def forward(self, x):
        return _LapLUFn.apply(x, self.b)


# ---------------------------------------------------------------------------
# 3. TorchScript (NNC CUDA fuser fuses the elementwise chain; no Triton)
# ---------------------------------------------------------------------------
@torch.jit.script
def _laplu_jit(x, b: float):
    ax = x.abs()
    return torch.relu(x) - 0.5 * ax * torch.exp(-ax / b)


class LapLUJit(nn.Module):
    def __init__(self, b=1.0):
        super().__init__(); self.b = float(b)

    def forward(self, x):
        return _laplu_jit(x, self.b)


# ---------------------------------------------------------------------------
# 4. pSigLU: eager and TorchScript-fused (rational, no exp)
# ---------------------------------------------------------------------------
class PSigLUEager(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = float(p)

    def forward(self, x):
        ax = x.abs()
        return torch.relu(x) - 0.5 * ax * (1.0 + ax) ** (-self.p)


@torch.jit.script
def _psiglu_jit_pow(x, p: float):
    # generic p: the fused kernel calls powf(), which is NOT the rational form
    ax = x.abs()
    return torch.relu(x) - 0.5 * ax * (1.0 + ax) ** (-p)


@torch.jit.script
def _psiglu_jit_rational(x):
    # p = 2, Proposition 5: one reciprocal and three multiplications, no transcendental call
    ax = x.abs()
    r = 1.0 / (1.0 + ax)
    return torch.relu(x) - 0.5 * ax * r * r


class PSigLUJitPow(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = float(p)

    def forward(self, x):
        return _psiglu_jit_pow(x, self.p)


class PSigLUJitRational(nn.Module):
    def forward(self, x):
        return _psiglu_jit_rational(x)


# ---------------------------------------------------------------------------
def make_stack(act, c: int = 3):
    # three activation sites interleaved with cheap 1x1 convs, so the activation is
    # measured inside a realistic forward/backward rather than in isolation
    return nn.Sequential(nn.Conv2d(c, c, 1), act, nn.Conv2d(c, c, 1), act, nn.Conv2d(c, c, 1), act)


def sync_conv_weights(models):
    ref = [m for m in models[0].modules() if isinstance(m, nn.Conv2d)]
    for mod in models[1:]:
        for a, b in zip([m for m in mod.modules() if isinstance(m, nn.Conv2d)], ref):
            a.weight.data.copy_(b.weight.data); a.bias.data.copy_(b.bias.data)


def check_correct(name, model, ref_model, x):
    """Same forward values and same input-gradient as the eager reference.
    TF32 is switched off for the check so the 1x1 convs run in true fp32 and any
    difference is attributable to the activation; timing re-enables the defaults."""
    tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _check_correct(name, model, ref_model, x)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32


def _check_correct(name, model, ref_model, x):
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    y1 = ref_model(x1); y2 = model(x2)
    fe = (y1 - y2).abs().max().item()
    y1.sum().backward(); y2.sum().backward()
    ge = (x1.grad - x2.grad).abs().max().item()
    # relative tolerances: the activations sit between 1x1 convs, so absolute
    # round-off grows with the channel width of --shape
    fs = y1.abs().max().item() + 1e-12; gs = x1.grad.abs().max().item() + 1e-12
    ok = fe / fs < 1e-5 and ge / gs < 1e-4
    print(f"  check {name:20s} max|dy|={fe:.2e} (rel {fe/fs:.1e})  max|dgrad|={ge:.2e} (rel {ge/gs:.1e})  {'OK' if ok else 'MISMATCH'}")
    return ok


def time_fn(model, x, iters, warmup):
    for _ in range(warmup):
        y = model(x); y.sum().backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); y = model(x); y.sum().backward(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    q = statistics.quantiles(times, n=4)
    return statistics.median(times), q[0], q[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs="+", default=[128, 3, 32, 32])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--outdir", default="results_timing")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    dev = "cuda"; torch.manual_seed(0)
    x = torch.randn(*args.shape, device=dev, requires_grad=True)
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| shape", tuple(args.shape))

    c = int(args.shape[1])
    eager = make_stack(LapLUEager(), c).to(dev)
    variants = {
        "relu": make_stack(nn.ReLU(), c).to(dev),
        "silu": make_stack(nn.SiLU(), c).to(dev),
        "laplu_eager": eager,
        "laplu_customgrad": make_stack(LapLUCustomGrad(), c).to(dev),
        "laplu_jit": make_stack(LapLUJit(), c).to(dev),
        "psiglu_eager": make_stack(PSigLUEager(), c).to(dev),
        "psiglu_jit_pow": make_stack(PSigLUJitPow(), c).to(dev),
        "psiglu_jit_rational": make_stack(PSigLUJitRational(), c).to(dev),
    }
    try:
        variants["laplu_compiled"] = torch.compile(make_stack(LapLUEager(), c).to(dev))
        # force compilation now so a Triton failure is caught here, not during timing
        variants["laplu_compiled"](x).sum().backward()
    except Exception as e:
        print("torch.compile skipped:", str(e).splitlines()[0][:100])
        variants.pop("laplu_compiled", None)
    sync_conv_weights(list(variants.values()))

    print("\ncorrectness against eager LapLU (identical conv weights):")
    for name in ("laplu_customgrad", "laplu_jit", "laplu_compiled"):
        if name in variants:
            # run twice first so the TorchScript profiling executor has fused
            for _ in range(3): variants[name](x).sum().backward()
            if not check_correct(name, variants[name], eager, x):
                raise SystemExit(f"{name} does not match eager LapLU; not timing it")
    print("\ncorrectness of fused pSigLU against eager pSigLU:")
    for name in ("psiglu_jit_pow", "psiglu_jit_rational"):
        for _ in range(3): variants[name](x).sum().backward()
        if not check_correct(name, variants[name], variants["psiglu_eager"], x):
            raise SystemExit(f"{name} does not match eager pSigLU; not timing it")

    print("\ntiming (median forward+backward per batch):")
    rows = []
    for name, model in variants.items():
        med, q1, q3 = time_fn(model, x, args.iters, args.warmup)
        rows.append((name, med, q1, q3))
        print(f"  {name:20s} {med:8.4f} ms   IQR [{q1:.4f}, {q3:.4f}]")

    t = {r[0]: r[1] for r in rows}
    base = t["silu"]
    print(f"\noverhead relative to silu (fused built-in):")
    ov_eager = t["laplu_eager"] / base - 1.0
    for name in ("laplu_eager", "laplu_customgrad", "laplu_jit", "laplu_compiled", "psiglu_eager", "psiglu_jit_pow", "psiglu_jit_rational"):
        if name in t:
            ov = t[name] / base - 1.0
            removed = "" if name in ("laplu_eager", "psiglu_eager") or ov_eager <= 0 else f"   removes {100*(1-ov/ov_eager):5.1f}% of the eager overhead"
            print(f"  {name:20s} {100*ov:+6.1f}%{removed}")

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "timing_fused_psiglu.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "median_ms", "q1_ms", "q3_ms"]); w.writerows(rows)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
