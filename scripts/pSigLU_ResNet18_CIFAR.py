"""
pSigLU on ResNet-18 (CIFAR variant) -- 60 epochs, SGD + cosine, 15 seeds

============================================================================
  ResNet-18 (CIFAR variant)  --  activation comparison on CIFAR-10 / CIFAR-100
============================================================================
  Modern-architecture evaluation: the main results use a compact custom CNN,
  so we add a standard ResNet-18 (CIFAR
  stem: 3x3 conv, no max-pool) and swap ONLY the activation. Every block's
  nonlinearity is replaced by the chosen activation; all else is held fixed,
  so accuracy differences are attributable to the activation alone.

  Activations: laplu (b=1), relu, gelu, swish, elu  (+ laplub2 optional).
  Datasets:    cifar10, cifar100  (downloaded by torchvision on first run).
  Protocol:    SGD(momentum=0.9, wd=5e-4), lr=0.1, cosine annealing,
               standard CIFAR augmentation (pad-4 random crop + h-flip),
               60 epochs default, 3 seeds. (SGD+cosine is the expected
               ResNet/CIFAR recipe; use --optimizer adam to match the
               custom-CNN experiments instead.)

  Output (identical schema to the other scripts):
    raw_results.csv  : dataset, activation, seed, final_acc, best_acc, seconds
    history_long.csv : dataset, activation, seed, epoch, train_loss, test_acc
    checkpoint.jsonl : finished (dataset,activation,seed) jobs, skipped on re-run
    epoch_ckpt/*.pt  : per-epoch checkpoint so a single interrupted run resumes

  ---- FITTING A <5-HOUR WINDOW ----
  One ResNet-18 run is ~25-50 min on a modern GPU. The full default grid
  (2 datasets * 5 activations * 3 seeds = 30 runs) will NOT fit in 5 h, so
  CHUNK IT and let the checkpoint stitch the pieces together. Good chunks:
      # chunk 1 (~5 runs): CIFAR-10, all activations, one seed
      python pSigLU_ResNet18_CIFAR.py --datasets cifar10 --seeds 1 --seed-start 0
      # chunk 2: CIFAR-10, seed 1 ; chunk 3: CIFAR-10, seed 2 ; etc.
      python pSigLU_ResNet18_CIFAR.py --datasets cifar10 --seeds 1 --seed-start 1
      # then CIFAR-100 the same way:
      python pSigLU_ResNet18_CIFAR.py --datasets cifar100 --seeds 1 --seed-start 0
  Or chunk by activation subset (--activations laplu relu gelu). Re-running
  the SAME --outdir resumes; point every chunk at the same --outdir so all
  rows land in one raw_results.csv. Reduce --epochs (e.g. 40) if you need
  smaller chunks; keep it the SAME across activations for a fair comparison.

  USAGE:
      pip install torch torchvision numpy pandas scipy
      python pSigLU_ResNet18_CIFAR.py --datasets cifar10 --seeds 1 --seed-start 0
      python pSigLU_ResNet18_CIFAR.py --workers 0      # low-RAM / Windows
============================================================================
"""

import argparse, os, time, json
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Return a FRESH activation module (call once per usage site)."""
    name = name.lower()
    if name in ("laplu", "laplub1"):
        return LapLU(b=1.0, learnable=False)
    if name in ("laplub05", "laplub0p5"):            # b = 1/2: f''(0) = 2, the origin curvature of pSigLU(p = 2)
        return LapLU(b=0.5, learnable=False)
    if name == "laplub2":
        return LapLU(b=2.0, learnable=False)
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
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("swish", "silu"):
        return nn.SiLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"unknown activation: {name}")


ACTIVATIONS = ['psiglu', 'laplu', 'relu', 'elu', 'gelu', 'swish']


# ---------------------------------------------------------------------------
# 2.  ResNet-18 (CIFAR variant) with injectable activation
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, act_name):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = make_activation(act_name)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = make_activation(act_name)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, act_name, n_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem (3x3, no maxpool)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = make_activation(act_name)
        self.layer1 = self._make_layer(64, 2, 1, act_name)
        self.layer2 = self._make_layer(128, 2, 2, act_name)
        self.layer3 = self._make_layer(256, 2, 2, act_name)
        self.layer4 = self._make_layer(512, 2, 2, act_name)
        self.fc = nn.Linear(512, n_classes)

    def _make_layer(self, planes, n_blocks, stride, act_name):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, act_name))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# 3.  DATA
# ---------------------------------------------------------------------------
def get_loaders(dataset, batch_size, data_root, num_workers):
    import torchvision
    import torchvision.transforms as T
    if dataset == "cifar10":
        mean, std, n_classes, DS = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10, torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        mean, std, n_classes, DS = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100, torchvision.datasets.CIFAR100
    else:
        raise ValueError(dataset)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tr = DS(root=data_root, train=True, download=True, transform=train_tf)
    te = DS(root=data_root, train=False, download=True, transform=test_tf)
    pin = torch.cuda.is_available()
    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=pin, drop_last=False)
    vl = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin)
    return tl, vl, n_classes


# ---------------------------------------------------------------------------
# 4.  TRAIN / EVAL
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


def train_one(dataset, act_name, seed, epochs, lr, optimizer, device,
              data_root, batch_size, num_workers, epoch_ckpt_dir):
    set_seed(seed)
    tl, vl, n_classes = get_loaders(dataset, batch_size, data_root, num_workers)
    model = ResNet18(act_name, n_classes=n_classes).to(device)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    start_ep, best_acc, history = 0, 0.0, []
    ep_ckpt = None
    if epoch_ckpt_dir is not None:
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        ep_ckpt = os.path.join(epoch_ckpt_dir, f"epoch_{dataset}_{act_name}_seed{seed}.pt")
        if os.path.exists(ep_ckpt):
            try:
                ck = torch.load(ep_ckpt, map_location=device)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"]); start_ep = ck["epoch"]
                best_acc = ck.get("best_acc", 0.0); history = ck.get("history", [])
                print(f"      resumed {dataset}/{act_name}/seed{seed} from epoch {start_ep}")
            except Exception as e:
                print(f"      (could not load epoch checkpoint, starting over: {e})")
                start_ep, best_acc, history = 0, 0.0, []

    for ep in range(start_ep, epochs):
        model.train(); loss_sum = nb = 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            loss_sum += loss.item(); nb += 1
        sched.step()
        acc = evaluate(model, vl, device)
        best_acc = max(best_acc, acc)
        history.append({"epoch": ep, "train_loss": round(loss_sum / max(nb, 1), 5), "test_acc": round(acc, 5)})
        print(f"        epoch {ep+1}/{epochs}  loss={loss_sum/max(nb,1):.4f}  test_acc={acc:.4f}")
        if ep_ckpt is not None:
            tmp = ep_ckpt + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep + 1,
                        "best_acc": best_acc, "history": history}, tmp)
            os.replace(tmp, ep_ckpt)

    final_acc = evaluate(model, vl, device)
    if ep_ckpt is not None and os.path.exists(ep_ckpt):
        os.remove(ep_ckpt)
    return {"final_acc": final_acc, "best_acc": best_acc, "history": history}


# ---------------------------------------------------------------------------
# 5.  CHECKPOINT (job-level)
# ---------------------------------------------------------------------------
def _job_key(ds, act, seed):
    return f"{ds}|{act}|{seed}"


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
                rows.append(rec); done.add(_job_key(rec["dataset"], rec["activation"], rec["seed"]))
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
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--activations", nargs="+", default=ACTIVATIONS)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="index of first seed; use 0,1,2,... or 0,3,6,... to chunk distinct seeds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outdir", default="./results_resnet18_cifar")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.optimizer == "adam" and args.lr == 0.1:
        args.lr = 1e-3  # sane default if user switches to Adam without setting lr

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "checkpoint.jsonl")
    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path); print("(--fresh) removed existing checkpoint")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    jobs = [(d, a, s) for d in args.datasets for a in args.activations for s in seeds]
    done, all_rows = load_checkpoint(ckpt_path)
    todo = [j for j in jobs if _job_key(*j) not in done]
    print(f"device={device} | optimizer={args.optimizer} lr={args.lr} epochs={args.epochs}")
    print(f"jobs={len(jobs)} | already done={len(jobs)-len(todo)} | to run={len(todo)}")

    durations = [r["seconds"] for r in all_rows if "seconds" in r]
    for i, (ds, act, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {ds} / {act} / seed={seed}")
        t0 = time.time()
        r = train_one(ds, act, seed, args.epochs, args.lr, args.optimizer, device,
                      args.data_root, args.batch_size, args.workers,
                      epoch_ckpt_dir=os.path.join(args.outdir, "epoch_ckpt"))
        dt = time.time() - t0; durations.append(dt)
        rec = {"dataset": ds, "activation": act, "seed": seed,
               "final_acc": r["final_acc"], "best_acc": r["best_acc"],
               "seconds": round(dt, 1), "history": r["history"]}
        append_checkpoint(ckpt_path, rec); all_rows.append(rec)
        avg = sum(durations) / len(durations); eta = avg * (len(todo) - i)
        finish = datetime.now() + timedelta(seconds=eta)
        print(f"    -> final={r['final_acc']:.4f} best={r['best_acc']:.4f} "
              f"({_fmt_hms(dt)}; ETA ~{_fmt_hms(eta)}, ~{finish:%H:%M})")

    # ---- CSV / JSON ----
    try:
        import pandas as pd
        df = pd.DataFrame([{k: r[k] for k in ("dataset", "activation", "seed", "final_acc", "best_acc", "seconds")}
                           for r in all_rows])
        df.to_csv(os.path.join(args.outdir, "raw_results.csv"), index=False)
        for ds in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == ds].groupby("activation")["final_acc"].agg(["mean", "std", "count"])
            sub.to_csv(os.path.join(args.outdir, f"summary_{ds}.csv"))
    except Exception as e:
        print("pandas summary skipped:", e)

    import csv as _csv
    with open(os.path.join(args.outdir, "history_long.csv"), "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["dataset", "activation", "seed", "epoch", "train_loss", "test_acc"])
        for r in all_rows:
            for h in r.get("history", []):
                w.writerow([r["dataset"], r["activation"], r["seed"], h["epoch"], h["train_loss"], h["test_acc"]])

    print(f"\nDONE. Wrote raw_results.csv, history_long.csv, summary_<dataset>.csv to {args.outdir}/")
    print("Re-run the SAME command (same --outdir) to resume; --fresh restarts.")
    print("Send me the folder (or raw_results.csv + history_long.csv) and I'll plot + t-test it.")


if __name__ == "__main__":
    main()
