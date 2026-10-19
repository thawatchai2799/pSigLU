"""
pSigLU on Tiny ImageNet (100 epochs, 15 seeds)

============================================================================
  pSigLU (p=2) vs LapLU (b=1) vs baselines  --  Tiny ImageNet (64x64, 200 classes)
============================================================================
  Compares three infinite-support smooth activations on Tiny ImageNet, the standard bridge benchmark between CIFAR and full ImageNet: 64x64 RGB images across 200 classes, 100k train / 10k val. Tiny ImageNet is downloaded (~240 MB) on first run; the validation split is used as the test set. A 4-stage compact CNN is used.

  pSigLU is the single-parameter power-law-gated unit; LapLU is included as the
  exponential-gated reference point:
       LapLU(x) = x * Phi(x),   Phi the Laplace(0, b) CDF,
       LapLU(x) = relu(x) - 0.5 * |x| * exp(-|x| / b)   (branch-free),
  with slope 1/2 at the origin and INFINITE support (exponential
  tails; not compact). The default b = 1 is the Laplace(0,1) gate
  suggested in the original GELU paper.

  Activations evaluated:
       laplu (b = 1), swish (SiLU), gelu

  Default protocol: 1 dataset * 3 activations * 5 seeds * 100 epochs = 15 runs.
  Results are written to: ./results_tinyimagenet_e100/
  USAGE:
      pip install torch torchvision scipy numpy pandas matplotlib
      python psiglu_TinyImageNet_e100.py
      # on a low-RAM Windows machine, reduce DataLoader workers:
      python psiglu_TinyImageNet_e100.py --workers 0
============================================================================
"""

import argparse, os, time, json, math, zipfile, shutil, urllib.request
from datetime import datetime, timedelta
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# 1.  THE NEW ACTIVATION FUNCTION
# ---------------------------------------------------------------------------
class LapLU(nn.Module):
    """
    LapLU (Laplace Linear Unit):
        LapLU(x) = x * Phi(x),   where Phi is the Laplace(0, b) CDF.

    Branch-free closed form:
        LapLU(x) = relu(x) - 0.5 * |x| * exp(-|x| / b)
    Equivalent piecewise form:
        = 0.5 * x * exp(x / b)            for x <= 0
        = x * (1 - 0.5 * exp(-x / b))     for x >= 0

    The gate Phi is the CDF of a Laplace distribution centred at the origin
    with scale b: a smooth S-curve whose left half is a scaled exponential
    rising toward 1/2 and whose right half saturates toward 1, the two halves
    meeting at the origin with Phi(0) = 1/2. Multiplying by x gives a
    rectified unit minus an exponential bump localized at the origin.

    Properties:
        * slope 1/2 at the origin, for every b;
        * non-monotonic, with a minimum at x = -b of value -b/(2e);
        * INFINITE support: approaches 0 as x -> -inf and x as x -> +inf
          (exponential tails), unlike the compact-support RectLU family;
        * twice continuously differentiable (C^2) but not C^3: the Laplace
          density has a corner at the origin, but because that corner sits
          at the zero of the gating factor x, the activation gains one extra
          order of smoothness (second derivative 1/b on both sides at 0;
          third derivative jumps by 3/b^2 there).

    The default b = 1 is the Laplace(0,1) gate suggested (but not developed)
    in the original GELU paper.

    Args:
        b (float): Laplace scale (default 1.0). Larger b widens the soft
                   transition region; the origin slope stays 1/2.
        learnable (bool): if True, b becomes a trainable parameter. If
                   learned, keep b > 0 (e.g. via a softplus reparameterization)
                   so the gate does not degenerate.

    Closed-form derivative (for reference):
        LapLU'(x) = 0.5 * exp(x/b) * (1 + x/b)        for x <= 0
                  = 1 - 0.5 * exp(-x/b) * (1 - x/b)   for x >= 0
        LapLU'(0) = 1/2.
    """
    def __init__(self, b: float = 1.0, learnable: bool = False):
        super().__init__()
        t = torch.tensor(float(b))
        if learnable:
            self.b = nn.Parameter(t)
        else:
            self.register_buffer("b", t)

    def forward(self, x):
        ax = x.abs()
        # ReLU(x) - 0.5*|x|*exp(-|x|/b); equals 0.5*x*exp(x/b) for x<=0 and
        # x*(1 - 0.5*exp(-x/b)) for x>=0. No erf/trig; one exp + abs.
        return torch.relu(x) - 0.5 * ax * torch.exp(-ax / self.b)


# ---------------------------------------------------------------------------
# 2.  ACTIVATION REGISTRY  (baselines + ours)
# ---------------------------------------------------------------------------

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
    """Activation registry. LapLU is the scale-only (b) version; baselines are PyTorch built-ins."""
    name = name.lower()
    if name in ("laplu", "laplub1"):
        return LapLU(b=1.0,  learnable=False)        # default (Laplace(0,1))
    if name == "laplub025":
        return LapLU(b=0.25, learnable=False)
    if name == "laplub05":
        return LapLU(b=0.5,  learnable=False)
    if name == "laplub2":
        return LapLU(b=2.0,  learnable=False)
    if name == "laplub4":
        return LapLU(b=4.0,  learnable=False)
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
    raise ValueError(f"unknown activation: {name}")

ACTIVATIONS = ['psiglu', 'laplu', 'relu', 'elu', 'gelu', 'swish']


# ---------------------------------------------------------------------------
# 3.  MODELS
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Small MLP for MNIST (matches the shallow setup in the RectLU papers)."""
    def __init__(self, act_name, in_dim=784, hidden=128, n_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = make_activation(act_name)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = make_activation(act_name)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.out(x)


class SmallCNN(nn.Module):
    """Compact CNN for CIFAR-10 / CIFAR-100."""
    def __init__(self, act_name, n_classes=10):
        super().__init__()
        def act():
            return make_activation(act_name)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32), act(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), act(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), act(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), act(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), act(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TinyImageNetCNN(nn.Module):
    """Compact 4-stage CNN for 64x64 Tiny ImageNet (200 classes).

    Same design philosophy as SmallCNN but with one extra downsampling stage
    so that a 64x64 input is reduced 64 -> 32 -> 16 -> 8 -> 4 before the
    classifier (SmallCNN does 32 -> 16 -> 8 -> 4 for CIFAR's 32x32 input)."""
    def __init__(self, act_name, n_classes=200):
        super().__init__()
        def act():
            return make_activation(act_name)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   nn.BatchNorm2d(32),  act(),
            nn.Conv2d(32, 32, 3, padding=1),  nn.BatchNorm2d(32),  act(),
            nn.MaxPool2d(2),                                              # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  act(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  act(),
            nn.MaxPool2d(2),                                              # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), act(),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), act(),
            nn.MaxPool2d(2),                                              # 16 -> 8
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), act(),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), act(),
            nn.MaxPool2d(2),                                              # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), act(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 4.  DATA
# ---------------------------------------------------------------------------
TINYIMAGENET_URLS = [
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
]


def _download_tinyimagenet(zip_path):
    """Try each mirror in turn (https first). Uses a browser User-Agent and a
    timeout. Returns True on success, False if every URL failed."""
    for url in TINYIMAGENET_URLS:
        try:
            print(f"Downloading Tiny ImageNet (~240 MB) from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except Exception as e:
            print(f"  failed: {e}")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except OSError:
                    pass
    return False


def prepare_tinyimagenet(data_root):
    """Download (first run) and prepare Tiny ImageNet. Reorganizes the
    validation split into per-class folders so both splits can be read with
    torchvision ImageFolder. Returns (train_dir, val_dir). Idempotent."""
    root = os.path.join(data_root, "tiny-imagenet-200")
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir):
        os.makedirs(data_root, exist_ok=True)
        zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            if not _download_tinyimagenet(zip_path):
                raise RuntimeError(
                    "\n" + "=" * 70 + "\n"
                    "Could not download Tiny ImageNet automatically (network/DNS issue;\n"
                    "the cs231n.stanford.edu host is often unreachable).\n\n"
                    "MANUAL FIX -- download the file once in a web browser from:\n"
                    "    https://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                    "    (or any mirror, e.g. the Kaggle 'tiny-imagenet' dataset, or a\n"
                    "     Hugging Face mirror)\n\n"
                    "then put the downloaded file at exactly this path:\n"
                    "    " + os.path.abspath(zip_path) + "\n\n"
                    "OR unzip it yourself so that this folder exists:\n"
                    "    " + os.path.abspath(root) + "\n\n"
                    "and run the script again -- it will detect the file and continue.\n"
                    + "=" * 70)
        print("Extracting Tiny ImageNet ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_root)
    # Flatten train/<wnid>/images/*.JPEG into train/<wnid>/*.JPEG (only once)
    # so ImageFolder works regardless of torchvision version.
    if os.path.isdir(train_dir):
        for wnid in os.listdir(train_dir):
            cls = os.path.join(train_dir, wnid)
            img = os.path.join(cls, "images")
            if os.path.isdir(img):
                for fn in os.listdir(img):
                    shutil.move(os.path.join(img, fn), os.path.join(cls, fn))
                try:
                    os.rmdir(img)
                except OSError:
                    pass
    # Reorganize val/images/*.JPEG into val/<wnid>/*.JPEG (only once).
    val_images = os.path.join(val_dir, "images")
    val_ann = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isdir(val_images) and os.path.exists(val_ann):
        print("Reorganizing Tiny ImageNet validation set into class folders ...")
        with open(val_ann) as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, wnid)
                os.makedirs(cls_dir, exist_ok=True)
                s = os.path.join(val_images, fname)
                if os.path.exists(s):
                    shutil.move(s, os.path.join(cls_dir, fname))
        try:
            os.rmdir(val_images)
        except OSError:
            pass
    return train_dir, val_dir


def get_loaders(dataset, batch_size=128, data_root="./data", num_workers=0):
    if dataset == "mnist":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST(data_root, train=True,  download=True, transform=tf)
        test  = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=tf)
        n_classes, model_kind = 10, "mlp"
    elif dataset == "cifar10":
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        tf_test  = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        train = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=tf_train)
        test  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=tf_test)
        n_classes, model_kind = 10, "cnn"
    elif dataset == "cifar100":
        tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize((0.5071,0.4865,0.4409),(0.2673,0.2564,0.2762))])
        tf_test  = T.Compose([T.ToTensor(), T.Normalize((0.5071,0.4865,0.4409),(0.2673,0.2564,0.2762))])
        train = torchvision.datasets.CIFAR100(data_root, train=True,  download=True, transform=tf_train)
        test  = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=tf_test)
        n_classes, model_kind = 100, "cnn"
    elif dataset == "tinyimagenet":
        # Tiny ImageNet: 64x64 RGB, 200 classes, 100k train / 10k val images.
        train_dir, val_dir = prepare_tinyimagenet(data_root)
        mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        tf_train = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(mean, std)])
        tf_test  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.ImageFolder(train_dir, transform=tf_train)
        test  = torchvision.datasets.ImageFolder(val_dir,   transform=tf_test)
        n_classes, model_kind = 200, "cnn64"
    else:
        raise ValueError(dataset)

    # DataLoader settings tuned for systems with limited RAM (~8 GB) and a
    # 2-core CPU (e.g. Intel i3). On Windows + GPU these settings are usually
    # the safest and often the fastest combination:
    #   num_workers=0     -> no extra processes (saves RAM, avoids Windows
    #                        worker spawn overhead). Set to 2 if your CPU is
    #                        much stronger and RAM is comfortable.
    #   pin_memory=False  -> disable pinned (locked) RAM, which can starve a
    #                        small-RAM system. The speed cost is small.
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test,  batch_size=128, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader, n_classes, model_kind


def build_model(model_kind, act_name, n_classes):
    if model_kind == "mlp":
        return MLP(act_name, n_classes=n_classes)
    if model_kind == "cnn64":
        return TinyImageNetCNN(act_name, n_classes=n_classes)
    return SmallCNN(act_name, n_classes=n_classes)


# ---------------------------------------------------------------------------
# 5.  TRAIN / EVAL
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total

def train_one(dataset, act_name, seed, epochs, lr, device, epoch_ckpt_dir=None,
              batch_size=128, num_workers=0, data_root="./data"):
    set_seed(seed)
    train_loader, test_loader, n_classes, model_kind = get_loaders(
        dataset, batch_size=batch_size, data_root=data_root, num_workers=num_workers)
    model = build_model(model_kind, act_name, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    # ---- epoch-level checkpoint (lets a single run resume mid-training) ----
    start_ep = 0
    best_acc = 0.0
    ep_ckpt = None
    history = []   # per-epoch records: {epoch, train_loss, test_acc}
    if epoch_ckpt_dir is not None:
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        ep_ckpt = os.path.join(epoch_ckpt_dir, f"epoch_{dataset}_{act_name}_seed{seed}.pt")
        if os.path.exists(ep_ckpt):
            try:
                state = torch.load(ep_ckpt, map_location=device, weights_only=True)
                model.load_state_dict(state["model"])
                opt.load_state_dict(state["opt"])
                sched.load_state_dict(state["sched"])
                start_ep = state["epoch"] + 1
                best_acc = state["best_acc"]
                history = state.get("history", [])
                print(f"      (resumed {act_name}/seed{seed} from epoch {start_ep}/{epochs})")
            except Exception as e:
                print(f"      (could not load epoch checkpoint, starting run over: {e})")
                start_ep, best_acc = 0, 0.0; history = []

    for ep in range(start_ep, epochs):
        model.train()
        _loss_sum, _n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            _loss_sum += loss.item(); _n_batches += 1
        sched.step()
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        history.append({"epoch": ep, "train_loss": round(_loss_sum / max(_n_batches, 1), 5), "test_acc": round(acc, 5)})
        # save epoch checkpoint (crash-safe: write to temp then rename)
        if ep_ckpt is not None:
            tmp = ep_ckpt + ".tmp"
            torch.save({"model":model.state_dict(), "opt":opt.state_dict(),
                        "sched":sched.state_dict(), "epoch":ep, "best_acc":best_acc, "history":history}, tmp)
            os.replace(tmp, ep_ckpt)   # atomic on same filesystem
    final_acc = evaluate(model, test_loader, device)
    # run finished: remove the per-epoch checkpoint to save space
    if ep_ckpt is not None and os.path.exists(ep_ckpt):
        os.remove(ep_ckpt)
    return {"final_acc": final_acc, "best_acc": best_acc, "history": history}


# ---------------------------------------------------------------------------
# 6.  STATISTICS
# ---------------------------------------------------------------------------
def pairwise_ttests(results_by_act, alpha=0.05):
    """results_by_act: {act: [acc_seed1, acc_seed2, ...]}  -> list of comparison dicts"""
    try:
        from scipy import stats
    except ImportError:
        print("scipy not installed; skipping t-tests"); return []
    acts = list(results_by_act.keys())
    out = []
    for i in range(len(acts)):
        for j in range(i+1, len(acts)):
            a, b = acts[i], acts[j]
            va, vb = np.array(results_by_act[a]), np.array(results_by_act[b])
            tstat, p = stats.ttest_ind(va, vb, equal_var=False)
            if p < alpha:
                winner = a if va.mean() > vb.mean() else b
                verdict = f"{a} > {b}" if winner == a else f"{b} > {a}"
            else:
                verdict = "no_consistent_difference"
            out.append({"pair": f"{a} vs {b}",
                        "mean_a": round(va.mean(),4), "mean_b": round(vb.mean(),4),
                        "p_value": float(f"{p:.4g}"), "verdict": verdict})
    return out


# ---------------------------------------------------------------------------
# 7.  CHECKPOINTING  (resume after a crash / power loss)
# ---------------------------------------------------------------------------
# Strategy: every finished (dataset, activation, seed) run is appended
# immediately to a JSON-lines checkpoint file. On startup we read that file
# and skip any run already completed, so an interrupted job resumes from the
# nearest point instead of restarting from zero.

def _job_key(dataset, act, seed):
    return f"{dataset}|{act}|{seed}"

def load_checkpoint(ckpt_path):
    """Return (done_dict, rows_list). done_dict maps job_key -> result row."""
    done = {}
    rows = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # a half-written final line (e.g. power loss mid-write) is ignored
                    continue
                done[_job_key(rec["dataset"], rec["activation"], rec["seed"])] = rec
                rows.append(rec)
    return done, rows

def append_checkpoint(ckpt_path, rec):
    """Append one completed run and flush to disk immediately (crash-safe)."""
    with open(ckpt_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())   # force write to disk so a power loss right after won't lose it


def _fmt_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# 8.  MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=['tinyimagenet'])
    ap.add_argument("--activations", nargs="+", default=ACTIVATIONS)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="index of the first seed; use 0,5,10 across runs for 15 distinct seeds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers (Tiny ImageNet benefits from >=2; set 0 on low-RAM Windows)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--outdir", default="./results_tinyimagenet_e100")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start over")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "checkpoint.jsonl")
    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("(--fresh) removed existing checkpoint; starting from scratch")

    print(f"device = {device}")

    # ---- build the full list of jobs in execution order ----
    jobs = []
    for dataset in args.datasets:
        for act in args.activations:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                jobs.append((dataset, act, seed))
    total_jobs = len(jobs)

    # ---- load any previous progress ----
    done, all_rows = load_checkpoint(ckpt_path)
    done = {k: v for k, v in done.items()}
    # keep only rows that belong to the current run configuration
    n_already = sum(1 for (d, a, s) in jobs if _job_key(d, a, s) in done)
    if n_already > 0:
        print(f"Resuming: {n_already}/{total_jobs} runs already completed "
              f"(found in {ckpt_path}). Skipping those and continuing.")
    else:
        print(f"Starting fresh: {total_jobs} runs to perform.")

    # ---- timing bookkeeping for ETA ----
    durations = [rec["seconds"] for rec in all_rows if "seconds" in rec]  # reuse past timings for ETA

    completed = n_already
    for idx, (dataset, act, seed) in enumerate(jobs):
        key = _job_key(dataset, act, seed)
        if key in done:
            continue  # already finished in a previous (possibly interrupted) run

        t0 = time.time()
        r = train_one(dataset, act, seed, args.epochs, args.lr, device,
                      epoch_ckpt_dir=os.path.join(args.outdir, "epoch_ckpt"),
                      batch_size=args.batch_size, num_workers=args.workers,
                      data_root=args.data_root)
        dt = time.time() - t0
        durations.append(dt)

        rec = {"dataset": dataset, "activation": act, "seed": seed,
               "final_acc": r["final_acc"], "best_acc": r["best_acc"],
               "seconds": round(dt, 1), "history": r.get("history", [])}
        append_checkpoint(ckpt_path, rec)   # <-- saved to disk immediately
        all_rows.append(rec)
        done[key] = rec
        completed += 1

        # ---- progress + current time + ETA ----
        now = datetime.now()
        avg = sum(durations) / len(durations)
        remaining_jobs = total_jobs - completed
        eta_seconds = remaining_jobs * avg
        finish_time = now + timedelta(seconds=eta_seconds)

        # what runs are still pending (next few)
        pending = [(d, a, s) for (d, a, s) in jobs
                   if _job_key(d, a, s) not in done]
        next_preview = ", ".join(f"{d}/{a}/seed{s}" for (d, a, s) in pending[:3])
        if len(pending) > 3:
            next_preview += f", ... (+{len(pending) - 3} more)"
        if not pending:
            next_preview = "none - this was the last run"

        print(f"[{now:%Y-%m-%d %H:%M:%S}] "
              f"DONE {dataset}/{act}/seed{seed}: "
              f"final={r['final_acc']:.4f} best={r['best_acc']:.4f} (took {_fmt_hms(dt)})")
        print(f"    progress : {completed}/{total_jobs} runs "
              f"({100.0*completed/total_jobs:.1f}%)")
        print(f"    next up  : {next_preview}")
        print(f"    time left: ~{_fmt_hms(eta_seconds)}  "
              f"(est. finish {finish_time:%Y-%m-%d %H:%M:%S})")

    # ---- aggregate results from all_rows (works whether resumed or fresh) ----
    summary = {}
    for dataset in args.datasets:
        summary[dataset] = {}
        results_by_act = {}
        for act in args.activations:
            accs = [rec["final_acc"] for rec in all_rows
                    if rec["dataset"] == dataset and rec["activation"] == act]
            if not accs:
                continue
            results_by_act[act] = accs
            m, s = float(np.mean(accs)), float(np.std(accs))
            summary[dataset][act] = {"mean": round(m, 4), "std": round(s, 4)}
            print(f"  => {dataset}/{act}: {m:.4f} +/- {s:.4f}  (n={len(accs)})")
        summary[dataset]["_ttests"] = pairwise_ttests(results_by_act)

    # ---- write CSV / JSON outputs ----
    try:
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in all_rows])
        df.to_csv(os.path.join(args.outdir, "raw_results.csv"), index=False)
        for dataset in args.datasets:
            sub = df[df.dataset == dataset]
            if len(sub) == 0:
                continue
            sub = (sub.groupby("activation")["final_acc"]
                   .agg(["mean", "std"]).sort_values("mean", ascending=False))
            sub.to_csv(os.path.join(args.outdir, f"summary_{dataset}.csv"))
            print(f"\n=== {dataset} ranking ===\n{sub}")
    except ImportError:
        print("pandas not installed; raw rows saved as JSON instead")
        with open(os.path.join(args.outdir, "raw_results.json"), "w") as f:
            json.dump(all_rows, f, indent=2)

    # ---- per-epoch learning-curve logs ----
    import csv as _csv
    with open(os.path.join(args.outdir, "history.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps({"dataset": r["dataset"], "activation": r["activation"],
                                "seed": r["seed"], "history": r.get("history", [])}) + "\n")
    with open(os.path.join(args.outdir, "history_long.csv"), "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["dataset", "activation", "seed", "epoch", "train_loss", "test_acc"])
        for r in all_rows:
            for h in r.get("history", []):
                w.writerow([r["dataset"], r["activation"], r["seed"], h["epoch"], h["train_loss"], h["test_acc"]])
    for dataset in args.datasets:
        rows_d = [r for r in all_rows if r["dataset"] == dataset and r.get("history")]
        if not rows_d:
            continue
        acts = sorted({r["activation"] for r in rows_d})
        n_ep = max(h["epoch"] for r in rows_d for h in r["history"]) + 1
        agg = {a: [[] for _ in range(n_ep)] for a in acts}
        for r in rows_d:
            for h in r["history"]:
                agg[r["activation"]][h["epoch"]].append(h["test_acc"])
        with open(os.path.join(args.outdir, f"curves_{dataset}.csv"), "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["epoch"] + acts)
            for ep in range(n_ep):
                row = [ep]
                for a in acts:
                    vals = agg[a][ep]
                    row.append(round(sum(vals) / len(vals), 5) if vals else "")
                w.writerow(row)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll done at {datetime.now():%Y-%m-%d %H:%M:%S}. Results in: {args.outdir}/")
    print("Files: checkpoint.jsonl, raw_results.csv, summary_<dataset>.csv, summary.json, history.jsonl, history_long.csv, curves_<dataset>.csv")
    print("Tip: re-running the same command resumes from the checkpoint; add --fresh to start over.")


if __name__ == "__main__":
    main()
