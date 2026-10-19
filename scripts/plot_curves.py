#!/usr/bin/env python3
"""
Plot learning curves from the per-epoch logs written by the experiment scripts.

After a run, each results directory contains `history_long.csv` with columns
    dataset, activation, seed, epoch, train_loss, test_acc
This script reads that file and draws, for every dataset, the mean test
accuracy per epoch for each activation (with a shaded +/- 1 std band across
seeds). It can also plot the training loss.

USAGE:
    python plot_curves.py --dir results_main_mnist_cifar
    python plot_curves.py --dir results_main_mnist_cifar --metric train_loss
    python plot_curves.py --dir results_main_mnist_cifar --out figs

Outputs <out>/learning_curve_<dataset>_<metric>.png and .pdf at 300 dpi.
"""
import argparse
import csv
import math
import os
from collections import defaultdict


def read_history(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "dataset": r["dataset"], "activation": r["activation"],
                "seed": int(r["seed"]), "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]), "test_acc": float(r["test_acc"]),
            })
    return rows


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results directory containing history_long.csv")
    ap.add_argument("--metric", default="test_acc", choices=["test_acc", "train_loss"])
    ap.add_argument("--out", default=None, help="output directory (default: same as --dir)")
    ap.add_argument("--no-band", action="store_true", help="do not draw the +/- std band")
    args = ap.parse_args()

    hist_path = os.path.join(args.dir, "history_long.csv")
    if not os.path.exists(hist_path):
        raise SystemExit(f"{hist_path} not found. Run an experiment script first (it writes history_long.csv).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 300,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    out = args.out or args.dir
    os.makedirs(out, exist_ok=True)
    rows = read_history(hist_path)

    # group: dataset -> activation -> epoch -> [values]
    G = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        G[r["dataset"]][r["activation"]][r["epoch"]].append(r[args.metric])

    ylabel = "test accuracy (%)" if args.metric == "test_acc" else "training loss"
    scale = 100.0 if args.metric == "test_acc" else 1.0

    for dataset, acts in sorted(G.items()):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        # stable colour per activation; LapLU emphasised
        palette = ["#534AB7", "#1D9E75", "#BA7517", "#A32D2D", "#185FA5", "#888780"]
        for i, act in enumerate(sorted(acts)):
            ep = sorted(acts[act])
            means, stds = [], []
            for e in ep:
                m, s = mean_std(acts[act][e])
                means.append(m * scale); stds.append(s * scale)
            lw = 2.4 if act.startswith("laplu") else 1.5
            c = "#534AB7" if act.startswith("laplu") else palette[(i + 1) % len(palette)]
            ax.plot(ep, means, label=act, color=c, lw=lw)
            if not args.no_band:
                lo = [m - s for m, s in zip(means, stds)]
                hi = [m + s for m, s in zip(means, stds)]
                ax.fill_between(ep, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset}: {ylabel} vs epoch (mean over seeds)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        base = os.path.join(out, f"learning_curve_{dataset}_{args.metric}")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", base + ".png", "and .pdf")


if __name__ == "__main__":
    main()
