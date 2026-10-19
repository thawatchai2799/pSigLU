"""The nine figures of the pSigLU manuscript.
usage: python make_psiglu_figures_all.py <main_results_root> <extra_results_root>
  main_results_root  : folder holding results_psiglu_main, results_psiglu_p, results_psiglu_p_fm,
                       results_psiglu_fm_svhn, results_psiglu_sobolev_lam1, results_psiglu_eikonal, results_psiglu_resnet
  extra_results_root : folder holding results_2026-10-08_resnet_p12, results_2026-10-08_resnet_p15,
                       results_2026-10-08_eikonal_psweep, results_2026-10-08_sobolev_lam1_psweep
Writes figures/Figure*_<name>.png (600 dpi, 6.1 in wide) and the matching PDFs.
ASCII-only source; non-ASCII glyphs in labels are written as \\u escapes."""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import erf
from matplotlib.ticker import NullLocator
R1, R2 = sys.argv[1], sys.argv[2]
os.makedirs("figures", exist_ok=True)
W = 6.1   # width of the figure as placed in the manuscript (inches); fonts below are therefore the print sizes
RC = {"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "lines.linewidth": 1.3}
C = {"psiglu": "#534AB7", "laplu": "#1D9E75", "gelu": "#BA7517", "swish": "#888780", "relu": "#A32D2D", "elu": "#C77DB0"}
LAB = {"psiglu": "pSigLU (p = 2)", "laplu": "LapLU (b = 1)", "gelu": "GELU", "swish": "Swish", "elu": "ELU", "relu": "ReLU"}
ORDER = ["psiglu", "laplu", "gelu", "swish", "elu", "relu"]
PS = [(1.2, "psiglup12"), (1.5, "psiglup15"), (2, "psiglu"), (3, "psiglup3"), (4, "psiglup4"), (6, "psiglup6")]
XT = [1.2, 1.5, 2, 3, 4, 6]; XL = ["1.2", "1.5", "2", "3", "4", "6"]


def save(fig, name):
    fig.savefig("figures/%s.png" % name, dpi=600); fig.savefig("figures/%s.pdf" % name); plt.close(fig)


def load(root, d):
    rows = []
    for line in open(os.path.join(root, d, "checkpoint.jsonl")):
        if line.strip():
            j = json.loads(line); j.pop("history", None); rows.append(j)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ closed forms
def psiglu(x, p=2.0): return np.maximum(x, 0) - 0.5 * np.abs(x) * (1 + np.abs(x)) ** (-p)
def laplu(x, b=1.0): return np.maximum(x, 0) - 0.5 * np.abs(x) * np.exp(-np.abs(x) / b)
def gelu(x): return 0.5 * x * (1 + erf(x / np.sqrt(2)))
def swish(x): return x / (1 + np.exp(-x))
def relu(x): return np.maximum(x, 0)
def elu(x): return np.where(x > 0, x, np.expm1(x))
def dens(x, p): return 0.5 * p * (1 + np.abs(x)) ** (-p - 1)
def cdf(x, p):
    a = np.abs(x); g = 0.5 * (1 + a) ** (-p); return np.where(x <= 0, g, 1 - g)
def d1(x, p):
    v = (x > 0).astype(float) - 0.5 * np.sign(x) * (1 + np.abs(x)) ** (-p - 1) * (1 - (p - 1) * np.abs(x))
    return np.where(x == 0, 0.5, v)          # f'(0) = 1/2 (Proposition 2); sign(0) = 0 would give a spurious 0
def d2(x, p): return 0.5 * p * (1 + np.abs(x)) ** (-p - 2) * (2 - (p - 1) * np.abs(x))


x = np.linspace(-6, 4, 2001)
with plt.rc_context(RC):
    # ---------------------------------------------------------------- construction (p = 2)
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.05))
    ax[0].plot(x, dens(x, 2), color=C["psiglu"]); ax[0].set_title("(a) density q\u2082(x)"); ax[0].set_xlabel("x")
    ax[1].plot(x, cdf(x, 2), color=C["psiglu"]); ax[1].axhline(0.5, ls=":", color="k", lw=0.8)
    ax[1].set_title("(b) gate G\u2082(x) (CDF)"); ax[1].set_xlabel("x")
    ax[2].plot(x, psiglu(x, 2), color=C["psiglu"], label="pSigLU (p = 2)"); ax[2].plot(x, relu(x), "--", color=C["relu"], lw=1, label="ReLU")
    ax[2].set_title("(c) activation x\u00b7G\u2082(x)"); ax[2].set_xlabel("x"); ax[2].legend(frameon=False)
    fig.tight_layout(); save(fig, "Figure1_construction")

    # ---------------------------------------------------------------- the family and the tails
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    for f, k in [(psiglu, "psiglu"), (laplu, "laplu"), (gelu, "gelu"), (swish, "swish"), (elu, "elu"), (relu, "relu")]:
        ax[0].plot(x, f(x), color=C[k], label=LAB[k], ls="--" if k == "relu" else "-")
    ax[0].set_xlim(-6, 4); ax[0].set_ylim(-1.1, 4); ax[0].set_title("(a) activations"); ax[0].set_xlabel("x"); ax[0].legend(frameon=False, ncol=2)
    xn = np.linspace(-30, -0.5, 600)
    for f, k in [(psiglu, "psiglu"), (laplu, "laplu"), (gelu, "gelu"), (swish, "swish")]:
        ax[1].plot(-xn, -f(xn), color=C[k], label=LAB[k])
    ax[1].set_yscale("log"); ax[1].set_xscale("log"); ax[1].set_ylim(1e-8, 1); ax[1].set_xlabel("|x| (x < 0)"); ax[1].set_ylabel("\u2212f(x)")
    ax[1].set_title("(b) negative tails (log\u2013log)"); ax[1].legend(frameon=False)
    fig.tight_layout(); save(fig, "Figure2_family")

    # ---------------------------------------------------------------- derivatives
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.45))
    for p, c in zip([1.5, 2, 4], plt.cm.viridis(np.linspace(0.2, 0.8, 3))):
        ax[0].plot(x, d1(x, p), color=c, label="p = %g" % p); ax[1].plot(x, d2(x, p), color=c, label="p = %g" % p)
    ax[0].axhline(1, ls=":", color="k", lw=0.8); ax[0].set_xlim(-6, 4); ax[0].set_title("(a) first derivative"); ax[0].set_xlabel("x"); ax[0].legend(frameon=False)
    ax[1].axhline(0, ls=":", color="k", lw=0.8); ax[1].set_xlim(-6, 4); ax[1].set_title("(b) second derivative (continuous; f\u2033(0) = p)"); ax[1].set_xlabel("x"); ax[1].legend(frameon=False)
    fig.tight_layout(); save(fig, "Figure3_derivatives")

    # ---------------------------------------------------------------- effect of p
    xw = np.linspace(-10, 3, 2601)
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.45))
    for p, c in zip([1.2, 1.5, 2, 3, 4, 6], plt.cm.viridis(np.linspace(0.1, 0.9, 6))):
        ax[0].plot(xw, psiglu(xw, p), color=c, label="p = %g" % p)
    ax[0].plot(xw, relu(xw), "--", color="k", lw=0.9, label="ReLU (p \u2192 \u221e)"); ax[0].set_xlim(-10, 1.5); ax[0].set_ylim(-0.33, 1.0)
    ax[0].set_title("(a) pSigLU for several p"); ax[0].set_xlabel("x"); ax[0].legend(frameon=False, ncol=2)
    pp = np.linspace(1.05, 12, 500)
    ax[1].plot(pp, 0.5 * (pp - 1) ** (pp - 1) / pp ** pp, color=C["psiglu"], label="depth of dip = \u2016pSigLU\u209a \u2212 ReLU\u2016\u221e")
    ax[1].plot(pp, 1 / (pp - 1), color=C["laplu"], label="location of dip 1/(p \u2212 1)")
    ax[1].plot(pp, 2 / (pp - 1), color=C["gelu"], label="curvature zero 2/(p \u2212 1)")
    ax[1].set_ylim(0, 3); ax[1].set_xlabel("p"); ax[1].set_title("(b) shape constants (Proposition 3)"); ax[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "Figure4_effect_of_p")

    # ---------------------------------------------------------------- p-ablation (LapLU from the same sweep)
    abl = pd.concat([load(R1, "results_psiglu_p"), load(R1, "results_psiglu_p_fm")])
    cls = pd.concat([load(R1, "results_psiglu_main"), load(R1, "results_psiglu_fm_svhn")])
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.95))
    for a, ds in zip(ax, ["cifar100", "cifar10", "svhn"]):
        xs, ms, ss = [], [], []
        for p, k in PS:
            v = abl[(abl.activation == k) & (abl.dataset == ds)].final_acc.values * 100; xs.append(p); ms.append(v.mean()); ss.append(v.std(ddof=1))
        a.errorbar(xs, ms, yerr=ss, marker="o", ms=3.5, color=C["psiglu"], capsize=2, label="pSigLU (15 seeds)")
        v = abl[(abl.activation == "laplu") & (abl.dataset == ds)].final_acc.values * 100
        a.axhline(v.mean(), ls="--", color=C["laplu"], lw=1.1, label="LapLU (b = 1), same sweep")
        for b, ls in [("relu", ":"), ("elu", "-."), ("swish", (0, (3, 1, 1, 1)))]:
            v = cls[(cls.activation == b) & (cls.dataset == ds)].final_acc.values * 100
            a.axhline(v.mean(), ls=ls, color=C[b], lw=1.1, label=LAB[b] + ", main comparison")
        a.set_xscale("log"); a.set_xticks(XT); a.set_xticklabels(XL, fontsize=6.5); a.xaxis.set_minor_locator(NullLocator()); a.set_xlabel("tail index p")
        a.set_title({"cifar100": "(a) CIFAR-100", "cifar10": "(b) CIFAR-10", "svhn": "(c) SVHN"}[ds])
    ax[0].set_ylabel("test accuracy (%)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, columnspacing=1.5, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); save(fig, "Figure5_p_ablation")

    # ---------------------------------------------------------------- CIFAR-100 learning curves
    h = pd.read_csv(os.path.join(R1, "results_psiglu_main", "history_long.csv")); h = h[h.dataset == "cifar100"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    for a in ORDER:
        g = h[h.activation == a].groupby("epoch")
        m = g.test_acc.mean() * 100; s = g.test_acc.std(ddof=1) * 100
        ax[0].plot(m.index, m.values, color=C[a], label=LAB[a]); ax[0].fill_between(m.index, (m - s).values, (m + s).values, color=C[a], alpha=0.15)
        m2 = g.train_loss.mean(); ax[1].plot(m2.index, m2.values, color=C[a], label=LAB[a])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test accuracy (%)"); ax[0].set_title("(a) CIFAR-100 test accuracy"); ax[0].set_ylim(40, 68); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training loss"); ax[1].set_title("(b) CIFAR-100 training loss"); ax[1].set_yscale("log")
    ax[1].set_yticks([0.3, 0.5, 1, 2, 5]); ax[1].set_yticklabels(["0.3", "0.5", "1", "2", "5"]); ax[1].minorticks_off()
    fig.tight_layout(); save(fig, "Figure6_cifar100_curves")

    # ---------------------------------------------------------------- Sobolev at lambda = 1
    hist = pd.read_csv(os.path.join(R1, "results_psiglu_sobolev_lam1", "history_long.csv")); res = load(R1, "results_psiglu_sobolev_lam1")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    sub = hist[hist.task == "sincos2d"]
    for a in ORDER:
        med = sub[sub.activation == a].groupby("step").grad_mse.median().sort_index(); med = med[med > 0]
        ax[0].plot(med.index.values, med.values, color=C[a], label=LAB[a])
    ax[0].set_yscale("log"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("gradient-matching MSE (median)"); ax[0].set_title("(a) convergence on the 2-D target"); ax[0].set_ylim(top=60); ax[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    tasks = ["sin1d", "gauss1d", "sincos2d"]; xb = np.arange(3); w = 0.13
    for i, a in enumerate(ORDER):
        m = [res[(res.task == t) & (res.activation == a)].grad_mse.mean() for t in tasks]; s = [res[(res.task == t) & (res.activation == a)].grad_mse.std(ddof=1) for t in tasks]
        ax[1].bar(xb + (i - 2.5) * w, m, w, yerr=s, color=C[a], label=LAB[a], error_kw=dict(ecolor="black", elinewidth=0.7, capsize=1.5))
    ax[1].set_yscale("log"); ax[1].set_xticks(xb); ax[1].set_xticklabels(["sin(3x)\n1-D", "exp(\u2212x\u00b2)\n1-D", "sin2x\u00b7cos2y\n2-D"]); ax[1].set_ylabel("final gradient-matching MSE"); ax[1].set_title("(b) final error (20 seeds, log scale)")
    fig.tight_layout(); save(fig, "Figure7_sobolev")

    # ---------------------------------------------------------------- Eikonal circle
    e = load(R1, "results_psiglu_eikonal"); fig, ax = plt.subplots(1, 3, figsize=(W, 2.45))
    for a_, (col, sc, title) in zip(ax, [("sdf_mae", 1e3, "(a) true SDF error (\u00d710\u207b\u00b3)"), ("eikonal_res", 1e4, "(b) Eikonal residual (\u00d710\u207b\u2074)"), ("surface_err", 1e4, "(c) boundary error (\u00d710\u207b\u2074)")]):
        for i, a in enumerate(ORDER):
            v = e[e.activation == a][col].values * sc
            a_.bar(i, v.mean(), yerr=v.std(ddof=1), color=C[a], error_kw=dict(ecolor="black", elinewidth=0.7, capsize=1.5)); a_.scatter([i] * len(v), v, s=5, color="k", alpha=0.5, zorder=3)
        a_.set_xticks(range(6)); a_.set_xticklabels(["pSigLU", "LapLU", "GELU", "Swish", "ELU", "ReLU"], rotation=30, fontsize=7); a_.set_title(title)
    fig.tight_layout(); save(fig, "Figure8_eikonal")

    # ---------------------------------------------------------------- p across tasks
    fig, ax = plt.subplots(2, 2, figsize=(W, 5.9)); ax = ax.ravel()
    # (a) compact network, CIFAR-100: pSigLU sweep and LapLU from the same sweep; ReLU from the main comparison
    a = ax[0]; xs, ms, ss = [], [], []
    for p, k in PS:
        v = abl[(abl.activation == k) & (abl.dataset == "cifar100")].final_acc.values * 100; xs.append(p); ms.append(v.mean()); ss.append(v.std(ddof=1))
    a.errorbar(xs, ms, yerr=ss, marker="o", ms=3.5, color=C["psiglu"], capsize=2, label="pSigLU")
    v = abl[(abl.activation == "laplu") & (abl.dataset == "cifar100")].final_acc.values * 100
    a.axhline(v.mean(), ls="--", color=C["laplu"], lw=1.1, label="LapLU (same sweep)")
    v = cls[(cls.activation == "relu") & (cls.dataset == "cifar100")].final_acc.values * 100
    a.axhline(v.mean(), ls=":", color=C["relu"], lw=1.1, label="ReLU (main comparison)")
    a.set_title("(a) compact network, CIFAR-100"); a.set_ylabel("test accuracy (%)")
    # (b) ResNet-18 relative to p = 2
    res = load(R1, "results_psiglu_resnet"); r12 = load(R2, "results_2026-10-08_resnet_p12"); r15 = load(R2, "results_2026-10-08_resnet_p15")
    a = ax[1]
    for ds, mk, lab, ls in [("cifar10", "o", "CIFAR-10", "-"), ("cifar100", "s", "CIFAR-100", "--")]:
        ref = res[(res.activation == "psiglu") & (res.dataset == ds)].final_acc.values * 100
        xs, ms, ss = [], [], []
        for p, df in [(1.2, r12), (1.5, r15), (2, res[res.activation == "psiglu"])]:
            v = df[df.dataset == ds].final_acc.values * 100
            if len(v) == 0: continue
            xs.append(p); ms.append(v.mean() - ref.mean()); ss.append(v.std(ddof=1))
        a.errorbar(xs, ms, yerr=ss, marker=mk, ms=3.5, color=C["psiglu"], capsize=2, label="pSigLU, " + lab, ls=ls)
        v = res[(res.activation == "laplu") & (res.dataset == ds)].final_acc.values * 100
        a.axhline(v.mean() - ref.mean(), ls=ls, color=C["laplu"], lw=1.1, label="LapLU, " + lab)
    a.axhline(0, color="k", lw=0.6)
    a.set_title("(b) ResNet-18, relative to p = 2 (p \u2264 2)"); a.set_ylabel("accuracy \u2212 accuracy at p = 2 (pp)")
    # (c) Eikonal circle
    eik = load(R1, "results_psiglu_eikonal"); ep = load(R2, "results_2026-10-08_eikonal_psweep")
    a = ax[2]
    for c, sc, mk, lab, ls in [("sdf_mae", 1e3, "o", "pSigLU, SDF error (\u00d710\u207b\u00b3)", "-"), ("eikonal_res", 1e4, "s", "pSigLU, residual (\u00d710\u207b\u2074)", "--")]:
        xs, ms, ss = [], [], []
        for p, k in PS:
            src = eik if k == "psiglu" else ep; v = src[src.activation == k][c].values * sc; xs.append(p); ms.append(v.mean()); ss.append(v.std(ddof=1))
        a.errorbar(xs, ms, yerr=ss, marker=mk, ms=3.5, color=C["psiglu"], capsize=2, label=lab, ls=ls)
    for c, sc, ls, lab in [("sdf_mae", 1e3, "-", "LapLU, SDF error"), ("eikonal_res", 1e4, "--", "LapLU, residual")]:
        v = eik[eik.activation == "laplu"][c].values * sc; a.axhline(v.mean(), ls=ls, color=C["laplu"], lw=1.1, label=lab)
    a.set_yscale("log"); a.set_title("(c) Eikonal PINN, circle"); a.set_ylabel("error (log scale)"); a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    # (d) Sobolev lambda = 1
    sob = load(R1, "results_psiglu_sobolev_lam1"); sp = load(R2, "results_2026-10-08_sobolev_lam1_psweep")
    a = ax[3]
    for task, mk, al, ls in [("sin1d", "o", 0.55, (0, (1, 1))), ("gauss1d", "^", 0.75, (0, (3, 1, 1, 1))), ("sincos2d", "s", 1.0, "--")]:
        xs, ms = [], []
        for p, k in PS:
            src = sob if k == "psiglu" else sp; v = src[(src.activation == k) & (src.task == task)].grad_mse.values; xs.append(p); ms.append(max(v.mean(), 1e-8))
        a.plot(xs, ms, marker=mk, ms=3.5, color=C["psiglu"], label="pSigLU, " + task, alpha=al)
        v = sob[(sob.activation == "laplu") & (sob.task == task)].grad_mse.values
        a.axhline(max(v.mean(), 1e-8), ls=ls, color=C["laplu"], lw=1.0, alpha=al, label="LapLU, " + task)
    a.set_yscale("log"); a.set_title("(d) Sobolev, $\\lambda$ = 1"); a.set_ylabel("gradient-matching MSE")
    a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for a in ax:
        a.set_xscale("log"); a.set_xticks(XT); a.set_xticklabels(XL, fontsize=6.5); a.set_xlabel("tail index p"); a.xaxis.set_minor_locator(NullLocator())
        if a.get_legend() is None: a.legend(frameon=False, fontsize=6.5)
    ax[2].set_yticks([0.5, 1, 2, 5]); ax[2].set_yticklabels(["0.5", "1", "2", "5"]); ax[2].yaxis.set_minor_locator(NullLocator())
    fig.tight_layout(); save(fig, "Figure9_p_across_tasks")
print("done")
