"""Plot the AoA / acquisition-trajectory figures from the harvested surprisal CSVs.

Reads results/register/aoa/traj_{a,b,c,d}_surprisal.csv and writes two figures:
  figures/register/aoa_trajectory.png  — Latinate surprisal vs tokens seen, per arm (dose ladder)
  figures/register/aoa_gap.png         — Latinate-Germanic gap vs tokens seen, per arm

    python scripts/register/plot_aoa.py --indir results/register/aoa --outdir figures/register
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# arm key -> (label, colour, linestyle). A/C/D = vocab-only dose ladder; B = syntax+vocab.
ARMS = {
    "a": ("A  original", "#111111", "-"),
    "c": ("C  register mild (−5.5pp Lat)", "#2c7fb8", "-"),
    "d": ("D  register aggressive (−18pp Lat)", "#d95f0e", "-"),
    "b": ("B  simplify (syntax+vocab)", "#888888", "--"),
}


def load(indir: Path, key: str):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(indir / f"traj_{key}_surprisal.csv") as fh:
        for r in csv.DictReader(fh):
            by[int(r["tokens_seen"])][r["klass"]].append(float(r["surprisal_bits"]))
    xs = sorted(by)
    lat = [sum(by[t]["Latinate"]) / len(by[t]["Latinate"]) for t in xs]
    ger = [sum(by[t]["Germanic"]) / len(by[t]["Germanic"]) for t in xs]
    gap = [l - g for l, g in zip(lat, ger)]
    return xs, lat, ger, gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/register/aoa")
    ap.add_argument("--outdir", default="figures/register")
    args = ap.parse_args()
    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {k: load(indir, k) for k in ARMS}

    # --- Figure 1: Latinate acquisition trajectory (the dose ladder) ---
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for k, (label, c, ls) in ARMS.items():
        xs, lat, _, _ = data[k]
        ax.plot(xs, lat, ls, color=c, lw=2.2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen (log)")
    ax.set_ylabel("Latinate word surprisal (bits)")
    ax.set_title("Latinate acquisition slows with register dose", loc="left")
    ax.grid(True, which="both", color="#eee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "aoa_trajectory.png", dpi=200)
    fig.savefig(outdir / "aoa_trajectory.svg")

    # --- Figure 2: Latinate-Germanic gap over training ---
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for k, (label, c, ls) in ARMS.items():
        xs, _, _, gap = data[k]
        ax.plot(xs, gap, ls, color=c, lw=2.2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen (log)")
    ax.set_ylabel("Latinate − Germanic gap (bits)")
    ax.set_title("Register dose widens the Germanic–Latinate gap (monotonic)", loc="left")
    ax.grid(True, which="both", color="#eee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "aoa_gap.png", dpi=200)
    fig.savefig(outdir / "aoa_gap.svg")

    # print final-step summary for the log
    tf = max(data["a"][0])
    print("final gap (bits):", {k.upper(): round(load(indir, k)[3][-1], 2) for k in ARMS})
    print(f"wrote {outdir}/aoa_trajectory.png and {outdir}/aoa_gap.png")


if __name__ == "__main__":
    main()
