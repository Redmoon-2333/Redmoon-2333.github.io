"""Plot exact theoretical comparison counts. Author: RedMoon."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    n = np.arange(1, 1025)
    harmonic = np.cumsum(1.0 / n)
    expected = 2 * (n + 1) * harmonic - 4 * n
    worst = n * (n - 1) / 2
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#25292c",
        "text.color": "#25292c",
        "xtick.color": "#626a70",
        "ytick.color": "#626a70",
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    fig.set_facecolor("white")
    ax.plot(n[1:], worst[1:], color="#b31b1b", lw=2.1, label="最坏：n(n − 1) / 2")
    ax.plot(n[1:], expected[1:], color="#157568", lw=2.5,
            label="随机枢轴的期望：2(n + 1)Hₙ − 4n")
    ax.set(xscale="log", yscale="log", xlabel="元素个数 n（互异键）",
           ylabel="关键字比较次数", xlim=(2, 1024))
    ax.set_xticks([2, 8, 32, 128, 512, 1024])
    ax.set_xticklabels(["2", "8", "32", "128", "512", "1024"])
    ax.grid(which="major", color="#e3e7e8", lw=0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    for value in (32, 128, 1024):
        index = value - 1
        ax.scatter(value, expected[index], color="#157568", s=22, zorder=4)
    fig.savefig(root / "quicksort-comparisons.png", dpi=260, facecolor="white")
    fig.savefig(root / "quicksort-comparisons.pdf", facecolor="white")
    plt.close(fig)
    rows = [
        {"n": int(value), "expected": float(expected[value - 1]),
         "worst": int(worst[value - 1])}
        for value in (2, 8, 32, 128, 512, 1024)
    ]
    (root / "quicksort-comparison-data.json").write_text(
        json.dumps({"kind": "exact theoretical counts", "data": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
