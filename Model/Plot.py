import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _axis_limits(labels, preds):
    values = np.concatenate([labels, preds]) if labels.size else np.asarray([0.0, 1.0])
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad

def save_group_scatter(output_path, title, labels, preds):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = _axis_limits(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.scatter(labels, preds, s=26, alpha=0.8, color="#4C72B0", edgecolors="white", linewidths=0.4)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental pK$_a$", fontsize=12)
    ax.set_ylabel("Predicted pK$_a$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def draw_from_dataframe(df, output_dir, prefix):
    output_dir = Path(output_dir)
    written = []
    for name, group in df.groupby("Dataset"):
        labels = group["Actual"].to_numpy(float)
        preds = group["Predicted"].to_numpy(float)
        out = output_dir / f"{prefix}_{name}.png"
        save_group_scatter(out, name, labels, preds)
        written.append(out)
    return written

def parse_args():
    p = argparse.ArgumentParser(description="Draw HIP-IT prediction scatter plots from a results CSV.")
    p.add_argument("--predictions", required=True, help="results CSV with columns Dataset, Actual, Predicted")
    p.add_argument("--output-dir", default=None, help="directory for the PNGs (default: alongside the CSV)")
    p.add_argument("--prefix", default=None, help="output filename prefix (default: CSV stem)")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.predictions)
    df = pd.read_csv(csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    prefix = args.prefix or csv_path.stem
    written = draw_from_dataframe(df, output_dir, prefix)
    if written:
        for out in written:
            print(f"wrote {out}")
    else:
        print("no plottable test groups found in", csv_path)

if __name__ == "__main__":
    main()
