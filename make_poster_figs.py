# make_poster_figs.py — poster-ready figures for both KIBA and Davis

import os, re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ====== CONFIG ======
INPUT_DIR = Path("C:/Users/administer/Desktop/fusion_results")
OUT_DIR = INPUT_DIR / "poster_figs"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = {"ds0": "KIBA", "ds1": "Davis"}  # label → dataset name
DPI = 600
COLOR_MAP = {"baseline": "#1f77b4", "crossattention": "#ff7f0e", "transformer": "#2ca02c"}
SHORT_MODEL = {"baseline": "Baseline", "crossattention": "CrossAttn", "transformer": "Transformer"}
ORDER_MODELS = ["baseline", "crossattention", "transformer"]

# ====== HELPERS ======
def sci_lr(x: float):
    if x is None or x == 0: return str(x)
    exp = int(math.floor(math.log10(abs(x))))
    base = x / (10**exp)
    base = round(base, 2)
    base_str = str(int(base)) if float(base).is_integer() else str(base)
    return f"{base_str}e{exp}"

def parse_fname(fname: str):
    f = fname.lower()
    model = "unknown"
    if "transformer" in f: model = "transformer"
    elif "crossattention" in f: model = "crossattention"
    elif "baseline" in f: model = "baseline"
    m_bs = re.search(r"bs(\d+)", f)
    bs = int(m_bs.group(1)) if m_bs else None
    m_lr = re.search(r"lr([0-9eE\.\-]+)", f)
    lr = None
    if m_lr:
        try: lr = float(m_lr.group(1))
        except: lr = None
    m_ds = re.search(r"ds(\d+)", f)
    ds = m_ds.group(0) if m_ds else None  # e.g. 'ds0'
    return model, bs, lr, ds

def select_best_epoch_row(df: pd.DataFrame):
    return df.loc[df["mse"].idxmin()]

def save_best_metrics_fig(df, dataset, out_dir, title_prefix="Performance with Best Hyperparameters"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # 获取每个模型的最优行（按最小 MSE 排序）
    best_rows = (
        df.sort_values(["Model", "MSE"], ascending=[True, True])
          .groupby("Model", as_index=False).first()
    )
    best_rows["Model"] = pd.Categorical(best_rows["Model"], categories=ORDER_MODELS, ordered=True)
    best_rows = best_rows.sort_values("Model")

    metrics = [("RMSE", "↓"), ("MSE", "↓"), ("CI", "↑"), ("Pearson", "↑"), ("Spearman", "↑")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6), dpi=DPI)

    for ax, (m, arrow) in zip(axes, metrics):
        labels = [SHORT_MODEL[x] for x in best_rows["Model"].astype(str).tolist()]
        colors = [COLOR_MAP[x] for x in best_rows["Model"].astype(str).tolist()]
        vals = best_rows[m].values

        ax.bar(labels, vals, color=colors)
        ax.set_title(f"{m} ({arrow})", fontsize=13, pad=6)
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

        # 添加数值标注
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        # 可选添加参考线：均值线
        # ax.axhline(np.mean(vals), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 总标题
    fig.suptitle(f"{title_prefix} ({dataset})", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0.0, 1.0, 0.92])

    # 保存图像
    fig_path = out_dir / f"{dataset}_poster_best_hparam_metrics.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)

    return best_rows

def save_gridsearch_fig(df, dataset, out_dir):
    df2 = df.copy()
    df2["LRs"] = df2["LR"].map(sci_lr)
    df2["Label"] = "bs" + df2["Batch"].astype(str) + "|lr" + df2["LRs"].astype(str)
    df2 = df2.sort_values("RMSE", ascending=True).reset_index(drop=True)
    df2["Xtick"] = df2.index.astype(str)
    colors = [COLOR_MAP[m] for m in df2["Model"].astype(str)]
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=DPI)
    bars = ax.bar(df2["Xtick"], df2["RMSE"], color=colors)
    ax.set_title(f"RMSE across Hyperparameter Settings ({dataset})", fontsize=26, pad=8, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xticks(range(len(df2)))
    ax.set_xticklabels(df2["Label"], rotation=45, ha="right", fontsize=9)
    for rect, v in zip(bars, df2["RMSE"].values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    legend_handles = [Patch(facecolor=COLOR_MAP[k], label=SHORT_MODEL[k]) for k in ORDER_MODELS]
    ax.legend(handles=legend_handles, title="Model", loc="upper left", frameon=False,
              fontsize=14, title_fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_poster_gridsearch_rmse.png", dpi=DPI)
    plt.close(fig)

def save_ablation_csv(df, best_transformer, dataset, out_dir):
    bs, lr = best_transformer["Batch"], best_transformer["LR"]
    ab_df = df[(df["Batch"] == bs) & (df["LR"] == lr)].copy()
    ab_df["Model"] = ab_df["Model"].map(SHORT_MODEL)
    ab_df = ab_df.sort_values("Model")
    ab_df.rename(columns=lambda x: x + "_mean" if x in ["RMSE", "MSE", "CI", "Pearson", "Spearman"] else x,
                 inplace=True)
    ab_df.to_csv(out_dir / f"{dataset}_ablation_controlled_at_transformer_best.csv", index=False)

def save_all_metrics_fig(df, best_transformer, dataset, out_dir):
    bs, lr = best_transformer["Batch"], best_transformer["LR"]
    sub_df = df[(df["Batch"] == bs) & (df["LR"] == lr)].copy()
    sub_df["Model"] = pd.Categorical(sub_df["Model"], categories=ORDER_MODELS, ordered=True)
    sub_df = sub_df.sort_values("Model")

    metrics = [("RMSE", "↓"), ("MSE", "↓"), ("CI", "↑"), ("Pearson", "↑"), ("Spearman", "↑")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6), dpi=DPI)
    for ax, (m, arrow) in zip(axes, metrics):
        labels = [SHORT_MODEL[x] for x in sub_df["Model"].astype(str).tolist()]
        colors = [COLOR_MAP[x] for x in sub_df["Model"].astype(str).tolist()]
        vals = sub_df[m].values
        ax.bar(labels, vals, color=colors)
        ax.set_title(f"{m} ({arrow})", fontsize=13, pad=6)
        ax.set_ylabel(m)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Ablation under Transformer Best Hyperparams ({dataset})", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0.0, 1.0, 0.92])
    fig.savefig(out_dir / f"{dataset}_poster_fig3_all_metrics.png", dpi=DPI)
    plt.close(fig)

# ====== MAIN LOOP ======
for tag, dataset in DATASETS.items():
    print(f"📊 Processing {dataset}...")
    hist_files = []
    for p, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            if f.startswith("history_") and f.endswith(".csv") and tag in f:
                hist_files.append(Path(p) / f)
    if not hist_files:
        print(f"⚠️ No history files found for {dataset}. Skipped.")
        continue

    records = []
    for f in hist_files:
        model, bs, lr, _ds = parse_fname(f.name)
        if model == "unknown" or bs is None or lr is None: continue
        try:
            df = pd.read_csv(f)
            best = select_best_epoch_row(df)
            records.append({
                "Model": model, "Batch": int(bs), "LR": float(lr), "File": f.name,
                "Epoch": int(best["epoch"]), "RMSE": float(best["rmse"]),
                "MSE": float(best["mse"]), "CI": float(best["ci"]),
                "Pearson": float(best["pearson"]), "Spearman": float(best["spearman"]),
            })
        except Exception as e:
            print(f"[WARN] {f.name} skipped: {e}")

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print(f"⚠️ No valid data for {dataset}. Skipped.")
        continue

    out_dir = OUT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    best_rows = save_best_metrics_fig(df_all, dataset, out_dir)
    save_gridsearch_fig(df_all, dataset, out_dir)

    try:
        best_transformer = best_rows[best_rows["Model"] == "transformer"].iloc[0]
        save_ablation_csv(df_all, best_transformer, dataset, out_dir)
        save_all_metrics_fig(df_all, best_transformer, dataset, out_dir)
        print(f"✅ Finished {dataset}: 3 plots + ablation CSV saved.")
    except Exception as e:
        print(f"⚠️ No transformer best row found for {dataset}: {e}")
