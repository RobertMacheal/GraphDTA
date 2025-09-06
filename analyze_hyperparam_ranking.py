import os
import pandas as pd
import matplotlib.pyplot as plt

# 🔧 设置主目录
input_dir = "C:/Users/administer/Desktop/fusion_results"
output_root = os.path.join(input_dir, "ranking_output")
os.makedirs(output_root, exist_ok=True)

DATASETS = {"ds0": "kiba", "ds1": "davis"}
metrics = {
    "RMSE": "asc",
    "MSE": "asc",
    "CI": "desc",
    "Pearson": "desc",
    "Spearman": "desc"
}

for ds_tag, ds_name in DATASETS.items():
    output_dir = os.path.join(output_root, ds_name)
    os.makedirs(output_dir, exist_ok=True)

    # 📦 收集当前数据集的 history 文件
    csv_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv") and file.startswith("history_") and ds_tag in file:
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print(f"❌ No valid records found for {ds_name.upper()}.")
        continue

    # 📊 收集每个文件的最佳指标
    records = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            best_row = df.loc[df["rmse"].idxmin()]
            fname = os.path.basename(fpath)
            parts = fname.split("_")
            model = parts[1]  # baseline, transformer, etc.
            bs = int([p for p in parts if p.startswith("bs")][0][2:])
            lr = float([p for p in parts if p.startswith("lr")][0][2:])
            records.append({
                "Model": model,
                "BatchSize": bs,
                "LR": lr,
                "File": fname,
                "Best Epoch": int(best_row["epoch"]),
                "RMSE": round(best_row["rmse"], 4),
                "MSE": round(best_row["mse"], 4),
                "CI": round(best_row["ci"], 4),
                "Pearson": round(best_row["pearson"], 4),
                "Spearman": round(best_row["spearman"], 4)
            })
        except Exception as e:
            print(f"⚠️ Failed to process {fpath}: {e}")

    # 🔢 转为DataFrame并进行排名
    df = pd.DataFrame(records)
    for metric, order in metrics.items():
        df[f"Rank_{metric}"] = df[metric].rank(ascending=(order == "asc"), method="min")

    df["AvgRank"] = df[[f"Rank_{m}" for m in metrics]].mean(axis=1)

    # ⬇️ 排序并保存
    df_sorted = df.sort_values("AvgRank")
    output_csv = os.path.join(output_dir, f"{ds_name}_best_metrics.csv")
    df_sorted.to_csv(output_csv, index=False)
    print(f"✅ Summary saved for {ds_name.upper()} ➜ {output_csv}")

    # 📈 生成排名柱状图
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sorted_df = df.sort_values(metric if metrics[metric] == "asc" else f"Rank_{metric}")
        x_labels = [f"{m}_bs{b}_lr{l}" for m, b, l in zip(sorted_df.Model, sorted_df.BatchSize, sorted_df.LR)]
        plt.bar(x_labels, sorted_df[metric])
        plt.title(f"{metric} across Hyperparameter Settings ({ds_name.upper()})")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric}_ranking.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"✅ Plots saved to: {output_dir}")
