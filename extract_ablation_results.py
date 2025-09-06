import os
import pandas as pd
import matplotlib.pyplot as plt

# === 参数设置 ===
FOLDER = "fusion_results"
BATCH = 512
LR = 0.0001
DATASET_IDX = 0
TAG_MATCH = f"bs{BATCH}_lr{LR}_ds{DATASET_IDX}"

# === 提取 ablation 数据 ===
results = []

for fname in os.listdir(FOLDER):
    if fname.endswith(".csv") and fname.startswith("history_") and TAG_MATCH in fname:
        fpath = os.path.join(FOLDER, fname)
        try:
            df = pd.read_csv(fpath)
            best_row = df.loc[df["rmse"].idxmin()]
            model_type = "Unknown"
            if "transformer" in fname:
                model_type = "Transformer"
            elif "crossattention" in fname:
                model_type = "CrossAttention"
            elif "baseline" in fname:
                model_type = "Baseline"

            results.append({
                "Model": model_type,
                "File": fname,
                "Best Epoch": int(best_row["epoch"]),
                "Best RMSE": round(best_row["rmse"], 4),
                "Best MSE": round(best_row["mse"], 4),
                "CI": round(best_row["ci"], 4)
            })
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")

# === 打印结果 & 作图 ===
if results:
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("Model")  # 保持一致顺序
    print("\n### 📊 Ablation Results (KIBA, bs=512, lr=0.0001):\n")
    print(df_out[["Model", "Best Epoch", "Best RMSE", "Best MSE", "CI"]].to_markdown(index=False))

    # ==== 画柱状图 ====
    colors = ["#B0C4DE", "#87CEFA", "#4682B4"]
    plt.rcParams.update({'font.size': 12})

    # --- RMSE ---
    plt.figure(figsize=(6, 4))
    plt.bar(df_out["Model"], df_out["Best RMSE"], color=colors)
    plt.title("Ablation: RMSE on KIBA (bs=512, lr=0.0001)")
    plt.ylabel("RMSE")
    for i, val in enumerate(df_out["Best RMSE"]):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig("ablation_rmse_kiba.png")
    plt.close()

    # --- MSE ---
    plt.figure(figsize=(6, 4))
    plt.bar(df_out["Model"], df_out["Best MSE"], color=colors)
    plt.title("Ablation: MSE on KIBA (bs=512, lr=0.0001)")
    plt.ylabel("MSE")
    for i, val in enumerate(df_out["Best MSE"]):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig("ablation_mse_kiba.png")
    plt.close()

    # --- CI ---
    plt.figure(figsize=(6, 4))
    plt.bar(df_out["Model"], df_out["CI"], color=colors)
    plt.title("Ablation: CI on KIBA (bs=512, lr=0.0001)")
    plt.ylabel("Concordance Index (CI)")
    plt.ylim(0, 1.0)
    for i, val in enumerate(df_out["CI"]):
        plt.text(i, val + 0.01, f"{val:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig("ablation_ci_kiba.png")
    plt.close()

    print("✅ Plots saved: ablation_rmse_kiba.png, ablation_mse_kiba.png, ablation_ci_kiba.png")
else:
    print(f"⚠️ No matching files found for bs={BATCH}, lr={LR}, ds={DATASET_IDX}")
