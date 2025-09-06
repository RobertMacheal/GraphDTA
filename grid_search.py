import os
import sys

# 检查参数
if len(sys.argv) != 3 or sys.argv[2] not in ['-t', '-c', '-b']:
    print("Usage: python grid_search.py [dataset_idx: 0|1] [-t | -c | -b]")
    print("  -t = transformer fusion")
    print("  -c = crossattention fusion")
    print("  -b = baseline (no fusion)")
    sys.exit(1)

# 参数解析
dataset_idx = int(sys.argv[1])  # 0 = davis, 1 = kiba
fusion_arg = sys.argv[2]

fusion_map = {'-t': 0, '-c': 1}
fusion_names = {0: "transformer", 1: "crossattention"}

if fusion_arg == '-b':
    fusion_idx = None
    tag = "baseline"
else:
    fusion_idx = fusion_map[fusion_arg]
    tag = fusion_names[fusion_idx]

# 搜索参数
batch_sizes = [128, 256, 512]
learning_rates = [1e-3, 5e-4, 1e-4]
model_idx = 0
cuda = 0

# PBS 模板
pbs_template = """#!/bin/bash
#PBS -N train_{tag}_{bs}_{lr}
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=12:00:00
#PBS -q half_day
#PBS -j oe
#PBS -m n
#PBS -V

cd $PBS_O_WORKDIR
module use /apps2/modules/all
module load Anaconda3
source /apps2/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate graphdta_env

python training.py \\
  --dataset_idx {dataset_idx} \\
  --model_idx {model_idx} \\
  --cuda {cuda} \\
  {fusion_line} \\
  --batch_size {bs} \\
  --lr {lr} \\
  --history_file history_{tag}_bs{bs}_lr{lr}_ds{dataset_idx}.csv
"""

# 创建 PBS 脚本目录
pbs_dir = "pbs_jobs"
os.makedirs(pbs_dir, exist_ok=True)

for bs in batch_sizes:
    for lr in learning_rates:
        job_name = f"{tag}_bs{bs}_lr{lr}_ds{dataset_idx}"
        fusion_line = f"--fusion {fusion_idx}" if fusion_idx is not None else ""
        script = pbs_template.format(
            tag=tag, bs=bs, lr=lr,
            dataset_idx=dataset_idx, model_idx=model_idx,
            cuda=cuda, fusion_line=fusion_line
        )
        script_path = os.path.join(pbs_dir, f"job_{job_name}.sh")
        with open(script_path, "w") as f:
            f.write(script)
        os.system(f"qsub {script_path}")
        print(f"Submitted: {job_name}")
