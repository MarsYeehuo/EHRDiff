# EHRDiff 项目进度总结

## 项目概述

复现论文《EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models》，使用扩散模型生成合成EHR数据。

分两阶段：
- **第一阶段（已完成）**：在 MIMIC-III 上复现论文方法，验证可行性
- **第二阶段（当前目标）**：迁移到 MIMIC-IV，在性能更好的平台上完成训练和全面评估

------

## 第一阶段：MIMIC-III 复现（已完成）

### 环境配置

- Python 3.9 + PyTorch (Windows 本地机器, gloo 后端, GPU 显存有限)
- 关键依赖：omegaconf, tqdm, numpy, scikit-learn, lightgbm, einops
- 主要修改：main.py 中 nccl→gloo，修复 linear_model.py 维度计算，修复 denoiser.py 广播问题
- opacus（差分隐私库）已内置在 src/opacus/，非 DP 训练不受影响

### MIMIC-III 数据处理

- 提取诊断(ICD-9)和手术(ICD-9)代码，截断到前3位
- 最终维度：1034（论文为1782，因 MIMIC-III 版本差异导致可用代码数不同）
- 输出：mimic_data.npy (46520样本)，train/test indices (90/10划分)
- 处理脚本：data_preprocessing/process_mimic3_for_ehrdiff.py

### 模型训练

- 配置文件：configs/mimic/train_edm.yaml
- 架构：EDM Denoiser + MLP 骨干网络 (LinearModel)
- 关键参数：z_dim=1034, time_dim=384, unit_dims=[1034, 1024, 1024, 1024, 1034]
- 参数量：约 4.4M
- batch_size=512, 论文使用 41868/4652 训练/测试划分
- 训练状态：已完成，有多个 checkpoint

### 样本生成

- 配置：configs/mimic/sample_edm_1034.yaml
- 命令：main.py --mode eval ++model.ckpt=checkpoint路径
- 输出：all_x.npy, sample.npy（位于 results/mimic_edm/samples/）

### 评估现状

- 已有：NZC (None-Zero Columns)、Prevalence Correlation 的实时监控（训练过程中 plot_dim_dist）
- 已生成：prevalence_comparison.png（真实 vs 合成数据 feature prevalence 对比图）
- 缺少：CMD、AUROC、F1、隐私风险评估等全面指标

------

## 第二阶段：MIMIC-IV 迁移（当前重点）

### MIMIC-IV 数据概况

| 指标 | MIMIC-III | MIMIC-IV |
|------|-----------|----------|
| 样本数 | 46,520 | 545,576 |
| 特征维度 | 1,034 | 2,241 |
| 数据大小 | ~190MB | ~4.9GB |
| 编码类型 | ICD-9 only | ICD-9 + ICD-10 混合 |
| 代码筛选 | top-1782(实际1034) | 出现≥50次 |
| 稀疏度 | ~99.4% | ~99.5% |

- 数据已处理完成：data/mimic4/mimic4_data.npy + train_indices.npy + test_indices.npy + metadata.json
- 处理脚本：data_preprocessing/process_mimic4_for_ehrdiff.py
- ICD-9/10 混合编码：当前方案统一截断到前3位，不做版本区分。这是一个可能需要改进的点（ICD-9 的 "250" 和 ICD-10 的 "E11" 虽然语义不同，但在 3-digit 级别可以区分）

### MIMIC-IV 模型配置

- 配置文件：configs/mimic4/train_edm.yaml
- 架构：EDM Denoiser + MLP 骨干网络
- 关键参数：z_dim=2241, time_dim=384, unit_dims=[2241, 1024, 1024, 1024, 2241]
- 参数量：约 20.9M（是 MIMIC-III 模型的 ~5倍）
- batch_size=64, n_epochs=5000
- 需要创建：configs/mimic4/sample_edm_2241.yaml（样本生成配置，目前尚不存在）

### 本地训练尝试（未成功）

- 使用 configs/mimic4/train_edm.yaml 在本地 Windows 机器启动训练
- 成功完成模型初始化（20.9M 参数确认）
- 因 GPU 显存不足无法继续训练（~4.9GB 数据 + 20.9M 参数 + batch_size 需求）
- 日志位于 results/mimic4/stdout.txt

### 新平台迁移要点

迁移到性能更好的平台（Linux + 高性能 GPU）时需要注意：

**环境**：
- Python 3.9+, PyTorch with CUDA, 以下依赖：omegaconf, tqdm, numpy, scikit-learn, lightgbm, einops
- pip install -r requirements.txt
- 确保 src/opacus 目录完整（DP 训练可选，非 DP 训练仅需 import 不报错即可）

**后端切换**：
- Linux + NVIDIA GPU → 可将 main.py 中的 gloo 改回 nccl（或保持 gloo 也可工作）
- 如有多 GPU，可调整 n_gpus_per_node

**显存估算**：
- 数据：545,576 × 2241 × 4 bytes ≈ 4.9GB（磁盘）/ 训练时按 batch 加载
- 模型参数：20.9M × 4 bytes ≈ 84MB
- 优化器状态(AdamW)：~168MB
- 激活值 + 梯度：取决于 batch_size
- 建议：至少 8GB+ 显存，推荐 16GB+（batch_size 可适当提高到 128-256）

**待完成任务**（按优先级排列）：

1. **环境搭建**：在新平台上 clone 仓库，安装依赖，验证 src/opacus 可导入
2. **数据验证**：确认 mimic4_data.npy 可正常加载，维度正确
3. **创建采样配置**：基于 configs/mimic/sample_edm_1034.yaml 创建 configs/mimic4/sample_edm_2241.yaml
4. **MIMIC-IV 训练**：
   ```bash
   python main.py --config configs/mimic4/train_edm.yaml --workdir ./results/mimic4 --mode train
   ```
5. **样本生成**：
   ```bash
   python main.py --config configs/mimic4/sample_edm_2241.yaml --workdir ./results/mimic4 --mode eval ++model.ckpt=checkpoints/checkpoint_XXXXX.pth
   ```
6. **全面评估**：NZC, CMD, Prevalence Correlation, AUROC, F1, 隐私风险
7. **与论文 Table 1 对比**：论文目标 NZC≈1770, CMD≈7.769, Prevalence Correlation>0.99

------

## 项目结构

```
EHRDiff/
├── main.py                     # 主入口
├── denoiser.py                 # EDM/VPSDE/VESDE Denoiser 实现
├── samplers.py                 # 采样器（Heun solver + EDM discretization）
├── score_losses.py             # EDM/VPSDE/VESDE 损失函数
├── requirements.txt
├── configs/
│   ├── mimic/                  # MIMIC-III 配置
│   │   ├── train_edm.yaml
│   │   ├── sample_edm_1034.yaml
│   │   └── train_dp.yaml       # 差分隐私训练配置（未使用）
│   └── mimic4/                 # MIMIC-IV 配置
│       └── train_edm.yaml      # 需补充 sample_edm_2241.yaml
├── data/
│   ├── mimic3/                 # MIMIC-III npy 数据 + 原始 CSV
│   └── mimic4/                 # MIMIC-IV npy 数据 + 原始 CSV + metadata.json
├── data_preprocessing/
│   ├── process_mimic3_for_ehrdiff.py
│   ├── process_mimic4_for_ehrdiff.py
│   └── check_dim.py
├── model/
│   └── linear_model.py         # MLP 骨干网络 (Block + LinearModel)
├── runners/
│   ├── train_dpdm_base.py      # 训练入口（支持 DP 和非 DP 两种模式）
│   └── generate_base.py        # 样本生成入口
├── utils/
│   └── util.py                 # plot_dim_dist, sample_random_batch, save_checkpoint 等
├── src/opacus/                 # 内置 opacus 差分隐私库
├── results/
│   ├── mimic_edm/              # MIMIC-III 训练结果（checkpoints + samples）
│   └── mimic4/                 # MIMIC-IV 训练结果（当前为空，待训练）
│       ├── checkpoints/
│       ├── samples/
│       └── stdout.txt
├── dnnlib/                     # 工具库
└── torch_utils/                # PyTorch 工具
```

------

## 常用命令

### MIMIC-III（已完成）
```bash
# 训练
python main.py --config configs/mimic/train_edm.yaml --workdir ./results/mimic_edm --mode train

# 生成
python main.py --config configs/mimic/sample_edm_1034.yaml --workdir ./results/mimic_edm --mode eval ++model.ckpt=checkpoints/checkpoint_50000.pth
```

### MIMIC-IV（待执行）
```bash
# 训练
python main.py --config configs/mimic4/train_edm.yaml --workdir ./results/mimic4 --mode train

# 生成（需要先创建 sample_edm_2241.yaml）
python main.py --config configs/mimic4/sample_edm_2241.yaml --workdir ./results/mimic4 --mode eval ++model.ckpt=checkpoints/checkpoint_XXXXX.pth
```

------

## 注意事项

- **后端**：Windows 必须使用 gloo，Linux + NVIDIA 推荐 nccl
- **维度一致性**：配置文件中 resolution/z_dim/unit_dims 首尾值 必须与实际数据维度一致
- **checkpoint 路径**：生成时 ++model.ckpt 的路径相对于 workdir
- **差分隐私**：dp.do: False，DP 训练代码路径存在但未被调用
- **显存优化**：如需降低显存，可减小 batch_size、启用混合精度(AMP)、减小 unit_dims 隐藏层（如 1024→512）
- **ICD 编码**：MIMIC-IV 同时包含 ICD-9 和 ICD-10 代码，当前方案均截断前3位混合处理

------

## 参考论文

EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models (OpenReview 2024)
关键指标目标：NZC≈1770, CMD≈7.769, Prevalence Correlation>0.99
