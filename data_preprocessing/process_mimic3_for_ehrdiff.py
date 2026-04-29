import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

# 配置路径
MIMIC3_PATH = "data/mimic"  # 请修改为您的 MIMIC-III 路径
OUTPUT_PATH = "data/mimic"

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 50)
print("开始处理 MIMIC-III 数据集")
print("=" * 50)

# 1. 读取 ADMISSIONS 表（获取住院记录）
print("\n[1/5] 读取 ADMISSIONS 表...")
admissions = pd.read_csv(
    os.path.join(MIMIC3_PATH, "ADMISSIONS.csv"),
    dtype={'SUBJECT_ID': str, 'HADM_ID': str}
)
print(f"   共 {len(admissions)} 条住院记录")

# 2. 读取 DIAGNOSES_ICD 表（诊断代码）
print("\n[2/5] 读取 DIAGNOSES_ICD 表...")
diagnoses = pd.read_csv(
    os.path.join(MIMIC3_PATH, "DIAGNOSES_ICD.csv"),
    dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'ICD9_CODE': str}
)

# 过滤有效的 ICD-9 代码
diagnoses = diagnoses.dropna(subset=['ICD9_CODE'])
# 截断到前三位（论文要求）
diagnoses['ICD9_CODE_3D'] = diagnoses['ICD9_CODE'].str[:3]
print(f"   共 {len(diagnoses)} 条诊断记录")
print(f"   诊断代码种类（截断后）: {diagnoses['ICD9_CODE_3D'].nunique()}")

# 3. 读取 PROCEDURES_ICD 表（手术代码）
print("\n[3/5] 读取 PROCEDURES_ICD 表...")
procedures = pd.read_csv(
    os.path.join(MIMIC3_PATH, "PROCEDURES_ICD.csv"),
    dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'ICD9_CODE': str}
)

# 过滤有效的 ICD-9 代码
procedures = procedures.dropna(subset=['ICD9_CODE'])
# 截断到前三位（论文要求）
procedures['ICD9_CODE_3D'] = procedures['ICD9_CODE'].str[:3]
print(f"   共 {len(procedures)} 条手术记录")
print(f"   手术代码种类（截断后）: {procedures['ICD9_CODE_3D'].nunique()}")

# 4. 合并所有代码，构建代码字典
print("\n[4/5] 构建代码字典...")

# 收集所有出现的代码（诊断 + 手术）
all_codes = pd.concat([
    diagnoses[['HADM_ID', 'ICD9_CODE_3D']].rename(columns={'ICD9_CODE_3D': 'code'}),
    procedures[['HADM_ID', 'ICD9_CODE_3D']].rename(columns={'ICD9_CODE_3D': 'code'})
])

# 统计每个代码出现的频率
code_freq = all_codes['code'].value_counts()
print(f"   总代码种类（截断后）: {len(code_freq)}")

# 根据论文，最终使用的代码集应该是 1782 维
# 选择出现频率最高的 1782 个代码
if len(code_freq) > 1782:
    selected_codes = code_freq.head(1782).index.tolist()
    print(f"   选择前 1782 个最常见的代码")
else:
    selected_codes = code_freq.index.tolist()
    print(f"   实际代码数 {len(selected_codes)} < 1782，将使用所有代码")

# 创建代码到索引的映射
code_to_idx = {code: i for i, code in enumerate(selected_codes)}
idx_to_code = {i: code for code, i in code_to_idx.items()}

# 5. 为每个住院记录创建二进制向量
print("\n[5/5] 生成 EHR 向量...")

# 获取所有有效的住院记录（同时有诊断和手术的？论文没有明确要求，这里使用所有有代码的住院）
valid_admissions = all_codes['HADM_ID'].unique()
print(f"   有代码的住院记录数: {len(valid_admissions)}")

# 创建每个住院记录的代码集合
hadm_to_codes = {}
for hadm_id, group in tqdm(all_codes.groupby('HADM_ID'), desc="   处理住院记录"):
    hadm_to_codes[hadm_id] = set(group['code'].values)

# 创建特征矩阵
feature_dim = len(selected_codes)
n_samples = len(hadm_to_codes)
print(f"   特征维度: {feature_dim}")
print(f"   样本数量: {n_samples}")

# 初始化特征矩阵
X = np.zeros((n_samples, feature_dim), dtype=np.float32)
hadm_ids = []

# 填充特征矩阵
for i, (hadm_id, codes) in enumerate(tqdm(hadm_to_codes.items(), desc="   构建特征矩阵")):
    hadm_ids.append(hadm_id)
    for code in codes:
        if code in code_to_idx:
            X[i, code_to_idx[code]] = 1.0

print(f"\n最终数据形状: {X.shape}")

# 6. 保存数据
print("\n保存数据...")

# 保存完整数据集
np.save(os.path.join(OUTPUT_PATH, "mimic_data.npy"), X)
print(f"   数据已保存到: {os.path.join(OUTPUT_PATH, 'mimic_data.npy')}")

# 保存代码映射信息
import json
mapping_info = {
    'code_to_idx': {str(k): v for k, v in code_to_idx.items()},
    'idx_to_code': {str(k): v for k, v in idx_to_code.items()},
    'feature_dim': feature_dim,
    'n_samples': n_samples,
    'selected_codes': selected_codes,
    'code_frequencies': {str(k): int(v) for k, v in code_freq.head(1782).items()}
}
with open(os.path.join(OUTPUT_PATH, "code_mapping.json"), 'w') as f:
    json.dump(mapping_info, f, indent=2)
print(f"   代码映射已保存到: {os.path.join(OUTPUT_PATH, 'code_mapping.json')}")

# 7. 创建训练/测试集划分
print("\n创建训练/测试集划分...")

# 随机打乱
np.random.seed(42)  # 固定随机种子以保证可复现
indices = np.random.permutation(n_samples)

# 论文中的划分：41868 训练，4652 测试
train_size = 41868
test_size = 4652

if n_samples >= train_size + test_size:
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    print(f"   训练集大小: {len(train_indices)}")
    print(f"   测试集大小: {len(test_indices)}")
else:
    # 如果样本数不足，按比例划分
    train_size = int(n_samples * 0.9)
    test_size = n_samples - train_size
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    print(f"   警告: 样本数({n_samples})少于论文要求，使用 90/10 划分")
    print(f"   训练集大小: {len(train_indices)}")
    print(f"   测试集大小: {len(test_indices)}")

# 保存划分索引
np.save(os.path.join(OUTPUT_PATH, "train_indices.npy"), train_indices)
np.save(os.path.join(OUTPUT_PATH, "test_indices.npy"), test_indices)
print(f"   划分索引已保存")

# 8. 数据统计
print("\n" + "=" * 50)
print("数据统计")
print("=" * 50)

print(f"总样本数: {n_samples}")
print(f"特征维度: {feature_dim}")
print(f"训练集大小: {len(train_indices)}")
print(f"测试集大小: {len(test_indices)}")
print(f"数据稀疏度: {1 - X.mean():.4f}")

# 计算每个代码的频率
code_freq_in_data = X.mean(axis=0)
print(f"代码平均出现频率: {code_freq_in_data.mean():.4f}")
print(f"代码频率范围: [{code_freq_in_data.min():.4f}, {code_freq_in_data.max():.4f}]")

print("\n处理完成！")