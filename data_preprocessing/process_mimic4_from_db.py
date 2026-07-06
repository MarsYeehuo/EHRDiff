"""
从 PostgreSQL 数据库直接读取 MIMIC-IV 数据并生成 EHR 特征矩阵。
使用方法：
    python data_preprocessing/process_mimic4_from_db.py
"""

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sqlalchemy import create_engine

# 数据库连接信息（根据 mimic.md 文档）
DB_HOST = "localhost"
DB_PORT = "5433"
DB_USER = "mimicuser"
DB_PASSWORD = "knowlabMIMIC"
DB_NAME = "mimic"  # 如不同请修改

# MIMIC-IV schema（常见为 mimiciv_hosp 或 mimiciv）
SCHEMA = "mimiciv_hosp"

OUTPUT_PATH = "data/mimic4"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 创建数据库连接
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("=" * 50)
print("从 PostgreSQL 数据库读取 MIMIC-IV 数据集")
print("=" * 50)

# 1. 读取住院记录
print("\n[1/5] 读取住院记录...")
admissions = pd.read_sql_query(
    f"SELECT subject_id, hadm_id FROM {SCHEMA}.admissions",
    engine,
)
# 统一类型
admissions['subject_id'] = admissions['subject_id'].astype(str)
admissions['hadm_id'] = admissions['hadm_id'].astype(str)
print(f"   共 {len(admissions)} 条住院记录")

# 2. 读取诊断代码
print("\n[2/5] 读取诊断代码...")
diagnoses = pd.read_sql_query(
    f"SELECT subject_id, hadm_id, icd_code, icd_version FROM {SCHEMA}.diagnoses_icd",
    engine,
)
diagnoses['subject_id'] = diagnoses['subject_id'].astype(str)
diagnoses['hadm_id'] = diagnoses['hadm_id'].astype(str)


def process_icd_code(row):
    """处理 ICD 代码，统一为前3位"""
    code = row['icd_code']
    if pd.isna(code):
        return None
    code = str(code).strip()
    return code[:3]


diagnoses['icd_code_3d'] = diagnoses.apply(process_icd_code, axis=1)
diagnoses = diagnoses.dropna(subset=['icd_code_3d'])
print(f"   共 {len(diagnoses)} 条诊断记录")
print(f"   诊断代码种类（截断后）: {diagnoses['icd_code_3d'].nunique()}")

# 3. 读取手术代码
print("\n[3/5] 读取手术代码...")
try:
    procedures = pd.read_sql_query(
        f"SELECT subject_id, hadm_id, icd_code, icd_version FROM {SCHEMA}.procedures_icd",
        engine,
    )
    procedures['subject_id'] = procedures['subject_id'].astype(str)
    procedures['hadm_id'] = procedures['hadm_id'].astype(str)
    procedures['icd_code_3d'] = procedures.apply(process_icd_code, axis=1)
    procedures = procedures.dropna(subset=['icd_code_3d'])
    print(f"   共 {len(procedures)} 条手术记录")
    print(f"   手术代码种类: {procedures['icd_code_3d'].nunique()}")

    all_codes = pd.concat([
        diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'}),
        procedures[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})
    ])
except Exception as e:
    print(f"   手术代码表读取失败 ({e})，仅使用诊断代码")
    all_codes = diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})

# 4. 构建代码字典
print("\n[4/5] 构建代码字典...")
code_freq = all_codes['code'].value_counts()
print(f"   总代码种类: {len(code_freq)}")

min_freq = 50
selected_codes = code_freq[code_freq >= min_freq].index.tolist()
print(f"   选择出现至少 {min_freq} 次的代码: {len(selected_codes)} 个")

code_to_idx = {code: i for i, code in enumerate(selected_codes)}

# 5. 为每个住院记录创建二进制向量
print("\n[5/5] 生成 EHR 向量...")
hadm_to_codes = {}
for hadm_id, group in tqdm(all_codes.groupby('hadm_id'), desc="   处理住院记录"):
    hadm_to_codes[hadm_id] = set(group['code'].values)

feature_dim = len(selected_codes)
n_samples = len(hadm_to_codes)
print(f"   特征维度: {feature_dim}")
print(f"   样本数量: {n_samples}")

X = np.zeros((n_samples, feature_dim), dtype=np.float32)
hadm_ids = []

for i, (hadm_id, codes) in enumerate(tqdm(hadm_to_codes.items(), desc="   构建特征矩阵")):
    hadm_ids.append(hadm_id)
    for code in codes:
        if code in code_to_idx:
            X[i, code_to_idx[code]] = 1.0

# 6. 保存数据
print("\n保存数据...")
np.save(os.path.join(OUTPUT_PATH, "mimic4_data.npy"), X)
print(f"   数据已保存到: {os.path.join(OUTPUT_PATH, 'mimic4_data.npy')}")

# 训练/测试集划分
print("\n创建训练/测试集划分...")
np.random.seed(42)
indices = np.random.permutation(n_samples)
train_size = int(n_samples * 0.9)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

np.save(os.path.join(OUTPUT_PATH, "train_indices.npy"), train_indices)
np.save(os.path.join(OUTPUT_PATH, "test_indices.npy"), test_indices)
print(f"   训练集大小: {len(train_indices)}")
print(f"   测试集大小: {len(test_indices)}")

# 7. 保存元数据
metadata = {
    'feature_dim': feature_dim,
    'n_samples': n_samples,
    'train_size': int(train_size),
    'test_size': int(test_indices.size),
    'min_freq': min_freq,
    'selected_codes': selected_codes,
    'code_frequencies': {str(code): int(code_freq[code]) for code in selected_codes}
}
with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n数据稀疏度: {1 - X.mean():.4f}")
print("处理完成！")
