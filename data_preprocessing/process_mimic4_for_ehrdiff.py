import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

# 配置路径
MIMIC4_PATH = "data/mimic4"  # 请修改为您的 MIMIC-IV 路径
OUTPUT_PATH = "data/mimic4"

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 50)
print("开始处理 MIMIC-IV 数据集")
print("=" * 50)

# 1. 读取住院记录
print("\n[1/5] 读取住院记录...")
# MIMIC-IV 的主表是 hosp 模块下的 admissions.csv
admissions = pd.read_csv(
    os.path.join(MIMIC4_PATH,  "admissions.csv"),
    dtype={'subject_id': str, 'hadm_id': str}
)
print(f"   共 {len(admissions)} 条住院记录")

# 2. 读取诊断代码
print("\n[2/5] 读取诊断代码...")
# MIMIC-IV 的诊断代码在 hosp 模块下的 diagnoses_icd.csv
diagnoses = pd.read_csv(
    os.path.join(MIMIC4_PATH,  "diagnoses_icd.csv"),
    dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str, 'icd_version': int}
)

# 处理 ICD 代码
def process_icd_code(row):
    """处理 ICD 代码，统一为前3位"""
    code = row['icd_code']
    version = row['icd_version']
    
    if pd.isna(code):
        return None
    
    code = str(code).strip()
    # 截断到前3位
    code_3d = code[:3]
    
    # 可以选择添加版本前缀来区分 ICD-9 和 ICD-10
    # return f"v{version}_{code_3d}"
    return code_3d

diagnoses['icd_code_3d'] = diagnoses.apply(process_icd_code, axis=1)
diagnoses = diagnoses.dropna(subset=['icd_code_3d'])
print(f"   共 {len(diagnoses)} 条诊断记录")
print(f"   诊断代码种类（截断后）: {diagnoses['icd_code_3d'].nunique()}")

# 3. 读取手术代码（可选，如果存在）
print("\n[3/5] 读取手术代码...")
procedures_path = os.path.join(MIMIC4_PATH,  "procedures_icd.csv")
if os.path.exists(procedures_path):
    procedures = pd.read_csv(
        procedures_path,
        dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str, 'icd_version': int}
    )
    procedures['icd_code_3d'] = procedures.apply(process_icd_code, axis=1)
    procedures = procedures.dropna(subset=['icd_code_3d'])
    print(f"   共 {len(procedures)} 条手术记录")
    print(f"   手术代码种类: {procedures['icd_code_3d'].nunique()}")
    
    # 合并诊断和手术代码
    all_codes = pd.concat([
        diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'}),
        procedures[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})
    ])
else:
    print("   未找到手术代码文件，只使用诊断代码")
    all_codes = diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})

# 4. 构建代码字典
print("\n[4/5] 构建代码字典...")

# 统计代码频率
code_freq = all_codes['code'].value_counts()
print(f"   总代码种类: {len(code_freq)}")

# 选择最常见的代码
# MIMIC-III 用了 1782 维，MIMIC-IV 可能更多
# 可以选择 top-K 或使用所有出现一定次数以上的代码
min_freq = 50  # 至少出现50次的代码
selected_codes = code_freq[code_freq >= min_freq].index.tolist()
print(f"   选择出现至少 {min_freq} 次的代码: {len(selected_codes)} 个")

# 如果需要固定维度，可以取前 N 个
# N = 1782  # 如果想保持与 MIMIC-III 相同的维度
# if len(selected_codes) > N:
#     selected_codes = selected_codes[:N]
#     print(f"   限制为前 {N} 个代码")

# 创建代码映射
code_to_idx = {code: i for i, code in enumerate(selected_codes)}

# 5. 为每个住院记录创建二进制向量
print("\n[5/5] 生成 EHR 向量...")

# 获取所有有代码的住院记录
valid_admissions = all_codes['hadm_id'].unique()
print(f"   有代码的住院记录数: {len(valid_admissions)}")

# 创建每个住院记录的代码集合
hadm_to_codes = {}
for hadm_id, group in tqdm(all_codes.groupby('hadm_id'), desc="   处理住院记录"):
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

# 6. 保存数据
print("\n保存数据...")

# 保存完整数据集
np.save(os.path.join(OUTPUT_PATH, "mimic4_data.npy"), X)
print(f"   数据已保存到: {os.path.join(OUTPUT_PATH, 'mimic4_data.npy')}")

# 创建训练/测试集划分
print("\n创建训练/测试集划分...")
np.random.seed(42)
indices = np.random.permutation(n_samples)

# 90/10 划分
train_size = int(n_samples * 0.9)
test_size = n_samples - train_size

train_indices = indices[:train_size]
test_indices = indices[train_size:]

np.save(os.path.join(OUTPUT_PATH, "train_indices.npy"), train_indices)
np.save(os.path.join(OUTPUT_PATH, "test_indices.npy"), test_indices)
print(f"   训练集大小: {len(train_indices)}")
print(f"   测试集大小: {len(test_indices)}")

# 7. 保存元数据
print("\n保存元数据...")
metadata = {
    'feature_dim': feature_dim,
    'n_samples': n_samples,
    'train_size': train_size,
    'test_size': test_size,
    'min_freq': min_freq,
    'selected_codes': selected_codes,
    'code_frequencies': {code: int(code_freq[code]) for code in selected_codes}
}
import json
with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n处理完成！")