"""
MIMIC-IV 预处理：ICD 代码 + 人口统计学特征 合并版本
用法：python data_preprocessing/process_mimic4_with_demographics.py
"""

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sqlalchemy import create_engine

DB_HOST = "localhost"
DB_PORT = "5433"
DB_USER = "mimicuser"
DB_PASSWORD = "knowlabMIMIC"
DB_NAME = "mimic"
SCHEMA = "mimiciv_hosp"

OUTPUT_PATH = "data/mimic4"
os.makedirs(OUTPUT_PATH, exist_ok=True)

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("=" * 50)
print("MIMIC-IV 预处理：ICD 代码 + 人口学特征")
print("=" * 50)

# =====================
# 1. 读取住院记录 + 人口学信息
# =====================
print("\n[1/6] 读取住院记录及人口学信息...")
query = f"""
SELECT
    a.subject_id, a.hadm_id,
    p.gender,
    p.anchor_age,
    a.admission_type,
    a.insurance
FROM {SCHEMA}.admissions a
JOIN {SCHEMA}.patients p ON a.subject_id = p.subject_id
"""
admissions = pd.read_sql_query(query, engine)
admissions['subject_id'] = admissions['subject_id'].astype(str)
admissions['hadm_id'] = admissions['hadm_id'].astype(str)
print(f"   共 {len(admissions)} 条记录")
print(f"   性别: {admissions['gender'].value_counts().to_dict()}")
print(f"   年龄: {admissions['anchor_age'].describe()[['min','max','mean']].to_dict()}")

# =====================
# 2. 人口学特征编码
# =====================
print("\n[2/6] 编码人口学特征...")

# 性别: M=0, F=1
admissions['dem_gender'] = (admissions['gender'] == 'F').astype(np.float32)

# 年龄: 归一化到 [0, 1]
age_min, age_max = 18.0, 91.0
admissions['dem_age'] = ((admissions['anchor_age'].clip(age_min, age_max) - age_min) / (age_max - age_min)).astype(np.float32)

# 入院类型: one-hot
def categorize_admission(atype):
    if atype is None or pd.isna(atype):
        return 'other'
    atype = str(atype).upper()
    if 'EMER' in atype:
        return 'emergency'
    elif 'OBSERV' in atype:
        return 'observation'
    elif 'URGENT' in atype:
        return 'urgent'
    else:
        return 'other'

admissions['adm_cat'] = admissions['admission_type'].apply(categorize_admission)
adm_dummies = pd.get_dummies(admissions['adm_cat'], prefix='adm')
for col in ['adm_emergency', 'adm_observation', 'adm_urgent', 'adm_other']:
    if col not in adm_dummies.columns:
        adm_dummies[col] = 0.0
admissions['dem_adm_emerg'] = adm_dummies['adm_emergency'].astype(np.float32)
admissions['dem_adm_obs'] = adm_dummies['adm_observation'].astype(np.float32)
admissions['dem_adm_urg'] = adm_dummies['adm_urgent'].astype(np.float32)
admissions['dem_adm_other'] = adm_dummies['adm_other'].astype(np.float32)

# 保险: one-hot
def categorize_insurance(ins):
    if ins is None or pd.isna(ins):
        return 'other'
    ins = str(ins).strip()
    if 'MEDICARE' in ins.upper():
        return 'medicare'
    elif 'MEDICAID' in ins.upper():
        return 'medicaid'
    else:
        return 'other'

admissions['ins_cat'] = admissions['insurance'].apply(categorize_insurance)
ins_dummies = pd.get_dummies(admissions['ins_cat'], prefix='ins')
for col in ['ins_medicare', 'ins_medicaid', 'ins_other']:
    if col not in ins_dummies.columns:
        ins_dummies[col] = 0.0
admissions['dem_ins_medicare'] = ins_dummies['ins_medicare'].astype(np.float32)
admissions['dem_ins_medicaid'] = ins_dummies['ins_medicaid'].astype(np.float32)
admissions['dem_ins_other'] = ins_dummies['ins_other'].astype(np.float32)

# 人口学列名
DEM_COLS = [
    'dem_gender', 'dem_age',
    'dem_adm_emerg', 'dem_adm_obs', 'dem_adm_urg', 'dem_adm_other',
    'dem_ins_medicare', 'dem_ins_medicaid', 'dem_ins_other'
]
N_DEM = len(DEM_COLS)
print(f"   人口学特征维度: {N_DEM}")
print(f"   列名: {DEM_COLS}")

# =====================
# 3. 读取诊断代码
# =====================
print("\n[3/6] 读取诊断代码...")
diagnoses = pd.read_sql_query(
    f"SELECT subject_id, hadm_id, icd_code, icd_version FROM {SCHEMA}.diagnoses_icd",
    engine,
)
diagnoses['subject_id'] = diagnoses['subject_id'].astype(str)
diagnoses['hadm_id'] = diagnoses['hadm_id'].astype(str)

def process_icd_code(row):
    code = row['icd_code']
    if pd.isna(code):
        return None
    return str(code).strip()[:3]

diagnoses['icd_code_3d'] = diagnoses.apply(process_icd_code, axis=1)
diagnoses = diagnoses.dropna(subset=['icd_code_3d'])
print(f"   共 {len(diagnoses)} 条诊断记录")

# =====================
# 4. 读取手术代码
# =====================
print("\n[4/6] 读取手术代码...")
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
    all_codes = pd.concat([
        diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'}),
        procedures[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})
    ])
except Exception as e:
    print(f"   手术代码表读取失败 ({e})，仅使用诊断代码")
    all_codes = diagnoses[['hadm_id', 'icd_code_3d']].rename(columns={'icd_code_3d': 'code'})

# =====================
# 5. 构建代码字典
# =====================
print("\n[5/6] 构建代码字典...")
code_freq = all_codes['code'].value_counts()
min_freq = 50
selected_codes = code_freq[code_freq >= min_freq].index.tolist()
N_CODE = len(selected_codes)
print(f"   ICD 代码维度: {N_CODE} (出现 >= {min_freq} 次)")
code_to_idx = {code: i for i, code in enumerate(selected_codes)}

# =====================
# 6. 构建合并特征矩阵
# =====================
print("\n[6/6] 构建合并特征矩阵 (ICD + 人口学)...")

# 按 hadm_id 建立人口学索引
hadm_to_dem = {}
for _, row in admissions.iterrows():
    hadm_to_dem[row['hadm_id']] = {col: row[col] for col in DEM_COLS}

# 按 hadm_id 建立代码集合
hadm_to_codes = {}
for hadm_id, group in tqdm(all_codes.groupby('hadm_id'), desc="   处理住院记录"):
    hadm_to_codes[hadm_id] = set(group['code'].values)

# 取交集：既有代码又有人口学的样本
common_hadm = sorted(set(hadm_to_codes.keys()) & set(hadm_to_dem.keys()))
n_samples = len(common_hadm)
total_dim = N_CODE + N_DEM
print(f"   合并后样本数: {n_samples}")
print(f"   总特征维度: {N_CODE}(ICD) + {N_DEM}(人口学) = {total_dim}")

# 填充特征矩阵
X = np.zeros((n_samples, total_dim), dtype=np.float32)
hadm_ids = []

for i, hadm_id in enumerate(tqdm(common_hadm, desc="   构建特征矩阵")):
    hadm_ids.append(hadm_id)

    # ICD 代码（二进制）
    for code in hadm_to_codes[hadm_id]:
        if code in code_to_idx:
            X[i, code_to_idx[code]] = 1.0

    # 人口学特征
    dem = hadm_to_dem[hadm_id]
    for col in DEM_COLS:
        X[i, N_CODE + DEM_COLS.index(col)] = dem[col]

# =====================
# 保存
# =====================
print("\n保存数据...")
# 保存完整数据
np.save(os.path.join(OUTPUT_PATH, "mimic4_dem_data.npy"), X)
print(f"   数据已保存: {os.path.join(OUTPUT_PATH, 'mimic4_dem_data.npy')}")

# 训练/测试划分
np.random.seed(42)
indices = np.random.permutation(n_samples)
train_size = int(n_samples * 0.9)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

np.save(os.path.join(OUTPUT_PATH, "train_indices.npy"), train_indices)
np.save(os.path.join(OUTPUT_PATH, "test_indices.npy"), test_indices)
print(f"   训练集: {len(train_indices)}, 测试集: {len(test_indices)}")

# 列映射信息
col_meta = {
    'n_code': N_CODE,
    'n_dem': N_DEM,
    'total_dim': total_dim,
    'dem_cols': DEM_COLS,
    'selected_codes': selected_codes,
    'code_frequencies': {str(code): int(code_freq[code]) for code in selected_codes},
}
with open(os.path.join(OUTPUT_PATH, "dem_metadata.json"), 'w') as f:
    json.dump(col_meta, f, indent=2)
print(f"   元数据已保存")

# 数据统计
sparsity = 1 - X[:, :N_CODE].mean()
print(f"\nICD 代码稀疏度: {sparsity:.4f}")
print(f"性别比例 F: {X[:, N_CODE].mean():.3f}")
print(f"平均年龄(归一化): {X[:, N_CODE+1].mean():.3f} (≈ {age_min + X[:, N_CODE+1].mean() * (age_max - age_min):.1f} 岁)")
print(f"入院类型: emerg={X[:, N_CODE+2].mean():.3f} obs={X[:, N_CODE+3].mean():.3f} urg={X[:, N_CODE+4].mean():.3f}")
print(f"保险: medicare={X[:, N_CODE+6].mean():.3f} medicaid={X[:, N_CODE+7].mean():.3f}")
print("\n处理完成！")
