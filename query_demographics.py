"""
查询 MIMIC-IV 人口统计学特征，了解可用字段和数据分布
用法：python query_demographics.py
"""

import pandas as pd
from sqlalchemy import create_engine

DB_HOST = "localhost"
DB_PORT = "5433"
DB_USER = "mimicuser"
DB_PASSWORD = "knowlabMIMIC"
DB_NAME = "mimic"
SCHEMA = "mimiciv_hosp"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("=" * 50)
print("1. patients 表 - 性别/年龄")
print("=" * 50)
patients = pd.read_sql_query(
    f"SELECT gender, anchor_age, COUNT(*) as cnt FROM {SCHEMA}.patients GROUP BY gender, anchor_age ORDER BY gender, anchor_age LIMIT 20",
    engine,
)
print(patients.head(20))
print(f"... (共 {patients.shape[0]} 行)")

print("\n性别分布:")
gender = pd.read_sql_query(
    f"SELECT gender, COUNT(*) as cnt, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct FROM {SCHEMA}.patients GROUP BY gender",
    engine,
)
print(gender)

print("\n年龄统计:")
age = pd.read_sql_query(
    f"SELECT MIN(anchor_age) as min_age, MAX(anchor_age) as max_age, AVG(anchor_age)::numeric(10,2) as avg_age, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY anchor_age)::numeric(10,2) as median_age FROM {SCHEMA}.patients",
    engine,
)
print(age)

print("\n" + "=" * 50)
print("2. admissions 表 - 入院类型/种族/保险")
print("=" * 50)
adm_types = pd.read_sql_query(
    f"SELECT admission_type, COUNT(*) as cnt, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct FROM {SCHEMA}.admissions GROUP BY admission_type ORDER BY cnt DESC",
    engine,
)
print("入院类型:", adm_types.to_string(index=False))

race = pd.read_sql_query(
    f"SELECT race, COUNT(*) as cnt FROM {SCHEMA}.admissions WHERE race IS NOT NULL GROUP BY race ORDER BY cnt DESC LIMIT 15",
    engine,
)
print("\n种族分布(top 15):", race.to_string(index=False))

insurance = pd.read_sql_query(
    f"SELECT insurance, COUNT(*) as cnt, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct FROM {SCHEMA}.admissions GROUP BY insurance ORDER BY cnt DESC",
    engine,
)
print("\n保险类型:", insurance.to_string(index=False))
