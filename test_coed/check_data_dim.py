import numpy as np
import os

data_path = "data/mimic/mimic_data.npy"
print(f"检查数据文件: {data_path}")

if os.path.exists(data_path):
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    print(f"数据维度: {data.shape[1]}")
    print(f"配置文件中的 z_dim: 1782")
    
    if data.shape[1] != 1782:
        print(f"⚠️ 维度不匹配! 实际维度 {data.shape[1]} != 配置维度 1782")
else:
    print(f"❌ 数据文件不存在: {data_path}")