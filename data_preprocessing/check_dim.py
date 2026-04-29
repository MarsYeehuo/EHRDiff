import numpy as np
import os

data_path = "data/mimic4/mimic4_data.npy"

if os.path.exists(data_path):
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    print(f"样本数量: {data.shape[0]}")
    print(f"特征维度: {data.shape[1]}")
    
    # 查看一些统计信息
    print(f"数据类型: {data.dtype}")
    print(f"数值范围: [{data.min()}, {data.max()}]")
    print(f"稀疏度: {1 - data.mean():.4f}")