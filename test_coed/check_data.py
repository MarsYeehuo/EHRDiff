import numpy as np
import os

data_path = "data/mimic/mimic_data.npy"
train_idx_path = "data/mimic/train_indices.npy"
test_idx_path = "data/mimic/test_indices.npy"

print("验证数据文件...")
print(f"数据文件存在: {os.path.exists(data_path)}")
print(f"训练索引存在: {os.path.exists(train_idx_path)}")
print(f"测试索引存在: {os.path.exists(test_idx_path)}")

if os.path.exists(data_path):
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{data.min()}, {data.max()}]")
    print(f"非零元素比例: {np.mean(data > 0):.4f}")
    
    if os.path.exists(train_idx_path):
        train_idx = np.load(train_idx_path)
        print(f"训练集大小: {len(train_idx)}")
        train_data = data[train_idx]
        print(f"训练集非零比例: {np.mean(train_data > 0):.4f}")
    
    if os.path.exists(test_idx_path):
        test_idx = np.load(test_idx_path)
        print(f"测试集大小: {len(test_idx)}")
        test_data = data[test_idx]
        print(f"测试集非零比例: {np.mean(test_data > 0):.4f}")