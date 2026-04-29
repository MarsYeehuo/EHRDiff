import torch
import torch.nn as nn
from omegaconf import OmegaConf
from model.linear_model import LinearModel, SinusoidalPositionEmbeddings

# 加载配置
config = OmegaConf.load("configs/mimic/train_edm.yaml")

print("=" * 50)
print("详细模型测试 - 修复版")
print("=" * 50)

print(f"\n配置参数:")
print(f"  z_dim: {config.model.network.z_dim}")
print(f"  time_dim: {config.model.network.time_dim}")
print(f"  unit_dims: {config.model.network.unit_dims}")

# 测试 SinusoidalPositionEmbeddings
print("\n1. 测试 SinusoidalPositionEmbeddings:")
time_emb = SinusoidalPositionEmbeddings(config.model.network.time_dim)
batch_size = 4
t = torch.randn(batch_size)
out = time_emb(t)
print(f"  输入 t 形状: {t.shape}")
print(f"  输出形状: {out.shape}")
print(f"  预期输出形状: [{batch_size}, {config.model.network.time_dim * 2}]")
assert out.shape[-1] == config.model.network.time_dim * 2, f"输出维度错误: {out.shape[-1]} != {config.model.network.time_dim * 2}"
print(f"  ✓ 正确")

# 测试完整的时间嵌入层
print("\n2. 测试完整的时间嵌入层:")
time_embedding = nn.Sequential(
    SinusoidalPositionEmbeddings(config.model.network.time_dim),
    nn.Linear(config.model.network.time_dim * 2, config.model.network.time_dim),
    nn.SiLU(),
    nn.Linear(config.model.network.time_dim, config.model.network.time_dim),
)
t = torch.randn(batch_size)
out = time_embedding(t)
print(f"  输入 t 形状: {t.shape}")
print(f"  输出形状: {out.shape}")
print(f"  预期输出形状: [{batch_size}, {config.model.network.time_dim}]")
assert out.shape[-1] == config.model.network.time_dim, f"输出维度错误: {out.shape[-1]} != {config.model.network.time_dim}"
print(f"  ✓ 正确")

# 测试完整的模型
print("\n3. 测试完整的 LinearModel:")
model = LinearModel(
    z_dim=config.model.network.z_dim,
    time_dim=config.model.network.time_dim,
    unit_dims=config.model.network.unit_dims,
    use_cfg=False
)

# 测试不同的输入形状
test_cases = [
    ("1D time_steps", torch.randn(batch_size)),
    ("2D time_steps", torch.randn(batch_size, 1)),
]

for case_name, t_input in test_cases:
    print(f"\n  测试用例: {case_name}")
    x = torch.randn(batch_size, config.model.network.z_dim)
    
    try:
        out = model(x, t_input)
        print(f"  ✓ 成功!")
        print(f"    输入 x 形状: {x.shape}")
        print(f"    输入 t 形状: {t_input.shape}")
        print(f"    输出形状: {out.shape}")
        print(f"    预期输出形状: [{batch_size}, {config.model.network.z_dim}]")
        assert out.shape == x.shape, f"输出形状错误: {out.shape} != {x.shape}"
        print(f"    ✓ 输出形状正确")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 50)
print("所有测试完成")
print("=" * 50)