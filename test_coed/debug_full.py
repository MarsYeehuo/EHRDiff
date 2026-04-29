import torch
from omegaconf import OmegaConf
from denoiser import EDMDenoiser
from model.linear_model import LinearModel

# 加载配置
config = OmegaConf.load("configs/mimic/train_edm.yaml")

print("=" * 50)
print("完整调试")
print("=" * 50)

# 创建模型
model = LinearModel(
    z_dim=config.model.network.z_dim,
    time_dim=config.model.network.time_dim,
    unit_dims=config.model.network.unit_dims,
    use_cfg=False
)

# 创建 denoiser
denoiser = EDMDenoiser(
    model=model,
    sigma_min=config.model.params.sigma_min,
    sigma_max=config.model.params.sigma_max,
    sigma_data=config.model.params.sigma_data
)

# 测试数据
batch_size = 4
x = torch.randn(batch_size, config.model.network.z_dim)
sigma = torch.randn(batch_size, 1).abs()  # sigma 应该是正数
y = None

print(f"\n测试数据:")
print(f"x 形状: {x.shape}")
print(f"sigma 形状: {sigma.shape}")

# 测试 denoiser
print("\n" + "-" * 30)
print("测试 denoiser")
print("-" * 30)

try:
    out = denoiser(x, sigma, y)
    print(f"\n✓ 成功!")
    print(f"输出形状: {out.shape}")
    print(f"预期输出形状: {x.shape}")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)