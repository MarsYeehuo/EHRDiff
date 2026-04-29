import torch
import math
from denoiser import EDMDenoiser
from model.linear_model import LinearModel

print("=" * 50)
print("测试 EDMDenoiser")
print("=" * 50)

# 创建简单的模型用于测试
class SimpleModel(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.net = torch.nn.Linear(z_dim, z_dim)
    
    def forward(self, x, time_steps, y=None):
        print(f"[DEBUG] SimpleModel - 输入 x 形状: {x.shape}")
        print(f"[DEBUG] SimpleModel - time_steps 形状: {time_steps.shape}")
        out = self.net(x)
        return out

# 参数
batch_size = 4
z_dim = 1782
sigma_min = 0.02
sigma_max = 80.0
sigma_data = 0.14

# 创建模型和 denoiser
model = SimpleModel(z_dim)
denoiser = EDMDenoiser(
    model=model,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    sigma_data=sigma_data
)

# 测试数据
x = torch.randn(batch_size, z_dim)
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
    assert out.shape == x.shape, f"输出形状错误: {out.shape} != {x.shape}"
    print(f"✓ 输出形状正确")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)