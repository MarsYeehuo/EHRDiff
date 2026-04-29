import math
import torch
import torch.nn as nn
from einops import rearrange


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None):
        super().__init__()

        if time_emb_dim is not None:
            # 时间嵌入投影
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, dim_in),
                nn.SiLU(),
            )
        
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
        )
        
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x, time_emb=None):
        
        # 调试信息
        # print(f"[DEBUG] Block.forward - 输入 x 形状: {x.shape}, 期望 dim_in: {self.dim_in}")
        
        if time_emb is not None:
            t_emb = self.time_mlp(time_emb)  # t_emb 形状: [batch_size, dim_in]
            
            # 如果 t_emb 有额外的维度，压缩掉
            if len(t_emb.shape) > len(x.shape):
                t_emb = t_emb.squeeze(1)
            
            # 确保 t_emb 的形状与 x 匹配
            if t_emb.shape != x.shape:
                # print(f"[WARNING] 形状不匹配: t_emb {t_emb.shape}, x {x.shape}")
                # 如果 x 的维度与期望的 dim_in 不同，需要调整
                if x.shape[-1] != self.dim_in:
                    # print(f"[WARNING] x 的最后一维 {x.shape[-1]} != {self.dim_in}")
                    # 这里可能需要添加一个适配层
                    if not hasattr(self, 'adaptor'):
                        self.adaptor = nn.Linear(x.shape[-1], self.dim_in).to(x.device)
                    x = self.adaptor(x)
                    # print(f"[DEBUG] Block.forward - 调整后 x 形状: {x.shape}")
                
                # 确保 t_emb 的形状与调整后的 x 匹配
                if t_emb.shape[0] == x.shape[0] and t_emb.shape[1] == x.shape[1]:
                    # 形状匹配，可以相加
                    pass
                else:
                    # 如果 t_emb 需要广播
                    if t_emb.shape[1] == 1 and x.shape[1] > 1:
                        t_emb = t_emb.expand_as(x)
                    elif x.shape[1] == 1 and t_emb.shape[1] > 1:
                        x = x.expand_as(t_emb)
            
            # print(f"[DEBUG] Block.forward - 最终 x 形状: {x.shape}, t_emb 形状: {t_emb.shape}")
            h = x + t_emb  
        else:
            h = x
        
        out = self.out_proj(h)
        # print(f"[DEBUG] Block.forward - 输出形状: {out.shape}")
        return out


class LinearModel(nn.Module):
    def __init__(
            self, *,
            z_dim=1782, 
            time_dim=384,
            unit_dims=[1024, 384, 384, 384, 1024],

            random_fourier_features=False,
            learned_sinusoidal_dim=32,

            use_cfg=False,
            num_classes=2,
            class_dim=128,
            ):
        super().__init__()
        
        num_linears = len(unit_dims)

        if random_fourier_features:
            self.time_embedding = nn.Sequential(
                RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, is_random=True),
                nn.Linear(learned_sinusoidal_dim+1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.time_embedding = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim * 2, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        # 确保 unit_dims 的第一个和最后一个维度与 z_dim 匹配
        if unit_dims[0] != z_dim:
            # print(f"[WARNING] 调整 unit_dims[0] 从 {unit_dims[0]} 到 {z_dim}")
            unit_dims[0] = z_dim
        if unit_dims[-1] != z_dim:
            # print(f"[WARNING] 调整 unit_dims[-1] 从 {unit_dims[-1]} 到 {z_dim}")
            unit_dims[-1] = z_dim

        self.block_in = Block(dim_in=z_dim, dim_out=unit_dims[0], time_emb_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears-1):
            self.block_mid.append(Block(dim_in=unit_dims[i], dim_out=unit_dims[i+1], time_emb_dim=time_dim))
        self.block_out = Block(dim_in=unit_dims[-1], dim_out=z_dim, time_emb_dim=time_dim)

        ### Classifier-free 
        self.label_dim = num_classes
        self.use_cfg = use_cfg
        if use_cfg:
            self.class_emb = nn.Embedding(self.label_dim if not use_cfg else self.label_dim + 1, class_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(class_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )    

    def forward(self, x, time_steps, labels=None):
        
        # 调试信息
        # print(f"[DEBUG] LinearModel.forward - 输入 x 形状: {x.shape}")
        # print(f"[DEBUG] LinearModel.forward - time_steps 形状: {time_steps.shape if torch.is_tensor(time_steps) else time_steps}")
        
        time_steps = time_steps.float()
        # time_steps 形状: [batch_size] 或 [batch_size, 1]
        if len(time_steps.shape) == 2:
            time_steps = time_steps.squeeze(-1)
        
        t_emb = self.time_embedding(time_steps)  # t_emb 形状: [batch_size, time_dim]
        # print(f"[DEBUG] LinearModel.forward - t_emb 形状: {t_emb.shape}")
        
        if self.use_cfg:
            class_emb = self.class_mlp(self.class_emb(labels))
            t_emb += class_emb 

        # 保存原始输入用于调试
        original_x = x
        
        x = self.block_in(x, t_emb)
        # print(f"[DEBUG] LinearModel.forward - after block_in: {x.shape}")

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for i, block in enumerate(self.block_mid):
                x = block(x, t_emb)
                # print(f"[DEBUG] LinearModel.forward - after block_mid[{i}]: {x.shape}")

        x = self.block_out(x, t_emb)
        # print(f"[DEBUG] LinearModel.forward - after block_out: {x.shape}")
        
        # 确保输出形状与输入一致
        assert x.shape == original_x.shape, f"输出形状 {x.shape} 与输入形状 {original_x.shape} 不匹配"
        
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # 输出维度: [batch_size, dim * 2]


class RandomOrLearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered