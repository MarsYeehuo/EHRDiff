
import math
import numpy as np
import torch
import torch.nn as nn


class NaiveDenoiser(nn.Module):
    def __init__(self,
                model,
                ):

        super().__init__()
        self.model = model

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        return self.model(x, sigma.reshape(-1), y)


class EDMDenoiser(nn.Module):
    def __init__(self,
                model,
                sigma_min,
                sigma_max,
                sigma_data=math.sqrt(1. / 3)
                ):

        super().__init__()

        self.sigma_data = sigma_data
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        
        # 确保 sigma 是正确形状
        if len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(-1)
        
        c_skip = self.sigma_data ** 2. / (sigma ** 2. + self.sigma_data ** 2.)
        c_out = sigma * self.sigma_data / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_in = 1. / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_noise = 0.25 * torch.log(sigma)
        
        # 关键修复：确保 c_in 的形状可以广播到 x
        # c_in 当前形状可能是 [batch_size] 或 [batch_size, 1]
        if len(c_in.shape) == 1:
            c_in = c_in.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        elif len(c_in.shape) == 2 and c_in.shape[1] != 1:
            c_in = c_in.unsqueeze(-1)  # 如果第二维不是1，添加一个维度
        
        # 确保 c_in 的形状是 [batch_size, 1]
        # print(f"[DEBUG] EDMDenoiser - 输入 x 形状: {x.shape}")
        # print(f"[DEBUG] EDMDenoiser - sigma 形状: {sigma.shape}")
        # print(f"[DEBUG] EDMDenoiser - c_in 形状: {c_in.shape}")
        # print(f"[DEBUG] EDMDenoiser - c_in 需要广播到: {x.shape}")
        
        # 现在 c_in 是 [batch_size, 1]，可以广播到 [batch_size, feature_dim]
        scaled_x = c_in * x
        # print(f"[DEBUG] EDMDenoiser - scaled_x 形状: {scaled_x.shape}")
        
        # 确保 c_noise 是正确形状
        c_noise_flat = c_noise.reshape(-1)
        # print(f"[DEBUG] EDMDenoiser - c_noise 形状: {c_noise_flat.shape}")
        
        out = self.model(scaled_x, c_noise_flat, y)
        # print(f"[DEBUG] EDMDenoiser - model 输出形状: {out.shape}")
        
        # 确保 c_skip 和 c_out 可以广播
        if len(c_skip.shape) == 1:
            c_skip = c_skip.unsqueeze(-1)
        if len(c_out.shape) == 1:
            c_out = c_out.unsqueeze(-1)
        
        x_denoised = c_skip * x + c_out * out
        # print(f"[DEBUG] EDMDenoiser - 输出 x_denoised 形状: {x_denoised.shape}")
        
        return x_denoised


class VDenoiser(nn.Module):
    def __init__(
                self,
                model
                ):

        super().__init__()
        self.model = model

    def _sigma_inv(self, sigma):
        return 2. * torch.arccos(1. / (1. + sigma ** 2.).sqrt()) / np.pi

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        c_skip = 1. / (sigma ** 2. + 1.)
        c_out = sigma / torch.sqrt(1. + sigma ** 2.)
        c_in = 1. / torch.sqrt(1. + sigma ** 2.)
        c_noise = self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
    

class VESDEDenoiser(nn.Module):
    def __init__(self,
                sigma_min,
                sigma_max,
                model,
                ):

        super().__init__()

        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        
        x = x.to(torch.float32)

        c_skip = 1. 
        ### Essential adjustment for mimic data
        # c_skip = 0.11 ** 2. / \
        #     (sigma ** 2. + 0.11 ** 2.)
        
        c_out = sigma
        c_in = 1.
        c_noise = torch.log(sigma / 2.)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised

    

class VPSDEDenoiser(nn.Module):
    def __init__(
                self,
                beta_min,
                beta_d,
                M,
                eps_t,
                model
                ):

        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d
        ### https://github.com/NVlabs/edm/blob/main/training/networks.py
        self.sigma_min = float(self.sigma(eps_t))
        self.sigma_max = float(self.sigma(1))

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def _sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def forward(self, x, sigma, y=None):

        x = x.to(torch.float32)
        
        c_skip = 1.
        ### Essential adjustment for mimic data
        # c_skip = 0.13 ** 2. / \
        #     (sigma ** 2. + 0.13 ** 2.)
        
        c_out = -sigma
        c_in = 1. / torch.sqrt(sigma ** 2. + 1.)
        c_noise = (self.M-1) * self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
