"""
批量采样脚本：用无条件 2083 维 EDM 模型生成大样本集。

背景：generate_base.py 每次调用只生成一个 batch（batch_size 必须 == n_samples
才能生成目标数量）。本脚本分块循环采样，可生成任意数量（如 50K–100K）样本，
二值化后保存，供隐私评估等后续分析使用。

用法（服务器）：
    python generate_large_batch.py \
        --config configs/mimic4/sample_edm_2083.yaml \
        --n_samples 100000 --batch_size 20000 \
        --out results/mimic4/samples/all_x_large.npy
"""
import os
import sys
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from model.ema import ExponentialMovingAverage
from denoiser import EDMDenoiser
from model.linear_model import LinearModel
from samplers import ablation_sampler


def load_model(config, device):
    """与 generate_base.get_model 相同的加载逻辑，但去掉 DDP 包装（单卡推理）。"""
    model = EDMDenoiser(
        model=LinearModel(**config.model.network).to(device),
        **config.model.params)
    state = torch.load(config.model.ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=True)
    if config.model.use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='采样配置（sample_edm_2083.yaml）')
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=20000,
                        help='单块采样大小，受 GPU 显存限制')
    parser.add_argument('--out', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()

    config = OmegaConf.load(opt.config)
    # 与 generate_base 相同的字符串→None 归一化
    if config.data.n_classes == 'None':
        config.data.n_classes = None
    if config.sampler.guid_scale == 'None':
        config.sampler.guid_scale = None
    if config.test.labels == 'None':
        config.test.labels = None

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    device = opt.device

    z_dim = int(config.model.network.z_dim)
    model = load_model(config, device)
    print(f'模型加载完成: z_dim={z_dim}, ckpt={config.model.ckpt}', flush=True)

    def sampler(x, y=None):
        return ablation_sampler(x, y, model, **config.sampler)

    n_chunks = (opt.n_samples + opt.batch_size - 1) // opt.batch_size
    chunks = []
    for i in range(n_chunks):
        b = min(opt.batch_size, opt.n_samples - i * opt.batch_size)
        x = torch.randn((b, z_dim), device=device)
        with torch.no_grad():
            x = sampler(x)
        x = x.cpu().numpy()
        x = np.rint(np.clip(x, 0, 1)).astype(np.float32)
        chunks.append(x)
        print(f'[chunk {i+1}/{n_chunks}] {x.shape[0]} samples (acc '
              f'{sum(c.shape[0] for c in chunks)})', flush=True)

    all_x = np.concatenate(chunks, axis=0)
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    np.save(opt.out, all_x)
    print(f'saved: {opt.out} shape={all_x.shape} dtype={all_x.dtype}', flush=True)


if __name__ == '__main__':
    main()