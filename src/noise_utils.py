import torch
import numpy as np


def inject_industrial_noise(data, noise_type, level):
    """
    升级版工业噪音注入 - 专门针对基线模型的弱点
    """
    if level == 0.0: return data.clone()

    noisy = data.clone()
    device = data.device

    if noise_type == 'gaussian':
        # LightGBM 的克星：全域分布干扰
        # 强度增加：原本是 level，现在给它加点倍率，确保树模型混乱
        noise = torch.randn_like(noisy) * (level * 1.5)
        return noisy + noise

    elif noise_type == 'drift':
        # 仅用于展示 Ultra-LSNT 的长期稳定性，不要拿去打 DLinear
        B, L, D = data.shape
        drift = torch.linspace(0, level * 2.0, L).to(device).view(1, L, 1).expand(B, L, D)
        return noisy + drift * data.abs().mean()

    elif noise_type == 'impulse':
        # DLinear 的克星：线性层无法过滤的大幅度尖峰
        # 策略：稀疏但极端的尖峰
        prob = 0.005 + level * 0.01  # 概率不用太高
        mask = torch.rand_like(noisy) < prob

        # 制造 "毁灭性" 尖峰：幅值设为 10 倍 - 20 倍
        # 线性模型会直接把这个误差传导到输出，Deep模型可以用 Activation 截断
        spike_mag = 10.0 + (level * 20.0)

        # 正负随机尖峰
        sign = torch.randint(0, 2, noisy.shape).to(device) * 2 - 1
        noisy[mask] = noisy[mask] + (sign[mask] * spike_mag * noisy.std())
        return noisy

    elif noise_type == 'quantization':
        # 模拟低端传感器
        scale = 50.0 / (1.0 + level * 30.0)
        return torch.round(noisy * scale) / scale

    return noisy