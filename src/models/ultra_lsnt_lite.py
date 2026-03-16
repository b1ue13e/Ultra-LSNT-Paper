"""
Ultra-LSNT-Lite: 基于TimeMixer效率优势优化的轻量化版本
结合TimeMixer的多尺度混合框架和Ultra-LSNT的稀疏MoE优势
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

class TimeMixerInspiredEncoder(nn.Module):
    """
    TimeMixer启发的多尺度编码器
    核心思想：多尺度卷积捕获不同时间粒度特征
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, scales: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if scales is None:
            scales = [1, 2, 4]  # 与TimeMixer相同的多尺度因子
        
        self.scales = scales
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度卷积（轻量化设计）
        self.multiscale_convs = nn.ModuleList()
        for scale in scales:
            kernel_size = max(3, scale * 2)
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            )
            self.multiscale_convs.append(conv)
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 时序注意力（轻量化）- 作为类属性初始化
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x_t = x_proj.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        multiscale_features = []
        for conv in self.multiscale_convs:
            scale_feat = conv(x_t)  # (batch, hidden_dim, seq_len)
            # 确保序列长度一致
            if scale_feat.size(-1) != seq_len:
                diff = scale_feat.size(-1) - seq_len
                if diff > 0:
                    # 裁剪多余的
                    start = diff // 2
                    scale_feat = scale_feat[:, :, start:start+seq_len]
                else:
                    # 填充不足的
                    padding = -diff
                    scale_feat = F.pad(scale_feat, (padding//2, padding - padding//2))
            
            multiscale_features.append(scale_feat.transpose(1, 2))  # (batch, seq_len, hidden_dim)
        
        # 尺度融合
        fused = torch.cat(multiscale_features, dim=-1)  # (batch, seq_len, hidden_dim * len(scales))
        fused = self.scale_fusion(fused)  # (batch, seq_len, hidden_dim)
        
        # 时序注意力（轻量化）
        attn_out, _ = self.temporal_attn(fused, fused, fused)
        fused = fused + attn_out
        
        # 聚合为固定维度（与Ultra-LSNT兼容）
        aggregated = fused.mean(dim=1)  # (batch, hidden_dim)
        
        return aggregated

class LiteSparseMoERouter(nn.Module):
    """
    轻量化稀疏MoE路由器
    减少专家数量，简化网络结构
    """
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # 简化专家网络（2层MLP）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim = x.shape
        device = x.device
        
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 稀疏计算
        output = torch.zeros(batch_size, dim, device=device, dtype=x.dtype)
        flat_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        for expert_idx in range(self.num_experts):
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue
            token_indices = batch_indices[mask]
            token_weights = flat_weights[mask]
            expert_output = self.experts[expert_idx](x[token_indices])
            output.index_add_(0, token_indices, expert_output * token_weights.unsqueeze(-1))
        
        # 计算辅助损失（负载平衡）
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_usage = router_probs.mean(dim=0)
        expert_selection = expert_mask.sum(dim=1).mean(dim=0) / self.top_k
        load_balance_loss = self.num_experts * (expert_usage * expert_selection).sum()
        
        return output, load_balance_loss

class UltraLSNTLiteBlock(nn.Module):
    """
    Ultra-LSNT-Lite块
    结合TimeMixer多尺度特征和稀疏MoE
    """
    def __init__(self, hidden_dim: int = 128, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.moe_router = LiteSparseMoERouter(hidden_dim, num_experts, top_k)
        
        # 简化前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.input_norm(x)
        
        # MoE处理
        x_moe, aux_loss = self.moe_router(x)
        
        # 前馈网络
        x_ffn = self.ffn(x_moe)
        x = self.output_norm(x_moe + x_ffn)
        x = self.dropout(x)
        x = x + residual
        
        return x, aux_loss

class UltraLSNTLiteForecaster(nn.Module):
    """
    Ultra-LSNT-Lite时序预测模型
    结合TimeMixer效率和Ultra-LSNT稀疏性
    """
    def __init__(self, config: dict):
        super().__init__()
        input_dim = config['input_dim']
        hidden_dim = config.get('hidden_dim', 128)
        seq_len = config['seq_len']
        pred_len = config['pred_len']
        num_blocks = config.get('num_blocks', 2)  # 减少块数
        num_experts = config.get('num_experts', 4)  # 减少专家数
        top_k = config.get('top_k', 2)  # 减少top-k
        scales = config.get('scales', [1, 2, 4])
        
        # TimeMixer风格编码器
        self.encoder = TimeMixerInspiredEncoder(input_dim, hidden_dim, scales)
        
        # Ultra-LSNT-Lite块
        self.blocks = nn.ModuleList([
            UltraLSNTLiteBlock(hidden_dim, num_experts, top_k)
            for _ in range(num_blocks)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, pred_len)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码
        x = self.encoder(x)  # (batch, hidden_dim)
        
        # 稀疏MoE处理
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss
        
        # 解码
        output = self.decoder(x)  # (batch, pred_len)
        
        return output, total_aux_loss

if __name__ == "__main__":
    # 测试模型
    config = {
        'input_dim': 8,
        'hidden_dim': 128,
        'seq_len': 96,
        'pred_len': 24,
        'num_blocks': 2,
        'num_experts': 4,
        'top_k': 2,
        'scales': [1, 2, 4]
    }
    
    model = UltraLSNTLiteForecaster(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("Ultra-LSNT-Lite 模型测试")
    print("=" * 60)
    print(f"模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小 (MB): {total_params * 4 / (1024*1024):.2f}")
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, config['seq_len'], config['input_dim'])
    output, aux_loss = model(test_input)
    
    print(f"\n前向传播测试:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  辅助损失: {aux_loss.item():.4f}")
    print(f"\n✅ 模型创建成功！")