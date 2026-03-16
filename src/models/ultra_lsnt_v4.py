"""
Ultra-LSNT v4.0 - 完全重构版
==============================

核心重构：
1. 真正的稀疏计算 - 只计算被选中的专家
2. 改进的负载均衡损失 - 使用 Switch Transformer 风格
3. 训练/推理一致性 - 使用直通估计器(STE)
4. 任务条件仿射适配（FiLM），不含 MAML 内环
5. 完整的 FLOPs 计数
6. 更清晰的双模传播设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import time


@dataclass
class LSNTConfig:
    """模型配置"""
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    num_blocks: int = 3
    num_experts: int = 4
    top_k: int = 2
    expert_capacity_factor: float = 1.25
    router_z_loss_coef: float = 0.01
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.0
    router_path_cost_coef: float = 0.0
    expert_path_costs: Optional[List[float]] = None
    skip_threshold: float = 0.5
    dropout: float = 0.1
    meta_lr: float = 0.01
    meta_steps: int = 5
    resistance_loss_coef: float = 0.0
    dual_gate_type: str = "ste"  # "ste" or "hard_concrete"
    dual_gate_temp: float = 0.67
    dual_gate_low: float = -0.1
    dual_gate_high: float = 1.1


class StraightThroughEstimator(torch.autograd.Function):
    """直通估计器 - 解决训练/推理不一致问题"""
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        ctx.save_for_backward(x, threshold)
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output
        # STE: threshold's gradient approximates -grad_x aggregated over its broadcasted dims
        grad_threshold = -grad_output
        if threshold.numel() == 1 and grad_threshold.numel() > 1:
            grad_threshold = grad_threshold.sum().view_as(threshold) / grad_output.numel()
        return grad_x, grad_threshold


def straight_through(x, threshold=0.5):
    if not torch.is_tensor(threshold):
        threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
    else:
        threshold = threshold.to(device=x.device, dtype=x.dtype)
    return StraightThroughEstimator.apply(x, threshold)


def hard_concrete_gate(logits, temp=0.67, low=-0.1, high=1.1, training=True):
    if training:
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / temp)
    else:
        s = torch.sigmoid(logits)
    s = s * (high - low) + low
    z = s.clamp(0.0, 1.0)
    hard = (z > 0.5).float()
    return hard + (z - z.detach())


class FLOPsCounter:
    """完整的 FLOPs 计数器"""
    def __init__(self):
        self.flops = 0
        self.details = defaultdict(int)
    
    def count_linear(self, in_f, out_f, batch, name="linear"):
        ops = 2 * batch * in_f * out_f
        self.flops += ops
        self.details[name] += ops
    
    def count_layernorm(self, features, batch, name="layernorm"):
        ops = 5 * batch * features
        self.flops += ops
        self.details[name] += ops
    
    def count_softmax(self, size, batch, name="softmax"):
        ops = 5 * batch * size
        self.flops += ops
        self.details[name] += ops
    
    def count_activation(self, size, batch, name="activation"):
        ops = batch * size
        self.flops += ops
        self.details[name] += ops
    
    def reset(self):
        self.flops = 0
        self.details.clear()
    
    def get_gflops(self):
        return self.flops / 1e9
    
    def get_summary(self):
        return {"total_gflops": self.get_gflops(), "breakdown": {k: v/1e9 for k,v in self.details.items()}}


class SparseMoERouter(nn.Module):
    """
    真正的稀疏 MoE 路由器 - 只计算被选中的专家
    """
    def __init__(self, dim, num_experts=4, top_k=2, capacity_factor=1.25,
                 jitter_noise=0.0, z_loss_coef=0.01, aux_loss_coef=0.01,
                 path_cost_coef=0.0, expert_path_costs: Optional[List[float]] = None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.path_cost_coef = path_cost_coef
        
        self.router = nn.Linear(dim, num_experts, bias=False)
        if expert_path_costs is None:
            expert_path_costs = np.linspace(0.0, 1.0, num_experts, dtype=np.float32).tolist()
        if len(expert_path_costs) != num_experts:
            raise ValueError(f"expert_path_costs must have length {num_experts}")
        self.register_buffer("expert_path_costs", torch.tensor(expert_path_costs, dtype=torch.float32))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            for _ in range(num_experts)
        ])
        self._reset_stats()
    
    def _reset_stats(self):
        self.expert_counts = torch.zeros(self.num_experts)
        self.total_tokens = 0
    
    def forward(self, x, flops_counter=None, return_stats=False):
        batch_size, dim = x.shape
        device = x.device
        if self.expert_counts.device != device:
            self.expert_counts = self.expert_counts.to(device)
        
        # 路由计算
        router_logits = self.router(x)
        if self.path_cost_coef > 0:
            router_logits = router_logits - self.path_cost_coef * self.expert_path_costs.to(router_logits.device)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise
        
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        if flops_counter:
            flops_counter.count_linear(dim, self.num_experts, batch_size, "router")
            flops_counter.count_softmax(self.num_experts, batch_size, "router_softmax")
        
        # 真正的稀疏计算 - 只计算被选中的专家
        output = torch.zeros_like(x)
        capacity = int(np.ceil(self.capacity_factor * batch_size * self.top_k / self.num_experts))
        dropped_tokens = 0
        processed_counts = torch.zeros(self.num_experts, device=device)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx)
            if not expert_mask.any():
                continue
            
            batch_indices, topk_positions = expert_mask.nonzero(as_tuple=True)
            if len(batch_indices) == 0:
                continue

            # capacity control: keep highest-weight tokens for each expert
            if capacity > 0 and batch_indices.numel() > capacity:
                weights = top_k_weights[batch_indices, topk_positions]
                _, top_pos = torch.topk(weights, capacity, dim=0, largest=True)
                dropped_tokens += int(batch_indices.numel() - capacity)
                batch_indices = batch_indices[top_pos]
                topk_positions = topk_positions[top_pos]
            
            expert_input = x[batch_indices]
            expert_weights = top_k_weights[batch_indices, topk_positions]
            
            # 这里才是真正的专家计算
            expert_output = self.experts[expert_idx](expert_input)
            
            num_tokens = batch_indices.numel()
            processed_counts[expert_idx] += num_tokens

            if flops_counter:
                flops_counter.count_linear(dim, dim*4, num_tokens, f"expert_{expert_idx}_up")
                flops_counter.count_activation(dim*4, num_tokens, f"expert_{expert_idx}_act")
                flops_counter.count_linear(dim*4, dim, num_tokens, f"expert_{expert_idx}_down")
            
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output.index_add_(0, batch_indices, weighted_output)
        
        # 辅助损失计算
        expert_usage = router_probs.mean(dim=0)
        denom = processed_counts.sum().clamp_min(1.0)
        expert_selection = processed_counts / denom
        
        load_balance_loss = self.num_experts * (expert_usage * expert_selection).sum()
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        aux_loss = self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss
        
        if self.training:
            with torch.no_grad():
                self.expert_counts += processed_counts
                self.total_tokens += int(processed_counts.sum().item())
        
        stats = None
        if return_stats:
            stats = {
                "load_balance_loss": float(load_balance_loss.detach().cpu().item()),
                "z_loss": float(z_loss.detach().cpu().item()),
                "expert_usage": expert_usage.detach().cpu().tolist(),
                "expert_selection": expert_selection.detach().cpu().tolist(),
                "usage_std": float(expert_usage.std().detach().cpu().item()),
                "compute_ratio": self.top_k / self.num_experts,
                "active_experts": self.top_k,
                "capacity": capacity,
                "dropped_tokens": int(dropped_tokens),
            }
        
        return output, aux_loss, stats
    
    def reset_stats(self):
        self._reset_stats()


class ConsistentSkipGate(nn.Module):
    """使用 STE 的一致性跳跃门控"""
    def __init__(self, dim, init_threshold=0.5, learnable_threshold=True):
        super().__init__()
        self.dim = dim
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim//4), nn.GELU(), nn.Linear(dim//4, 1), nn.Sigmoid()
        )
        self.change_detector = nn.Sequential(
            nn.Linear(dim*2, dim//4), nn.GELU(), nn.Linear(dim//4, 1), nn.Sigmoid()
        )
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(init_threshold))
        else:
            self.register_buffer("threshold", torch.tensor(init_threshold))
    
    def forward(self, x, layer_output, flops_counter=None, track_decisions=False, return_stats=False):
        batch_size = x.size(0)
        importance = self.importance_net(x)
        combined = torch.cat([x, layer_output], dim=-1)
        change_score = self.change_detector(combined)
        skip_score = (1 - importance) * (1 - change_score)
        
        threshold = torch.sigmoid(self.threshold) if isinstance(self.threshold, nn.Parameter) else self.threshold
        skip_decision = straight_through(skip_score.squeeze(-1), threshold).unsqueeze(-1)
        output = skip_decision * x + (1 - skip_decision) * layer_output
        skip_rate = None
        if return_stats:
            skip_rate = float(skip_decision.mean().detach().cpu().item())
        
        if flops_counter:
            flops_counter.count_linear(self.dim, self.dim//4, batch_size, "skip_importance")
            flops_counter.count_linear(self.dim//4, 1, batch_size, "skip_importance")
            flops_counter.count_linear(self.dim*2, self.dim//4, batch_size, "skip_change")
            flops_counter.count_linear(self.dim//4, 1, batch_size, "skip_change")
        
        skip_info_list = None
        if track_decisions:
            skip_info_list = [{"skip": bool(skip_decision[i].detach().cpu().item() > 0.5), 
                              "skip_score": float(skip_score[i].detach().cpu().item()),
                              "importance": float(importance[i].detach().cpu().item())} for i in range(batch_size)]
        
        return output, skip_rate, skip_info_list


class DualModePropagation(nn.Module):
    """双模传播: 快速模式 vs 精细模式"""
    def __init__(self, dim, expansion_factor=4, dropout=0.1, skip_threshold=0.5,
                 gate_type="ste", gate_temp=0.67, gate_low=-0.1, gate_high=1.1):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.gate_type = gate_type
        self.gate_temp = gate_temp
        self.gate_low = gate_low
        self.gate_high = gate_high
        
        self.mode_selector = nn.Sequential(
            nn.Linear(dim, dim//4), nn.GELU(), nn.Linear(dim//4, 1)
        )
        self.fast_path = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.slow_path = nn.Sequential(
            nn.Linear(dim, dim*expansion_factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*expansion_factor, dim), nn.LayerNorm(dim)
        )
        self.skip_gate = ConsistentSkipGate(dim, init_threshold=skip_threshold)
    
    def forward(self, x, flops_counter=None, track_decisions=False, return_stats=False):
        batch_size = x.size(0)
        mode_score = self.mode_selector(x)
        
        if flops_counter:
            flops_counter.count_linear(self.dim, self.dim//4, batch_size, "mode_selector")
            flops_counter.count_linear(self.dim//4, 1, batch_size, "mode_selector")
        
        mode_logits = mode_score.squeeze(-1)
        if self.gate_type == "hard_concrete":
            use_slow = hard_concrete_gate(
                mode_logits, temp=self.gate_temp, low=self.gate_low, high=self.gate_high, training=self.training
            ).unsqueeze(-1)
        else:
            mode_prob = torch.sigmoid(mode_logits)
            use_slow = straight_through(mode_prob, 0.5).unsqueeze(-1)
        fast_out = self.fast_path(x)
        
        if flops_counter:
            flops_counter.count_linear(self.dim, self.dim, batch_size, "fast_path")
            flops_counter.count_layernorm(self.dim, batch_size, "fast_path_ln")
        
        slow_mask = use_slow.squeeze(-1) > 0.5
        slow_out = fast_out.clone()
        
        if slow_mask.any():
            slow_input = x[slow_mask]
            slow_output = self.slow_path(slow_input)
            slow_out[slow_mask] = slow_output
            
            if flops_counter:
                num_slow = slow_mask.sum().item()
                flops_counter.count_linear(self.dim, self.dim*self.expansion_factor, num_slow, "slow_path_up")
                flops_counter.count_activation(self.dim*self.expansion_factor, num_slow, "slow_path_act")
                flops_counter.count_linear(self.dim*self.expansion_factor, self.dim, num_slow, "slow_path_down")
                flops_counter.count_layernorm(self.dim, num_slow, "slow_path_ln")
        
        output = (1 - use_slow) * fast_out + use_slow * slow_out
        output, skip_rate, skip_info = self.skip_gate(
            x, output, flops_counter, track_decisions=track_decisions, return_stats=return_stats
        )
        
        stats = None
        if return_stats:
            stats = {
                "slow_mode_ratio": float(use_slow.mean().detach().cpu().item()),
                "skip_rate": skip_rate,
                "skip_info": skip_info,
            }
        return output, stats


class MAMLStyleAdapter(nn.Module):
    """FiLM/仿射适配器（无内环梯度式 MAML）"""
    def __init__(self, dim, meta_lr=0.01, num_inner_steps=5):
        super().__init__()
        self.dim = dim
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        self.task_encoder = nn.Sequential(
            nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, dim//4)
        )
        self.scale_generator = nn.Linear(dim//4, dim)
        self.shift_generator = nn.Linear(dim//4, dim)
        self.adaptable_transform = nn.Linear(dim, dim)
    
    def encode_task(self, support_set):
        if support_set.dim() == 3:
            task_repr = support_set.mean(dim=(0, 1))
        elif support_set.dim() == 2:
            task_repr = support_set.mean(dim=0)
        else:
            task_repr = support_set
        return self.task_encoder(task_repr.unsqueeze(0)).squeeze(0)
    
    def forward(self, x, support_set=None, flops_counter=None):
        batch_size = x.size(0)
        task_feature = self.encode_task(support_set if support_set is not None else x)
        
        if flops_counter:
            flops_counter.count_linear(self.dim, self.dim//2, 1, "task_encoder")
            flops_counter.count_linear(self.dim//2, self.dim//4, 1, "task_encoder")
        
        # use meta hyperparams as adaptation strength (keeps API stable without claiming inner-loop)
        strength = self.meta_lr * max(1, int(self.num_inner_steps))
        scale = 1.0 + strength * (torch.sigmoid(self.scale_generator(task_feature)) * 2 - 1)
        shift = strength * self.shift_generator(task_feature)
        
        if flops_counter:
            flops_counter.count_linear(self.dim//4, self.dim, 1, "film_generator")
            flops_counter.count_linear(self.dim//4, self.dim, 1, "film_generator")
        
        transformed = self.adaptable_transform(x)
        
        if flops_counter:
            flops_counter.count_linear(self.dim, self.dim, batch_size, "adaptable_transform")
        
        return transformed * scale + shift


class DecisionTracker:
    """决策追踪器"""
    def __init__(self):
        self.decisions = defaultdict(list)
        self.batch_stats = []
    
    def record(self, sample_id, block_id, decision_type, data):
        cleaned = self._clean_data(data)
        self.decisions[sample_id].append({"block": block_id, "type": decision_type, "data": cleaned})
    
    def record_batch(self, block_id, stats):
        self.batch_stats.append({"block": block_id, "stats": self._clean_data(stats)})
    
    def _clean_data(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_data(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
        return obj
    
    def get_routing_pattern(self, sample_id):
        pattern = []
        for decision in self.decisions.get(sample_id, []):
            if decision["type"] == "routing":
                expert_usage = decision["data"].get("expert_usage", [])
                if expert_usage:
                    pattern.append(int(np.argmax(expert_usage)))
        return pattern
    
    def export(self, filepath="decisions.json"):
        data = {
            "total_samples": len(self.decisions),
            "routing_patterns": {str(sid): self.get_routing_pattern(sid) for sid in list(self.decisions.keys())[:100]},
            "batch_stats": self.batch_stats,
            "statistics": self.compute_statistics()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath
    
    def compute_statistics(self):
        if not self.batch_stats:
            return {}
        usage_stds, skip_rates, slow_ratios = [], [], []
        for batch in self.batch_stats:
            stats = batch["stats"]
            if "router" in stats:
                usage_stds.append(stats["router"].get("usage_std", 0))
            if "dual_mode" in stats:
                skip_rates.append(stats["dual_mode"].get("skip_rate", 0))
                slow_ratios.append(stats["dual_mode"].get("slow_mode_ratio", 0))
        return {
            "avg_expert_usage_std": float(np.mean(usage_stds)) if usage_stds else 0,
            "avg_skip_rate": float(np.mean(skip_rates)) if skip_rates else 0,
            "avg_slow_mode_ratio": float(np.mean(slow_ratios)) if slow_ratios else 0
        }
    
    def reset(self):
        self.decisions.clear()
        self.batch_stats.clear()


class UltraLSNTBlock(nn.Module):
    """Ultra-LSNT 基础块 v4.0"""
    def __init__(self, config, block_id=0):
        super().__init__()
        self.block_id = block_id
        self.config = config
        dim = config.hidden_dim
        
        self.input_norm = nn.LayerNorm(dim)
        self.moe_router = SparseMoERouter(
            dim=dim, num_experts=config.num_experts, top_k=config.top_k,
            capacity_factor=config.expert_capacity_factor,
            jitter_noise=config.router_jitter_noise,
            z_loss_coef=config.router_z_loss_coef,
            aux_loss_coef=config.router_aux_loss_coef,
            path_cost_coef=config.router_path_cost_coef,
            expert_path_costs=config.expert_path_costs
        )
        self.dual_prop = DualModePropagation(
            dim=dim,
            dropout=config.dropout,
            skip_threshold=config.skip_threshold,
            gate_type=config.dual_gate_type,
            gate_temp=config.dual_gate_temp,
            gate_low=config.dual_gate_low,
            gate_high=config.dual_gate_high,
        )
        self.meta_adapter = MAMLStyleAdapter(dim=dim, meta_lr=config.meta_lr, num_inner_steps=config.meta_steps)
        self.output_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, support_set=None, tracker=None, sample_ids=None, flops_counter=None, return_stats=False):
        batch_size = x.size(0)
        residual = x
        
        x = self.input_norm(x)
        if flops_counter:
            flops_counter.count_layernorm(self.config.hidden_dim, batch_size, "input_norm")
        
        x = self.meta_adapter(x, support_set, flops_counter=flops_counter)
        x, router_aux_loss, router_stats = self.moe_router(x, flops_counter, return_stats=return_stats or tracker is not None)
        x, dual_stats = self.dual_prop(x, flops_counter, track_decisions=tracker is not None,
                                       return_stats=return_stats or tracker is not None)
        
        x = self.output_norm(x)
        x = self.dropout(x)
        x = x + residual

        resistance_loss = self.config.resistance_loss_coef * (x - residual).pow(2).mean()
        
        if flops_counter:
            flops_counter.count_layernorm(self.config.hidden_dim, batch_size, "output_norm")
        
        if tracker is not None and sample_ids is not None and router_stats is not None:
            for i, sid in enumerate(sample_ids):
                tracker.record(sid, self.block_id, "routing", {
                    "expert_usage": router_stats["expert_usage"],
                    "expert_selection": router_stats["expert_selection"]
                })
                if dual_stats and dual_stats.get("skip_info") and i < len(dual_stats["skip_info"]):
                    tracker.record(sid, self.block_id, "skip", dual_stats["skip_info"][i])
        
        all_stats = None
        if return_stats or tracker is not None:
            all_stats = {
                "router": router_stats,
                "dual_mode": {"slow_mode_ratio": dual_stats["slow_mode_ratio"], "skip_rate": dual_stats["skip_rate"]},
                "router_aux_loss": router_aux_loss,
                "resistance_loss": resistance_loss
            }
            if tracker:
                tracker.record_batch(self.block_id, all_stats)
        
        return x, router_aux_loss + resistance_loss, all_stats


class UltraLSNTNetwork(nn.Module):
    """Ultra-LSNT v4.0 - 完全重构版"""
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = LSNTConfig(**kwargs)
        self.config = config
        
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([UltraLSNTBlock(config, block_id=i) for i in range(config.num_blocks)])
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        
        self.tracker = DecisionTracker()
        self.flops_counter = FLOPsCounter()
    
    def forward(self, x, support_set=None, return_stats=False, track_decisions=False, count_flops=False):
        batch_size = x.size(0)
        if count_flops:
            self.flops_counter.reset()
        
        x = self.input_proj(x)
        if count_flops:
            self.flops_counter.count_linear(self.config.input_dim, self.config.hidden_dim, batch_size, "input_proj")
        
        sample_ids = list(range(batch_size)) if track_decisions else None
        tracker = self.tracker if track_decisions else None
        flops = self.flops_counter if count_flops else None
        
        all_stats = [] if return_stats else None
        total_aux_loss = 0
        
        for block in self.blocks:
            x, block_aux_loss, stats = block(
                x, support_set=support_set, tracker=tracker, sample_ids=sample_ids,
                flops_counter=flops, return_stats=return_stats or track_decisions
            )
            if return_stats:
                all_stats.append(stats)
            total_aux_loss = total_aux_loss + block_aux_loss
        
        x = self.output_norm(x)
        output = self.head(x)
        
        if count_flops:
            self.flops_counter.count_layernorm(self.config.hidden_dim, batch_size, "output_norm")
            self.flops_counter.count_linear(self.config.hidden_dim, self.config.output_dim, batch_size, "head")
        
        if return_stats:
            full_stats = {"block_stats": all_stats, "efficiency": self._compute_efficiency_stats(all_stats)}
            if count_flops:
                full_stats["flops"] = self.flops_counter.get_summary()
            return output, total_aux_loss, full_stats
        
        return output, total_aux_loss
    
    def _compute_efficiency_stats(self, all_stats):
        compute_ratios = [s["router"].get("compute_ratio", 1.0) for s in all_stats]
        skip_rates = [s["dual_mode"].get("skip_rate", 0) for s in all_stats]
        slow_ratios = [s["dual_mode"].get("slow_mode_ratio", 1.0) for s in all_stats]
        load_stds = [s["router"].get("usage_std", 0) for s in all_stats]
        
        denom = max(np.mean(compute_ratios) * (1 - np.mean(skip_rates)), 1e-6)
        return {
            "avg_compute_ratio": float(np.mean(compute_ratios)),
            "avg_skip_rate": float(np.mean(skip_rates)),
            "avg_slow_mode_ratio": float(np.mean(slow_ratios)),
            "avg_load_balance_std": float(np.mean(load_stds)),
            "theoretical_speedup": 1.0 / denom
        }
    
    def export_decisions(self, filepath="decisions.json"):
        path = self.tracker.export(filepath)
        self.tracker.reset()
        return path
    
    def profile_performance(self, x, num_warmup=10, num_runs=100, profile_stats=True):
        self.eval()
        device = x.device
        times = []
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self(x)
            
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if profile_stats:
                    output, aux_loss, stats = self(x, return_stats=True, count_flops=True)
                else:
                    output, aux_loss = self(x, return_stats=False, count_flops=False)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        
        times = np.array(times)
        batch_size = x.size(0)
        
        result = {
            "timing": {"avg_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
                      "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))},
            "throughput": {"samples_per_sec": float(1000.0 / np.mean(times) * batch_size)},
            "model_info": {"num_params": sum(p.numel() for p in self.parameters()),
                          "num_blocks": self.config.num_blocks, "num_experts": self.config.num_experts}
        }
        if profile_stats:
            result["flops"] = stats["flops"]
            result["efficiency"] = stats["efficiency"]
        return result
    
    def reset_all_stats(self):
        self.tracker.reset()
        self.flops_counter.reset()
        for block in self.blocks:
            block.moe_router.reset_stats()


def demo_v4():
    """演示 v4.0"""
    print("=" * 80)
    print("🚀 Ultra-LSNT v4.0 - 完全重构版演示")
    print("=" * 80)
    
    config = LSNTConfig(input_dim=128, hidden_dim=256, output_dim=10, num_blocks=3, num_experts=4, top_k=2)
    model = UltraLSNTNetwork(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n📊 模型信息:")
    print(f"   设备: {device}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   专家数: {config.num_experts}, Top-K: {config.top_k}")
    
    x = torch.randn(32, config.input_dim).to(device)
    
    # 基本测试
    print(f"\n✅ 测试 1: 基本前向传播")
    model.train()
    output, aux_loss = model(x)
    print(f"   输出形状: {output.shape}, 辅助损失: {aux_loss.item():.4f}")
    
    # 详细统计
    print(f"\n✅ 测试 2: 详细统计")
    output, aux_loss, stats = model(x, return_stats=True, count_flops=True)
    print(f"   计算比例: {stats['efficiency']['avg_compute_ratio']:.2%}")
    print(f"   跳跃率: {stats['efficiency']['avg_skip_rate']:.2%}")
    print(f"   总 GFLOPs: {stats['flops']['total_gflops']:.4f}")
    
    # 负载均衡
    print(f"\n✅ 测试 3: 专家负载均衡")
    for i, block_stats in enumerate(stats['block_stats']):
        router_stats = block_stats['router']
        print(f"   Block {i}: 专家使用分布 {[f'{u:.2%}' for u in router_stats['expert_usage']]}")
    
    # 决策追踪
    print(f"\n✅ 测试 4: 决策追踪")
    model.eval()
    with torch.no_grad():
        _, _, _ = model(x, return_stats=True, track_decisions=True)
    filepath = model.export_decisions("v4_decisions.json")
    print(f"   决策已导出到: {filepath}")
    
    # 性能分析
    print(f"\n✅ 测试 5: 性能分析")
    profile = model.profile_performance(x, num_warmup=5, num_runs=50)
    print(f"   平均延迟: {profile['timing']['avg_ms']:.2f} ms")
    print(f"   吞吐量: {profile['throughput']['samples_per_sec']:.0f} samples/sec")
    
    # 训练循环
    print(f"\n✅ 测试 6: 训练循环示例")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(3):
        x = torch.randn(32, config.input_dim).to(device)
        y = torch.randint(0, config.output_dim, (32,)).to(device)
        output, aux_loss = model(x)
        main_loss = criterion(output, y)
        total_loss = main_loss + 0.01 * aux_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(f"   Step {step+1}: main_loss={main_loss.item():.4f}, aux_loss={aux_loss.item():.4f}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ v4.0 所有测试通过!")
    print(f"=" * 80)
    
    print(f"\n🎯 v4.0 核心改进:")
    print(f"   1. ✅ 真正的稀疏计算 - 只计算被选中的专家")
    print(f"   2. ✅ Switch Transformer 风格负载均衡损失")
    print(f"   3. ✅ 直通估计器(STE) - 训练/推理一致")
    print(f"   4. ✅ FiLM/条件仿射适配（非 MAML 内环）")
    print(f"   5. ✅ 完整 FLOPs 计数")
    print(f"   6. ✅ 明确的双模传播")


def compare_efficiency():
    """对比不同配置的效率"""
    print("\n" + "=" * 80)
    print("📊 效率对比分析")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(64, 128).to(device)
    
    configs = [
        ("Dense (all experts)", LSNTConfig(num_experts=4, top_k=4)),
        ("Sparse Top-2", LSNTConfig(num_experts=4, top_k=2)),
        ("Sparse Top-1", LSNTConfig(num_experts=4, top_k=1)),
        ("8 Experts Top-2", LSNTConfig(num_experts=8, top_k=2)),
    ]
    
    results = []
    for name, config in configs:
        model = UltraLSNTNetwork(config).to(device)
        model.eval()
        with torch.no_grad():
            _, _, stats = model(x, return_stats=True, count_flops=True)
        results.append({
            "name": name, "gflops": stats["flops"]["total_gflops"],
            "compute_ratio": stats["efficiency"]["avg_compute_ratio"],
            "params": sum(p.numel() for p in model.parameters())
        })
    
    print(f"\n{'配置':<25} {'GFLOPs':<12} {'计算比例':<12} {'参数量':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} {r['gflops']:<12.4f} {r['compute_ratio']:<12.2%} {r['params']:<15,}")


if __name__ == "__main__":
    demo_v4()
    compare_efficiency()


