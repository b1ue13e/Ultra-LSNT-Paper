"""
绘制Ultra-LSNT详细架构图（基于用户提供的TikZ代码）
"""
from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# 确保输出目录
OUT_DIR = Path("figure")
OUT_DIR.mkdir(exist_ok=True)

def set_paper_style():
    """设置论文风格"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def draw_box(ax, x, y, width, height, text, facecolor='white', edgecolor='black', 
             linewidth=1.2, alpha=1.0, fontsize=10, fontweight='normal', 
             text_color='black', rounded=True):
    """绘制带文本的方框"""
    if rounded:
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.02,rounding_size=0.02",
                            linewidth=linewidth,
                            edgecolor=edgecolor,
                            facecolor=facecolor,
                            alpha=alpha)
    else:
        box = patches.Rectangle((x, y), width, height,
                               linewidth=linewidth,
                               edgecolor=edgecolor,
                               facecolor=facecolor,
                               alpha=alpha)
    ax.add_patch(box)
    
    # 添加文本（支持换行）
    lines = text.split('\n')
    for i, line in enumerate(lines):
        offset = (len(lines)-1)/2 - i
        ax.text(x + width/2, y + height/2 + offset*0.18*height,
                line, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color)
    return box

def draw_diamond(ax, x, y, size, text, facecolor='white', edgecolor='black', 
                 linewidth=1.2, fontsize=9):
    """绘制菱形"""
    diamond = patches.RegularPolygon((x, y), numVertices=4, radius=size,
                                     orientation=np.pi/4,
                                     facecolor=facecolor,
                                     edgecolor=edgecolor,
                                     linewidth=linewidth)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)
    return diamond

def draw_circle(ax, x, y, radius, text, facecolor='white', edgecolor='black',
                linewidth=1.2, fontsize=10):
    """绘制圆形"""
    circle = patches.Circle((x, y), radius, facecolor=facecolor,
                           edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)
    return circle

def draw_arrow(ax, start, end, style='->', color='gray', linewidth=1.5, 
               alpha=1.0, head_length=0.15, head_width=0.1, dashed=False):
    """绘制箭头"""
    if dashed:
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    
    arrow = FancyArrowPatch(start, end,
                           arrowstyle=f'{style},head_length={head_length},head_width={head_width}',
                           color=color,
                           linewidth=linewidth,
                           alpha=alpha,
                           linestyle=linestyle)
    ax.add_patch(arrow)
    return arrow

def plot_ultra_lsnt_architecture():
    """绘制Ultra-LSNT架构图"""
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色定义
    blue_light = '#E6F3FF'
    blue_dark = '#0055AA'
    orange_light = '#FFEECC'
    orange_dark = '#CC6600'
    gray_light = '#F5F5F5'
    gray_dark = '#666666'
    green_light = '#E8F5E8'
    green_dark = '#2E7D32'
    red_light = '#FFE6E6'
    red_dark = '#C62828'
    
    # --- 1. 输入和预处理 ---
    input_box = draw_box(ax, 0.5, 4.5, 1.8, 1.0, 
                        text='Input\n$\\mathbf{X}$\n$(L \\times C)$',
                        facecolor=gray_light, edgecolor=gray_dark, fontsize=9)
    
    ma_box = draw_box(ax, 2.8, 4.5, 1.8, 1.0,
                     text='Moving\nAverage\nDecomp. (MA)',
                     facecolor=gray_light, edgecolor=gray_dark, fontsize=9)
    
    draw_arrow(ax, (input_box.get_x() + input_box.get_width(), input_box.get_y() + input_box.get_height()/2),
               (ma_box.get_x(), ma_box.get_y() + ma_box.get_height()/2),
               color=gray_dark)
    
    # 分支分叉点
    split_x = ma_box.get_x() + ma_box.get_width() + 0.3
    split_y = ma_box.get_y() + ma_box.get_height()/2
    
    # --- 2. Trend Branch (顶部) ---
    trend_label = ax.text(split_x - 0.5, 7.8, 'Trend Branch\n(lightweight / low compute)',
                         fontsize=9, fontweight='bold', color=gray_dark, ha='center')
    
    trend_box = draw_box(ax, split_x + 0.5, 6.5, 2.5, 1.2,
                        text='$\\mathbf{X}_{\\text{trend}}$\n$\\rightarrow$ Tiny MLP / Linear Head\n$\\rightarrow \\hat{\\mathbf{y}}_{\\text{trend}}$',
                        facecolor=blue_light, edgecolor=blue_dark, fontsize=8)
    
    # Trend箭头
    draw_arrow(ax, (split_x, split_y), (split_x, trend_box.get_y() + trend_box.get_height()/2),
               color=blue_dark)
    draw_arrow(ax, (split_x, trend_box.get_y() + trend_box.get_height()/2),
               (trend_box.get_x(), trend_box.get_y() + trend_box.get_height()/2),
               color=blue_dark)
    
    # --- 3. Seasonal Branch (底部) ---
    seasonal_label = ax.text(split_x - 0.5, 3.2, 'Seasonal Branch\n(hard / nonlinear)',
                            fontsize=9, fontweight='bold', color=gray_dark, ha='center')
    
    # 多尺度嵌入容器
    embed_box = draw_box(ax, split_x + 0.5, 1.0, 2.5, 2.8,
                        text='',
                        facecolor=orange_light, edgecolor=orange_dark, linewidth=2)
    
    ax.text(embed_box.get_x() + embed_box.get_width()/2, embed_box.get_y() + embed_box.get_height() + 0.2,
            'Multi-Scale\nTemporal Embedding', fontsize=9, ha='center')
    
    # 内部卷积层
    conv1 = draw_box(ax, embed_box.get_x() + 0.15, embed_box.get_y() + 1.8, 2.2, 0.7,
                    text='Conv1D ($k_s=3$)\nShort-term',
                    facecolor=gray_light, edgecolor=gray_dark, fontsize=8)
    
    conv2 = draw_box(ax, embed_box.get_x() + 0.15, embed_box.get_y() + 0.5, 2.2, 0.7,
                    text='Conv1D ($k_m=7$)\nMid-term',
                    facecolor=gray_light, edgecolor=gray_dark, fontsize=8)
    
    # Seasonal箭头
    draw_arrow(ax, (split_x, split_y), (split_x, embed_box.get_y() + embed_box.get_height()/2),
               color=orange_dark)
    draw_arrow(ax, (split_x, embed_box.get_y() + embed_box.get_height()/2),
               (embed_box.get_x(), embed_box.get_y() + embed_box.get_height()/2),
               color=orange_dark)
    
    # 融合/投影
    fusion_x = embed_box.get_x() + embed_box.get_width() + 0.8
    fusion_y = embed_box.get_y() + embed_box.get_height()/2
    
    fusion_circle = draw_circle(ax, fusion_x, fusion_y, 0.25, '+',
                               facecolor='white', edgecolor=gray_dark, fontsize=12)
    
    ax.text(fusion_x, fusion_y - 0.4, 'Fusion/Projection\n$\\rightarrow \\mathbf{E}$',
            fontsize=8, ha='center')
    
    # 从卷积到融合的箭头
    draw_arrow(ax, (conv1.get_x() + conv1.get_width(), conv1.get_y() + conv1.get_height()/2),
               (fusion_x, fusion_y),
               color=gray_dark)
    draw_arrow(ax, (conv2.get_x() + conv2.get_width(), conv2.get_y() + conv2.get_height()/2),
               (fusion_x, fusion_y),
               color=gray_dark)
    
    # --- 4. 稀疏MoE编码器 ---
    # 路由器
    router_x = fusion_x + 1.2
    router = draw_diamond(ax, router_x, fusion_y, 0.4,
                         text='Router\n(Softmax)\n$\\rightarrow$ Top-K',
                         facecolor=gray_light, edgecolor=gray_dark, fontsize=8)
    
    draw_arrow(ax, (fusion_x + 0.25, fusion_y),
               (router_x - 0.4, fusion_y),
               color=gray_dark)
    
    # MoE容器框
    moe_start_x = router_x - 0.8
    moe_end_x = router_x + 3.5
    moe_start_y = fusion_y - 2.5
    moe_end_y = fusion_y + 2.5
    
    moe_box = patches.Rectangle((moe_start_x, moe_start_y),
                               moe_end_x - moe_start_x,
                               moe_end_y - moe_start_y,
                               linewidth=1.5, edgecolor='#666666',
                               facecolor='none', linestyle='--', alpha=0.7)
    ax.add_patch(moe_box)
    
    ax.text(moe_end_x - 0.1, moe_end_y - 0.1, 'Sparse MoE Encoder',
            fontsize=10, fontweight='bold', ha='right', va='top')
    
    # 专家
    experts_x = router_x + 1.5
    expert_spacing = 0.9
    
    e1 = draw_box(ax, experts_x, fusion_y + expert_spacing*1.5, 1.2, 0.6,
                 text='E1', facecolor=red_light, edgecolor=red_dark, fontsize=9, fontweight='bold')
    
    e2 = draw_box(ax, experts_x, fusion_y + expert_spacing*0.5, 1.2, 0.6,
                 text='E2', facecolor=gray_light, edgecolor=gray_dark, fontsize=9, alpha=0.4)
    
    e3 = draw_box(ax, experts_x, fusion_y - expert_spacing*0.5, 1.2, 0.6,
                 text='E3', facecolor=red_light, edgecolor=red_dark, fontsize=9, fontweight='bold')
    
    e4 = draw_box(ax, experts_x, fusion_y - expert_spacing*1.5, 1.2, 0.6,
                 text='E4', facecolor=gray_light, edgecolor=gray_dark, fontsize=9, alpha=0.4)
    
    # 更多专家指示
    ax.text(experts_x, fusion_y - expert_spacing*2.2, '...', fontsize=12, ha='center', fontweight='bold')
    en = draw_box(ax, experts_x, fusion_y - expert_spacing*3.0, 1.2, 0.6,
                 text='EN', facecolor=red_light, edgecolor=red_dark, fontsize=9, fontweight='bold')
    
    # 路由器到专家的箭头
    for expert, active in [(e1, True), (e2, False), (e3, True), (e4, False), (en, True)]:
        if active:
            draw_arrow(ax, (router_x + 0.4, fusion_y),
                      (expert.get_x(), expert.get_y() + expert.get_height()/2),
                      color=red_dark)
        else:
            draw_arrow(ax, (router_x + 0.4, fusion_y),
                      (expert.get_x(), expert.get_y() + expert.get_height()/2),
                      color=gray_dark, alpha=0.4, dashed=True)
    
    # 加权和
    sum_x = experts_x + 1.8
    sum_circle = draw_circle(ax, sum_x, fusion_y, 0.3, '$\sum$',
                            facecolor='white', edgecolor=gray_dark, fontsize=14)
    
    ax.text(sum_x, fusion_y - 0.4, 'Weighted Sum\n$\\rightarrow \\mathbf{h}_t$',
            fontsize=8, ha='center')
    
    # 专家到加权和的箭头
    for expert, active in [(e1, True), (e2, False), (e3, True), (e4, False), (en, True)]:
        if active:
            draw_arrow(ax, (expert.get_x() + expert.get_width(), expert.get_y() + expert.get_height()/2),
                      (sum_circle.get_x(), sum_circle.get_y()),
                      color=red_dark)
        else:
            draw_arrow(ax, (expert.get_x() + expert.get_width(), expert.get_y() + expert.get_height()/2),
                      (sum_circle.get_x(), sum_circle.get_y()),
                      color=gray_dark, alpha=0.4, dashed=True)
    
    # --- 5. 跳跃门（条件跳过）---
    # 跳跃连接线
    skip_start_x = (fusion_x + router_x) / 2
    skip_start_y = fusion_y
    skip_mid_y = fusion_y - 3.5
    skip_end_x = sum_x + 0.5
    skip_end_y = fusion_y
    
    # 绘制折线
    ax.plot([skip_start_x, skip_start_x], [skip_start_y, skip_mid_y],
            color='gray', linewidth=1.5, linestyle='-')
    ax.plot([skip_start_x, skip_end_x], [skip_mid_y, skip_mid_y],
            color='gray', linewidth=1.5, linestyle='-')
    
    # 开关符号
    switch_x = skip_end_x - 0.3
    switch_y = skip_mid_y
    ax.plot([switch_x - 0.2, switch_x + 0.2], [switch_y, switch_y + 0.1],
            color='black', linewidth=1.5)
    ax.plot([switch_x, switch_x], [switch_y - 0.05, switch_y + 0.05],
            color='black', linewidth=1.5)
    
    ax.text(skip_end_x - 0.3, skip_mid_y - 0.3, 'Jump Gate\n(Conditional Skip)',
            fontsize=8, ha='center')
    
    # 最终箭头到输出
    draw_arrow(ax, (skip_end_x, skip_mid_y), (skip_end_x, skip_end_y),
               color='gray')
    
    # --- 6. 输出阶段 ---
    # 预测头
    pred_x = sum_x + 1.2
    pred_box = draw_box(ax, pred_x, fusion_y - 0.3, 1.5, 0.8,
                       text='Prediction\nHead\n$\\rightarrow \\hat{\\mathbf{y}}_{\\text{season}}$',
                       facecolor=blue_light, edgecolor=blue_dark, fontsize=8)
    
    draw_arrow(ax, (sum_x + 0.3, fusion_y),
               (pred_box.get_x(), pred_box.get_y() + pred_box.get_height()/2),
               color=blue_dark)
    
    # 最终求和
    final_sum_x = pred_x + 1.8
    final_sum_circle = draw_circle(ax, final_sum_x, fusion_y, 0.25, '+',
                           facecolor='white', edgecolor=gray_dark, fontsize=12)
    
    # 季节性预测到最终求和
    draw_arrow(ax, (pred_box.get_x() + pred_box.get_width(), pred_box.get_y() + pred_box.get_height()/2),
               (final_sum_x, fusion_y),
               color=blue_dark)
    
    # 趋势预测到最终求和
    trend_to_final_x = trend_box.get_x() + trend_box.get_width() + 0.5
    ax.plot([trend_box.get_x() + trend_box.get_width(), trend_to_final_x],
            [trend_box.get_y() + trend_box.get_height()/2, trend_box.get_y() + trend_box.get_height()/2],
            color=blue_dark, linewidth=1.5)
    draw_arrow(ax, (trend_to_final_x, trend_box.get_y() + trend_box.get_height()/2),
               (final_sum.get_x(), final_sum.get_y() + 0.2),
               color=blue_dark)
    
    # 最终输出
    final_x = final_sum_x + 0.8
    final_text = ax.text(final_x, fusion_y, 'Final\nForecast\n$\\hat{\\mathbf{y}}$',
                         fontsize=10, fontweight='bold', ha='left', va='center')
    
    draw_arrow(ax, (final_sum.get_x() + final_sum.radius, final_sum.get_y()),
               (final_text.get_x(), final_text.get_y()),
               color='black')
    
    # --- 7. 注释 ---
    # Top-K激活注释
    ax.text(router_x, fusion_y - 1.8, 'Top-K Experts\nActivated\n(Sparse Routing)',
            fontsize=8, fontstyle='italic', color=gray_dark, ha='center')
    
    # Edge-Friendly标签
    edge_box = draw_box(ax, 13.5, 0.5, 2.2, 1.2,
                       text='$\\textbf{Edge-Friendly}$\n$\\bullet$ Reduced Average Compute\n$\\bullet$ Lower Energy-per-Inference',
                       facecolor=green_light, edgecolor=green_dark, fontsize=8, rounded=True)
    
    fig.tight_layout()
    
    # 保存图像
    output_path = OUT_DIR / "fig_architecture_detailed"
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"架构图已保存到: {output_path}.png 和 {output_path}.pdf")

if __name__ == "__main__":
    plot_ultra_lsnt_architecture()