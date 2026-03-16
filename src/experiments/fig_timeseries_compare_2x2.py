import numpy as np
import matplotlib.pyplot as plt
from ae_plot_style import set_ae_style, save_ae

def panel_label(ax, s: str) -> None:
    ax.text(0.02, 0.95, s, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, fontweight="bold")

def load_real_predictions():
    """加载真实实验数据：真实值和多个模型的预测值"""
    # 真实值（以TimeMixer的真实值为基准，所有模型共享相同的真实值）
    y_true = np.load('checkpoints_ts/TimeMixer_trues.npy')  # shape (18111, 24)
    # 选择第一个预测步长（h=1）作为1D序列
    y_true_1d = y_true[:, 0].flatten()
    
    # 加载各模型预测值
    preds_dict = {}
    
    # TimeMixer
    preds_dict["TimeMixer"] = np.load('checkpoints_ts/TimeMixer_preds.npy')[:, 0].flatten()
    
    # iTransformer
    preds_dict["iTransformer"] = np.load('checkpoints_ts/iTransformer_preds.npy')[:, 0].flatten()
    
    # PatchTST
    preds_dict["PatchTST"] = np.load('checkpoints_ts/PatchTST_preds.npy')[:, 0].flatten()
    
    # LightGBM (GBDT)
    try:
        preds_gbdt = np.load('preds_lgbm.npy')[:, 0].flatten()
        # 确保长度匹配
        if len(preds_gbdt) == len(y_true_1d):
            preds_dict["LightGBM"] = preds_gbdt
        else:
            print(f"LightGBM 长度不匹配: {len(preds_gbdt)} vs {len(y_true_1d)}")
    except Exception as e:
        print(f"加载 LightGBM 预测失败: {e}")
    
    # 检查长度一致性
    for name, pred in preds_dict.items():
        if len(pred) != len(y_true_1d):
            print(f"警告: {name} 长度不匹配: {len(pred)} vs {len(y_true_1d)}")
            # 截断到最短长度
            min_len = min(len(pred), len(y_true_1d))
            preds_dict[name] = pred[:min_len]
    
    min_len = min(len(y_true_1d), *(len(p) for p in preds_dict.values()))
    y_true_1d = y_true_1d[:min_len]
    for name in preds_dict:
        preds_dict[name] = preds_dict[name][:min_len]
    
    return y_true_1d, preds_dict

def plot_timeseries_panels(t, y_true, preds_dict, windows, titles) -> plt.Figure:
    """
    preds_dict: {"TimeMixer": y_pred, "iTransformer": y_pred2, ...}
    windows: list of (start_idx, end_idx) for 4 panels
    """
    set_ae_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), constrained_layout=True)

    for i, ax in enumerate(axes.ravel()):
        s, e = windows[i]
        ax.plot(t[s:e], y_true[s:e], label="True", alpha=0.9, linewidth=1.0)
        
        # 为每个模型绘制预测
        for name, yp in preds_dict.items():
            ax.plot(t[s:e], yp[s:e], label=name, alpha=0.8, linewidth=0.8)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Wind power (MW)")
        ax.set_title(titles[i])
        panel_label(ax, f"({chr(ord('a') + i)})")

        # 图例放在图内角落，避免占外部空间
        if i == 0:  # 只在第一个子图显示图例
            ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=6)
        else:
            # 其他子图不显示图例
            pass

    return fig

if __name__ == "__main__":
    print("加载真实实验数据...")
    y_true, preds_dict = load_real_predictions()
    
    n = len(y_true)
    print(f"数据长度: {n}")
    print(f"可用模型: {list(preds_dict.keys())}")
    
    # 创建时间索引（天）
    t = np.arange(n)
    
    # 定义四个窗口（每个窗口显示约100个点）
    window_size = 100
    windows = [
        (0, window_size),
        (window_size, 2*window_size),
        (2*window_size, 3*window_size),
        (3*window_size, 4*window_size)
    ]
    
    # 如果数据长度不够，调整窗口
    if 4*window_size > n:
        window_size = n // 4
        windows = [
            (0, window_size),
            (window_size, 2*window_size),
            (2*window_size, 3*window_size),
            (3*window_size, n)
        ]
    
    titles = ["Segment 1", "Segment 2", "Segment 3", "Segment 4"]
    
    print("生成时间序列对比图...")
    fig = plot_timeseries_panels(t, y_true, preds_dict, windows, titles)
    
    # 保存图形
    save_ae(fig, out_prefix="timeseries_compare_2x2_real", out_dir="figures_out")
    print("已保存: figures_out/timeseries_compare_2x2_real.pdf and .tif")
    
    # 显示图形
    plt.show()