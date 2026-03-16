"""
风电场景调度模型

给定风电预测均值 μ_t 和不确定性 σ_t（假设正态分布），
决定：
- 火电/燃机出力 g_t
- 备用容量 r_t
- 储能充放电 e_t^ch, e_t^dis, s_t（电量）

目标：最小化运行成本
   燃料成本 + 启停成本 + 备用成本 + 购电成本 + 储能退化成本

约束：
1. 功率平衡： g_t + w_t + e_t^dis - e_t^ch = d_t
2. 风电不确定性： w_t 服从 N(μ_t, σ_t^2)，采用机会约束或鲁棒优化
3. 备用约束： r_t >= β * σ_t  (β为安全系数)
4. 机组出力上下限： g_min <= g_t <= g_max
5. 爬坡约束： |g_t - g_{t-1}| <= Δg_max
6. 储能约束： s_{t+1} = s_t + η_ch * e_t^ch - e_t^dis/η_dis
7. 储能容量： 0 <= s_t <= S_max
8. 充放电功率限制： 0 <= e_t^ch <= P_ch_max, 0 <= e_t^dis <= P_dis_max
9. 充放电互补： e_t^ch * e_t^dis = 0

简化：采用确定性等价（将不确定性转化为备用要求）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
import sys
sys.path.append('.')

# 参数
def get_default_params():
    params = {
        'T': 24,               # 调度时段数
        'dt': 1,               # 小时
        'demand': 50000,       # 固定负荷 (kW)
        'wind_mean_scale': 0.7, # 风电预测均值缩放因子（相对于预测值）
        'wind_std_ratio': 0.2,  # 风电不确定性标准差比例（均值的20%）
        'beta': 1.5,           # 安全系数（备用需求）
        # 火电参数
        'g_min': 10000,        # 最小出力 (kW)
        'g_max': 80000,        # 最大出力 (kW)
        'ramp_max': 20000,     # 最大爬坡 (kW/h)
        'fuel_cost': 0.5,      # 燃料成本 元/kWh
        'startup_cost': 5000,  # 启停成本 元/次
        'reserve_cost': 0.2,   # 备用容量成本 元/kWh
        # 储能参数
        'S_max': 50000,        # 储能容量 (kWh)
        'P_ch_max': 10000,     # 最大充电功率 (kW)
        'P_dis_max': 10000,    # 最大放电功率 (kW)
        'eta_ch': 0.95,        # 充电效率
        'eta_dis': 0.95,       # 放电效率
        'storage_cost': 0.05,  # 储能退化成本 元/kWh
        'initial_soc': 0.5,    # 初始SOC比例
    }
    return params

def load_wind_forecast(mode='point'):
    """
    加载风电预测数据（支持概率预测）
    
    参数:
        mode: 'point' (点预测), 'gaussian' (高斯分布), 'quantile' (分位数回归)
    
    返回:
        wind_mean: 预测均值 (T,)
        wind_std: 预测标准差 (T,)
        wind_q95: 95%分位数预测 (T,) (仅概率模式有效)
    """
    try:
        # 尝试加载概率预测文件
        prob_data_path = 'checkpoints_ts/prob_predictions.npz'
        data = np.load(prob_data_path)
        
        if mode == 'gaussian' and 'mu' in data and 'sigma' in data:
            mu = data['mu']          # 形状可能为 (batch, pred_len) 或 (pred_len,)
            sigma = data['sigma']
            # 取第一个样本
            if mu.ndim == 2:
                mu = mu[0, :]
                sigma = sigma[0, :]
            wind_mean = mu
            wind_std = sigma
            wind_q95 = mu + 1.96 * sigma
        elif mode == 'quantile' and 'quantiles' in data and 'quantile_values' in data:
            quantiles = data['quantiles']          # 分位数列表，例如 [0.1, 0.5, 0.9, 0.95]
            quantile_values = data['quantile_values']  # 形状 (batch, pred_len, num_quantiles)
            # 取第一个样本
            if quantile_values.ndim == 3:
                qvals = quantile_values[0, :, :]   # (pred_len, num_quantiles)
            else:
                qvals = quantile_values
            # 寻找0.5分位数作为均值估计
            median_idx = np.where(np.abs(quantiles - 0.5) < 1e-6)[0]
            if len(median_idx) > 0:
                wind_mean = qvals[:, median_idx[0]].flatten()
            else:
                wind_mean = qvals[:, len(quantiles)//2].flatten()
            # 寻找0.95分位数
            q95_idx = np.where(np.abs(quantiles - 0.95) < 1e-6)[0]
            if len(q95_idx) > 0:
                wind_q95 = qvals[:, q95_idx[0]].flatten()
            else:
                # 如果没有0.95分位数，使用最大分位数近似
                wind_q95 = qvals[:, -1].flatten()
            # 标准差近似：假设正态分布，使用 (q95 - mean)/1.96
            wind_std = (wind_q95 - wind_mean) / 1.96
            wind_std = np.maximum(wind_std, wind_mean * 0.05)  # 避免为零
        else:
            # 回退到点预测文件
            data = np.load('checkpoints_ts/main/predictions.npz')
            preds = data['predictions']
            wind_mean = preds[0, :]  # (pred_len,)
            wind_std = wind_mean * 0.2
            wind_q95 = wind_mean + 1.96 * wind_std
            print(f"使用点预测数据，不确定性为均值的20%")
        
        T = len(wind_mean)
        print(f"加载 {mode} 预测，时段数={T}")
        print(f"均值范围: [{wind_mean.min():.0f}, {wind_mean.max():.0f}] kW")
        print(f"标准差范围: [{wind_std.min():.0f}, {wind_std.max():.0f}] kW")
        print(f"95%分位数范围: [{wind_q95.min():.0f}, {wind_q95.max():.0f}] kW")
        
        return wind_mean, wind_std, wind_q95
        
    except Exception as e:
        print(f"无法加载预测数据: {e}")
        # 生成模拟数据
        T = 24
        t = np.arange(T)
        wind_mean = 30000 + 10000 * np.sin(2 * np.pi * t / 24)
        wind_std = wind_mean * 0.2
        wind_q95 = wind_mean + 1.96 * wind_std
        print("使用模拟数据")
        return wind_mean, wind_std, wind_q95

def build_dispatch_model(params, wind_mean, wind_std):
    """
    构建调度优化问题
    变量顺序: [g_1,...,g_T, r_1,...,r_T, e_ch_1,...,e_ch_T, e_dis_1,...,e_dis_T]
    共 4*T 个变量
    """
    T = params['T']
    n_vars = 4 * T
    # 目标函数系数
    c = np.zeros(n_vars)
    # 燃料成本
    c[:T] = params['fuel_cost']
    # 备用成本
    c[T:2*T] = params['reserve_cost']
    # 充电成本（视为购电成本）
    c[2*T:3*T] = params['fuel_cost']  # 假设购电价格与燃料成本相同
    # 放电无直接成本，但有储能退化成本
    c[3*T:4*T] = params['storage_cost']
    
    # 添加启停成本近似（二次惩罚）
    def objective(x):
        g = x[:T]
        r = x[T:2*T]
        e_ch = x[2*T:3*T]
        e_dis = x[3*T:4*T]
        # 线性部分
        cost = np.dot(c, x)
        # 启停成本近似（绝对值变化）
        g_diff = np.diff(g, prepend=g[0])
        startup_cost = params['startup_cost'] * np.sum(np.maximum(g_diff, 0) / params['ramp_max'])
        shutdown_cost = params['startup_cost'] * np.sum(np.maximum(-g_diff, 0) / params['ramp_max'])
        cost += startup_cost + shutdown_cost
        return cost
    
    # 约束
    constraints = []
    
    # 1. 功率平衡约束: g_t + wind_mean_t + e_dis_t - e_ch_t = demand
    A_eq = np.zeros((T, n_vars))
    for t in range(T):
        A_eq[t, t] = 1                     # g_t
        A_eq[t, 3*T + t] = 1               # e_dis_t
        A_eq[t, 2*T + t] = -1              # -e_ch_t
    b_eq = params['demand'] - wind_mean    # demand - wind_mean_t
    constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    
    # 2. 备用约束: r_t >= beta * wind_std_t
    # 转换为 r_t - beta*wind_std_t >= 0
    A_ub = np.zeros((T, n_vars))
    for t in range(T):
        A_ub[t, T + t] = -1  # -r_t <= -beta*wind_std_t
    b_ub = -params['beta'] * wind_std
    constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
    
    # 3. 机组出力上下限
    bounds = Bounds(
        lb=np.concatenate([
            np.full(T, params['g_min']),   # g_t
            np.zeros(T),                   # r_t
            np.zeros(T),                   # e_ch_t
            np.zeros(T)                    # e_dis_t
        ]),
        ub=np.concatenate([
            np.full(T, params['g_max']),   # g_t
            np.full(T, params['g_max']),   # r_t（上限同机组）
            np.full(T, params['P_ch_max']),# e_ch_t
            np.full(T, params['P_dis_max'])# e_dis_t
        ])
    )
    
    # 4. 爬坡约束：|g_t - g_{t-1}| <= ramp_max
    # 线性化：g_t - g_{t-1} <= ramp_max, g_{t-1} - g_t <= ramp_max
    A_ramp = np.zeros((2*(T-1), n_vars))
    b_ramp = np.zeros(2*(T-1))
    for i in range(T-1):
        # 正向爬坡
        A_ramp[2*i, i] = 1
        A_ramp[2*i, i+1] = -1
        b_ramp[2*i] = params['ramp_max']
        # 反向爬坡
        A_ramp[2*i+1, i] = -1
        A_ramp[2*i+1, i+1] = 1
        b_ramp[2*i+1] = params['ramp_max']
    constraints.append(LinearConstraint(A_ramp, -np.inf, b_ramp))
    
    # 5. 储能动态约束（线性）
    # s_{t+1} = s_t + eta_ch * e_ch_t - e_dis_t/eta_dis
    # 约束：0 <= s_t <= S_max
    # 我们引入s_t作为变量（增加T个变量），但为简化，采用近似：充放电能量平衡
    # 替代：约束总充电量不超过容量
    # 我们增加约束：sum(eta_ch * e_ch_t - e_dis_t/eta_dis) = 0（周期内平衡）
    A_stor = np.zeros((1, n_vars))
    A_stor[0, 2*T:3*T] = params['eta_ch']      # 充电贡献
    A_stor[0, 3*T:4*T] = -1/params['eta_dis']  # 放电贡献
    b_stor = np.array([0.0])
    constraints.append(LinearConstraint(A_stor, b_stor, b_stor))
    
    # 6. 充放电互补约束（非线性），松弛处理：添加惩罚项
    def add_complementarity_penalty(x):
        e_ch = x[2*T:3*T]
        e_dis = x[3*T:4*T]
        penalty = 1000.0 * np.sum(e_ch * e_dis)  # 鼓励乘积为0
        return penalty
    
    # 修改目标函数包含互补惩罚
    def full_objective(x):
        return objective(x) + add_complementarity_penalty(x)
    
    # 初始猜测
    x0 = np.zeros(n_vars)
    x0[:T] = (params['g_min'] + params['g_max']) / 2
    x0[T:2*T] = params['beta'] * wind_std
    x0[2*T:3*T] = 0.0
    x0[3*T:4*T] = 0.0
    
    return full_objective, constraints, bounds, x0, n_vars

def solve_dispatch(params, wind_mean, wind_std):
    """求解调度问题"""
    objective, constraints, bounds, x0, n_vars = build_dispatch_model(params, wind_mean, wind_std)
    
    print("优化问题规模：变量数 =", n_vars)
    print("约束数：", len(constraints))
    
    # 使用SLSQP求解
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )
    
    if result.success:
        print("优化成功！")
        print("目标函数值:", result.fun)
        return result.x
    else:
        print("优化失败:", result.message)
        return None

def parse_solution(x, params):
    """解析优化结果"""
    T = params['T']
    g = x[:T]
    r = x[T:2*T]
    e_ch = x[2*T:3*T]
    e_dis = x[3*T:4*T]
    
    # 计算SOC（近似）
    soc = np.zeros(T)
    soc[0] = params['initial_soc'] * params['S_max']
    for t in range(1, T):
        soc[t] = soc[t-1] + params['eta_ch'] * e_ch[t-1] - e_dis[t-1] / params['eta_dis']
        soc[t] = max(0, min(soc[t], params['S_max']))
    
    return {
        'thermal': g,
        'reserve': r,
        'charge': e_ch,
        'discharge': e_dis,
        'soc': soc,
        'total_cost': None  # 将在外部计算
    }

def evaluate_cost(solution, params, wind_mean, wind_std):
    """评估调度方案的总成本"""
    g = solution['thermal']
    r = solution['reserve']
    e_ch = solution['charge']
    e_dis = solution['discharge']
    
    T = params['T']
    # 燃料成本
    fuel_cost = params['fuel_cost'] * np.sum(g)
    # 备用成本
    reserve_cost = params['reserve_cost'] * np.sum(r)
    # 购电成本（充电）
    purchase_cost = params['fuel_cost'] * np.sum(e_ch)
    # 储能退化成本
    storage_cost = params['storage_cost'] * np.sum(e_dis)
    # 启停成本近似
    g_diff = np.diff(g, prepend=g[0])
    startup_cost = params['startup_cost'] * np.sum(np.maximum(g_diff, 0) / params['ramp_max'])
    shutdown_cost = params['startup_cost'] * np.sum(np.maximum(-g_diff, 0) / params['ramp_max'])
    
    total = fuel_cost + reserve_cost + purchase_cost + storage_cost + startup_cost + shutdown_cost
    
    return {
        'fuel_cost': fuel_cost,
        'reserve_cost': reserve_cost,
        'purchase_cost': purchase_cost,
        'storage_cost': storage_cost,
        'startup_cost': startup_cost,
        'shutdown_cost': shutdown_cost,
        'total_cost': total
    }

def plot_dispatch(solution, params, wind_mean, wind_std, cost_breakdown):
    """绘制调度结果"""
    T = params['T']
    t = np.arange(T)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 子图1：功率平衡
    ax = axes[0, 0]
    ax.plot(t, solution['thermal'], 'r-', label='火电出力')
    ax.plot(t, wind_mean, 'b-', label='风电预测均值')
    ax.plot(t, solution['discharge'], 'g-', label='储能放电')
    ax.plot(t, solution['charge'], 'c-', label='储能充电')
    ax.plot(t, np.full(T, params['demand']), 'k--', label='负荷')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('功率平衡')
    ax.legend()
    ax.grid(True)
    
    # 子图2：备用容量
    ax = axes[0, 1]
    ax.bar(t, solution['reserve'], alpha=0.7, label='备用容量')
    ax.plot(t, params['beta'] * wind_std, 'r--', label='备用需求')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('备用 (kW)')
    ax.set_title('备用容量配置')
    ax.legend()
    ax.grid(True)
    
    # 子图3：SOC
    ax = axes[1, 0]
    ax.plot(t, solution['soc'] / params['S_max'], 'b-o')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('SOC (%)')
    ax.set_title('储能荷电状态')
    ax.grid(True)
    ax.set_ylim(0, 1)
    
    # 子图4：成本分解
    ax = axes[1, 1]
    cost_items = ['燃料', '备用', '购电', '储能', '启停']
    cost_values = [
        cost_breakdown['fuel_cost'],
        cost_breakdown['reserve_cost'],
        cost_breakdown['purchase_cost'],
        cost_breakdown['storage_cost'],
        cost_breakdown['startup_cost'] + cost_breakdown['shutdown_cost']
    ]
    ax.bar(cost_items, cost_values)
    ax.set_ylabel('成本 (元)')
    ax.set_title('成本分解')
    for i, v in enumerate(cost_values):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom')
    
    # 子图5：风电不确定性
    ax = axes[2, 0]
    ax.plot(t, wind_mean, 'b-', label='均值')
    ax.fill_between(t, wind_mean - 2*wind_std, wind_mean + 2*wind_std, alpha=0.3, label='95%置信区间')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('风电功率 (kW)')
    ax.set_title('风电预测不确定性')
    ax.legend()
    ax.grid(True)
    
    # 子图6：总功率与负荷
    ax = axes[2, 1]
    total_power = solution['thermal'] + wind_mean + solution['discharge'] - solution['charge']
    ax.plot(t, total_power, 'r-', label='总发电')
    ax.plot(t, np.full(T, params['demand']), 'k--', label='负荷')
    ax.fill_between(t, total_power - solution['reserve'], total_power + solution['reserve'], alpha=0.3, label='备用范围')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('总发电与负荷平衡')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('dispatch_results.png', dpi=150)
    plt.show()

def evaluate_outage_risk(solution, params, wind_mean, wind_std, wind_q95=None):
    """
    评估缺电概率（实际风电低于可用功率的概率）
    
    假设风电实际功率 w_t ~ N(μ_t, σ_t^2)
    可用功率 = 火电 + 风电预测均值 + 储能放电 - 储能充电 + 备用容量
    当 w_t < (火电 + 储能放电 - 储能充电) 时，可能缺电（考虑备用容量后）
    简化：定义缺电事件为 w_t < (火电 + 储能净放电) - 备用容量（即备用不足以覆盖负偏差）
    """
    T = params['T']
    g = solution['thermal']
    e_dis = solution['discharge']
    e_ch = solution['charge']
    r = solution['reserve']
    
    # 净火电+储能出力（不含风电）
    net_dispatch = g + e_dis - e_ch
    
    # 计算每个时段的缺电概率
    outage_probs = []
    for t in range(T):
        # 可用功率 = net_dispatch[t] + wind_mean[t] + r[t] (备用可上调)
        # 实际风电 w_t ~ N(wind_mean[t], wind_std[t]**2)
        # 缺电条件： w_t < net_dispatch[t] - r[t] （风电低于火电+储能净出力减去备用）
        # 即风电不足部分超过备用容量
        threshold = net_dispatch[t] - r[t]
        # 如果风电均值已经大于阈值，缺电概率较小
        if wind_std[t] < 1e-8:
            prob = 1.0 if wind_mean[t] < threshold else 0.0
        else:
            z = (threshold - wind_mean[t]) / wind_std[t]
            # 正态分布CDF
            from scipy.stats import norm
            prob = norm.cdf(z)
        outage_probs.append(prob)
    
    outage_probs = np.array(outage_probs)
    avg_outage = np.mean(outage_probs)
    max_outage = np.max(outage_probs)
    critical_hours = np.sum(outage_probs > 0.05)  # 缺电概率>5%的时段数
    
    return {
        'outage_probabilities': outage_probs,
        'average_outage': avg_outage,
        'max_outage': max_outage,
        'critical_hours': critical_hours
    }

def compare_probabilistic_modes():
    """比较不同概率预测模式的调度效果"""
    modes = ['point', 'gaussian', 'quantile']
    results = {}
    
    params = get_default_params()
    
    for mode in modes:
        print(f"\n========== 模式: {mode} ==========")
        wind_mean, wind_std, wind_q95 = load_wind_forecast(mode=mode)
        params['T'] = len(wind_mean)
        
        # 求解调度
        x_opt = solve_dispatch(params, wind_mean, wind_std)
        if x_opt is None:
            print(f"模式 {mode} 优化失败，跳过")
            continue
        
        solution = parse_solution(x_opt, params)
        cost_breakdown = evaluate_cost(solution, params, wind_mean, wind_std)
        outage_risk = evaluate_outage_risk(solution, params, wind_mean, wind_std, wind_q95)
        
        results[mode] = {
            'solution': solution,
            'cost_breakdown': cost_breakdown,
            'outage_risk': outage_risk,
            'wind_mean': wind_mean,
            'wind_std': wind_std,
            'wind_q95': wind_q95
        }
        
        print(f"总成本: {cost_breakdown['total_cost']:.2f} 元")
        print(f"备用成本: {cost_breakdown['reserve_cost']:.2f} 元")
        print(f"平均缺电概率: {outage_risk['average_outage']:.4f}")
        print(f"最大缺电概率: {outage_risk['max_outage']:.4f}")
        print(f"高风险时段数: {outage_risk['critical_hours']}")
    
    # 生成对比报告
    print("\n" + "="*70)
    print("概率预测模式对比报告")
    print("="*70)
    for mode in modes:
        if mode not in results:
            continue
        r = results[mode]
        print(f"{mode:12s} | 总成本: {r['cost_breakdown']['total_cost']:8.2f} 元 | "
              f"备用成本: {r['cost_breakdown']['reserve_cost']:8.2f} 元 | "
              f"缺电概率: {r['outage_risk']['average_outage']:.4f} | "
              f"高风险时段: {r['outage_risk']['critical_hours']}")
    
    # 保存对比结果
    df = pd.DataFrame({
        'mode': modes,
        'total_cost': [results.get(m, {}).get('cost_breakdown', {}).get('total_cost', np.nan) for m in modes],
        'reserve_cost': [results.get(m, {}).get('cost_breakdown', {}).get('reserve_cost', np.nan) for m in modes],
        'avg_outage': [results.get(m, {}).get('outage_risk', {}).get('average_outage', np.nan) for m in modes],
        'critical_hours': [results.get(m, {}).get('outage_risk', {}).get('critical_hours', np.nan) for m in modes]
    })
    df.to_csv('probabilistic_comparison.csv', index=False)
    print(f"\n对比结果已保存为 probabilistic_comparison.csv")
    
    return results

def main():
    """主函数（兼容旧版本）"""
    print("风电场景调度模型")
    params = get_default_params()
    print("参数:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 加载风电预测（默认点预测）
    wind_mean, wind_std, _ = load_wind_forecast(mode='point')
    params['T'] = len(wind_mean)
    print(f"风电预测时段数: {params['T']}")
    print(f"风电均值范围: [{wind_mean.min():.0f}, {wind_mean.max():.0f}] kW")
    print(f"风电标准差范围: [{wind_std.min():.0f}, {wind_std.max():.0f}] kW")
    
    # 求解
    x_opt = solve_dispatch(params, wind_mean, wind_std)
    if x_opt is None:
        print("无法获得优化解，使用初始猜测演示")
        x_opt = np.zeros(4 * params['T'])
        x_opt[:params['T']] = (params['g_min'] + params['g_max']) / 2
        x_opt[params['T']:2*params['T']] = params['beta'] * wind_std
    
    # 解析结果
    solution = parse_solution(x_opt, params)
    cost_breakdown = evaluate_cost(solution, params, wind_mean, wind_std)
    outage_risk = evaluate_outage_risk(solution, params, wind_mean, wind_std)
    
    print("\n调度结果摘要:")
    print(f"火电出力: [{solution['thermal'].min():.0f}, {solution['thermal'].max():.0f}] kW")
    print(f"备用容量: [{solution['reserve'].min():.0f}, {solution['reserve'].max():.0f}] kW")
    print(f"储能充电: 最大值 {solution['charge'].max():.0f} kW")
    print(f"储能放电: 最大值 {solution['discharge'].max():.0f} kW")
    print(f"SOC范围: [{solution['soc'].min()/params['S_max']:.2%}, {solution['soc'].max()/params['S_max']:.2%}]")
    
    print("\n成本分解:")
    for k, v in cost_breakdown.items():
        print(f"  {k}: {v:.2f} 元")
    
    print("\n风险指标:")
    print(f"平均缺电概率: {outage_risk['average_outage']:.4f}")
    print(f"最大缺电概率: {outage_risk['max_outage']:.4f}")
    print(f"高风险时段数 (p>0.05): {outage_risk['critical_hours']}")
    
    # 绘图
    try:
        plot_dispatch(solution, params, wind_mean, wind_std, cost_breakdown)
        print("图表已保存为 dispatch_results.png")
    except Exception as e:
        print(f"绘图失败: {e}")
    
    # 保存结果到CSV
    df = pd.DataFrame({
        'time': np.arange(params['T']),
        'wind_mean': wind_mean,
        'wind_std': wind_std,
        'thermal': solution['thermal'],
        'reserve': solution['reserve'],
        'charge': solution['charge'],
        'discharge': solution['discharge'],
        'soc': solution['soc'],
        'soc_percent': solution['soc'] / params['S_max'],
        'demand': params['demand']
    })
    df.to_csv('dispatch_schedule.csv', index=False)
    print("调度方案已保存为 dispatch_schedule.csv")
    
    return solution, cost_breakdown, outage_risk

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='风电调度模型')
    parser.add_argument('--mode', type=str, default='point', choices=['point', 'gaussian', 'quantile', 'compare'],
                        help='预测模式: point, gaussian, quantile, compare (对比所有模式)')
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_probabilistic_modes()
    else:
        main()