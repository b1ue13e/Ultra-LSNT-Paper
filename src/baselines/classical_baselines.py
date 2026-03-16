"""
经典统计/物理基线模型
=======================================
为风电预测提供工程界广泛接受的经典基线：
1. Persistence (持续性模型)
2. Seasonal Naive
3. ARIMA / SARIMA
4. 基于NWP特征的简单机器学习（随机森林）
5. 物理模型简化版（功率曲线拟合）

所有模型均支持分层的性能分析（按风速、风向、季节、时段等工况分组）。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 统计模型依赖
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("警告: statsmodels 或 pmdarima 未安装，ARIMA/SARIMA 模型不可用。")

# 机器学习依赖
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn 未安装，随机森林模型不可用。")


class PersistenceModel:
    """持久性模型 (最简基线)
    
    将最近的一个值作为未来所有步的预测。
    风电预测中常用“朴素持续性”（naive persistence）。
    """
    
    def __init__(self, horizon: int = 24):
        self.horizon = horizon
        self.last_value = None
    
    def fit(self, y: np.ndarray):
        """记录最后一个观测值"""
        self.last_value = y[-1]
        return self
    
    def predict(self, n_periods: int = None) -> np.ndarray:
        if self.last_value is None:
            raise ValueError("模型尚未训练")
        n = n_periods if n_periods is not None else self.horizon
        return np.full(n, self.last_value)
    
    def forecast(self, X: np.ndarray = None) -> np.ndarray:
        """单步预测（使用X的最后一个观测值）"""
        if X is not None and len(X) > 0:
            return np.full(self.horizon, X[-1])
        return self.predict()


class SeasonalNaive:
    """季节性朴素模型
    
    使用上一个完整周期的对应位置值作为预测。
    例如，对于日周期（96个15分钟间隔），用昨天的同一时刻预测今天。
    """
    
    def __init__(self, season_length: int = 96, horizon: int = 24):
        self.season_length = season_length
        self.horizon = horizon
        self.seasonal_values = None
    
    def fit(self, y: np.ndarray):
        """提取上一个完整周期的值"""
        if len(y) < self.season_length:
            raise ValueError(f"数据长度 {len(y)} 小于季节长度 {self.season_length}")
        self.seasonal_values = y[-self.season_length:]
        return self
    
    def predict(self, n_periods: int = None) -> np.ndarray:
        if self.seasonal_values is None:
            raise ValueError("模型尚未训练")
        n = n_periods if n_periods is not None else self.horizon
        # 重复季节模式直到填满预测长度
        repeats = int(np.ceil(n / self.season_length))
        full_cycle = np.tile(self.seasonal_values, repeats)
        return full_cycle[:n]
    
    def forecast(self, X: np.ndarray = None) -> np.ndarray:
        """使用最近的历史数据预测"""
        if X is not None and len(X) >= self.season_length:
            seasonal_vals = X[-self.season_length:]
            n = self.horizon
            repeats = int(np.ceil(n / self.season_length))
            full_cycle = np.tile(seasonal_vals, repeats)
            return full_cycle[:n]
        return self.predict()


class ARIMABaseline:
    """ARIMA/SARIMA 基线模型
    
    使用自动参数选择（pmdarima.auto_arima）或固定参数。
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 use_auto: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_auto = use_auto
        self.model = None
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, X: np.ndarray = None):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels 或 pmdarima 未安装")
        
        if self.use_auto:
            try:
                self.model = auto_arima(
                    y, seasonal=seasonal_order[3] > 0,
                    m=seasonal_order[3] if seasonal_order[3] > 0 else 1,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
            except:
                # 回退到固定参数
                self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order).fit()
        else:
            self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order).fit()
        
        self.is_fitted = True
        return self
    
    def predict(self, n_periods: int = 24, X: np.ndarray = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.forecast(steps=n_periods).values
    
    def forecast(self, X: np.ndarray = None) -> np.ndarray:
        return self.predict(n_periods=24, X=X)


class NwpRandomForest:
    """基于NWP（数值天气预报）特征的随机森林模型
    
    使用风速、风向、温度、湿度、气压等特征进行预测。
    这是一个简化的NWP‑ML模型，模拟工程实践中NWP后处理。
    """
    
    def __init__(self, horizon: int = 24, n_estimators: int = 100):
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从原始数据创建NWP特征（包括滞后项）"""
        df_feat = df.copy()
        # 滞后特征
        for lag in [1, 2, 3, 4, 96]:  # 前几个时刻和前一日同期
            df_feat[f'wind_lag_{lag}'] = df['windspeed'].shift(lag)
            df_feat[f'power_lag_{lag}'] = df['power'].shift(lag)
        # 时间特征
        if 'hour_sin' not in df.columns:
            # 假设有索引或时间列，这里简化
            pass
        return df_feat.dropna()
    
    def fit(self, df: pd.DataFrame, target_col: str = 'power'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn 未安装")
        
        df_feat = self._create_features(df)
        X = df_feat.drop(columns=[target_col])
        y = df_feat[target_col]
        
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型尚未训练")
        df_feat = self._create_features(df)
        X = df_feat[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def forecast(self, df: pd.DataFrame, steps: int = 24) -> np.ndarray:
        """递归预测（使用预测值作为滞后特征）"""
        # 简化：返回最后steps个预测
        preds = self.predict(df)
        return preds[-steps:] if len(preds) >= steps else preds


class PowerCurveModel:
    """物理模型简化：风机功率曲线拟合
    
    使用风速‑功率关系（通常为分段三次函数）进行预测。
    这是风电行业最基础的物理模型。
    """
    
    def __init__(self, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0,
                 rated_power: float = 100.0):
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out
        self.rated_power = rated_power
        self.fitted_params = None
    
    def _power_curve(self, wind_speed: np.ndarray) -> np.ndarray:
        """标准IEC功率曲线"""
        power = np.zeros_like(wind_speed)
        # 低于切入风速
        power[wind_speed < self.cut_in] = 0
        # 切入‑额定之间（三次关系）
        mask = (wind_speed >= self.cut_in) & (wind_speed < self.rated)
        power[mask] = self.rated_power * ((wind_speed[mask] - self.cut_in) / (self.rated - self.cut_in)) ** 3
        # 额定‑切出之间（恒定额定功率）
        mask = (wind_speed >= self.rated) & (wind_speed < self.cut_out)
        power[mask] = self.rated_power
        # 高于切出风速（停机）
        power[wind_speed >= self.cut_out] = 0
        return power
    
    def fit(self, df: pd.DataFrame, wind_col: str = 'windspeed', power_col: str = 'power'):
        """可拟合实际功率曲线的参数（这里使用标准曲线）"""
        # 实际中可拟合非线性关系，此处保持标准曲线
        return self
    
    def predict(self, wind_speed: np.ndarray) -> np.ndarray:
        return self._power_curve(wind_speed)
    
    def forecast(self, wind_forecast: np.ndarray) -> np.ndarray:
        return self.predict(wind_forecast)


def evaluate_baseline(model, X_train, y_train, X_test, y_test, 
                      metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """评估单个基线模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if hasattr(model, 'predict') else model.forecast(X_test)
    
    if len(y_pred) != len(y_test):
        # 截断或填充
        min_len = min(len(y_pred), len(y_test))
        y_pred = y_pred[:min_len]
        y_test = y_test[:min_len]
    
    results = {}
    for name, func in metric_funcs.items():
        try:
            results[name] = func(y_test, y_pred)
        except Exception as e:
            results[name] = np.nan
            print(f"指标 {name} 计算失败: {e}")
    
    return results


def run_all_baselines(df: pd.DataFrame, target_col: str = 'power',
                      train_ratio: float = 0.7, horizon: int = 24,
                      seasonal_period: int = 96) -> pd.DataFrame:
    """
    运行所有经典基线模型并返回性能对比表格。
    
    Parameters
    ----------
    df : pd.DataFrame
        包含特征和目标的时间序列数据
    target_col : str
        目标列名
    train_ratio : float
        训练集比例
    horizon : int
        预测步长
    seasonal_period : int
        季节长度（例如日周期96）
    
    Returns
    -------
    pd.DataFrame
        各模型的评估指标（行：模型，列：指标）
    """
    # 划分训练集和测试集（时间顺序）
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    # 提取目标序列
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # 定义评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    def mse(y, y_pred): return mean_squared_error(y, y_pred)
    def rmse(y, y_pred): return np.sqrt(mean_squared_error(y, y_pred))
    def mae(y, y_pred): return mean_absolute_error(y, y_pred)
    def r2(y, y_pred): return r2_score(y, y_pred)
    def mape(y, y_pred):
        mask = y != 0
        if mask.any():
            return np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        return np.nan
    
    metric_funcs = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
    }
    
    results = {}
    
    # 1. Persistence
    print("训练 Persistence 模型...")
    persistence = PersistenceModel(horizon=horizon)
    persistence.fit(y_train)
    # 为测试集的每个起始点计算持续性预测（滑动窗口）
    y_pred_pers = []
    for i in range(len(y_test) - horizon):
        y_pred_pers.append(persistence.forecast(y_test[i:i+horizon]))
    if y_pred_pers:
        y_pred_pers = np.array(y_pred_pers).mean(axis=0)  # 简单平均
    else:
        y_pred_pers = persistence.forecast(y_train)
    # 评估
    if len(y_pred_pers) > len(y_test):
        y_pred_pers = y_pred_pers[:len(y_test)]
    results['Persistence'] = {name: func(y_test[:len(y_pred_pers)], y_pred_pers) 
                              for name, func in metric_funcs.items()}
    
    # 2. Seasonal Naive
    print("训练 Seasonal Naive 模型...")
    snaive = SeasonalNaive(season_length=seasonal_period, horizon=horizon)
    snaive.fit(y_train)
    y_pred_sn = snaive.forecast(y_test)
    if len(y_pred_sn) > len(y_test):
        y_pred_sn = y_pred_sn[:len(y_test)]
    results['Seasonal_Naive'] = {name: func(y_test[:len(y_pred_sn)], y_pred_sn) 
                                 for name, func in metric_funcs.items()}
    
    # 3. ARIMA (若可用)
    if STATSMODELS_AVAILABLE and len(y_train) > 100:
        print("训练 ARIMA 模型...")
        try:
            arima = ARIMABaseline(order=(1,1,1), seasonal_order=(1,1,1,seasonal_period), use_auto=True)
            arima.fit(y_train)
            y_pred_ar = arima.predict(n_periods=len(y_test))
            results['ARIMA'] = {name: func(y_test[:len(y_pred_ar)], y_pred_ar) 
                                for name, func in metric_funcs.items()}
        except Exception as e:
            print(f"ARIMA 训练失败: {e}")
            results['ARIMA'] = {name: np.nan for name in metric_funcs.keys()}
    else:
        results['ARIMA'] = {name: np.nan for name in metric_funcs.keys()}
    
    # 4. 功率曲线模型（物理基线）
    print("训练 Power Curve 模型...")
    if 'windspeed' in df.columns:
        wind_train = train_df['windspeed'].values
        wind_test = test_df['windspeed'].values
        pc_model = PowerCurveModel()
        pc_model.fit(train_df)
        y_pred_pc = pc_model.predict(wind_test)
        results['PowerCurve'] = {name: func(y_test, y_pred_pc) 
                                 for name, func in metric_funcs.items()}
    else:
        results['PowerCurve'] = {name: np.nan for name in metric_funcs.keys()}
    
    # 5. 随机森林（NWP特征）若可用
    if SKLEARN_AVAILABLE and len(df.columns) > 1:
        print("训练 Random Forest (NWP) 模型...")
        try:
            rf_model = NwpRandomForest(horizon=horizon)
            rf_model.fit(train_df, target_col=target_col)
            y_pred_rf = rf_model.forecast(test_df, steps=len(y_test))
            results['RandomForest'] = {name: func(y_test[:len(y_pred_rf)], y_pred_rf) 
                                       for name, func in metric_funcs.items()}
        except Exception as e:
            print(f"Random Forest 训练失败: {e}")
            results['RandomForest'] = {name: np.nan for name in metric_funcs.keys()}
    else:
        results['RandomForest'] = {name: np.nan for name in metric_funcs.keys()}
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('baseline_results', exist_ok=True)
    results_df.to_csv(f'baseline_results/classical_baselines_{timestamp}.csv')
    results_df.to_json(f'baseline_results/classical_baselines_{timestamp}.json', indent=2)
    
    print("\n经典基线模型性能对比:")
    print(results_df)
    
    return results_df


def stratified_performance_analysis(df: pd.DataFrame, predictions: Dict[str, np.ndarray],
                                    target: np.ndarray, conditions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    按工况分层的性能分析
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    predictions : dict
        各模型的预测值 {模型名: 预测数组}
    target : np.ndarray
        真实值数组
    conditions : dict
        工况条件 {工况名: 布尔数组}，例如 {'high_wind': wind > 10}
    
    Returns
    -------
    pd.DataFrame
        每个工况下各模型的性能指标
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    stratified_results = []
    
    for cond_name, mask in conditions.items():
        if mask.sum() == 0:
            continue
        # 截断mask长度
        min_len = min(len(mask), len(target))
        mask = mask[:min_len]
        for model_name, pred in predictions.items():
            # 确保pred长度匹配
            pred = pred[:min_len]
            # 只在该工况下的样本
            y_true_cond = target[mask]
            y_pred_cond = pred[mask]
            if len(y_true_cond) < 10:
                continue
            # 计算指标
            mse_val = mean_squared_error(y_true_cond, y_pred_cond)
            rmse_val = np.sqrt(mse_val)
            mae_val = mean_absolute_error(y_true_cond, y_pred_cond)
            r2_val = r2_score(y_true_cond, y_pred_cond)
            
            stratified_results.append({
                'condition': cond_name,
                'model': model_name,
                'samples': len(y_true_cond),
                'MSE': mse_val,
                'RMSE': rmse_val,
                'MAE': mae_val,
                'R2': r2_val
            })
    
    results_df = pd.DataFrame(stratified_results)
    if not results_df.empty:
        results_df = results_df.round(4)
    
    return results_df


if __name__ == '__main__':
    # 示例使用
    print("加载风电数据...")
    df = pd.read_csv('wind_final.csv')
    
    # 选择特征列（假设目标为'power'）
    feature_cols = ['windspeed', 'winddirection', 'temperature', 'humidity', 'pressure']
    if all(col in df.columns for col in feature_cols):
        df = df[feature_cols + ['power']].dropna()
    else:
        # 使用所有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols].dropna()
    
    print(f"数据形状: {df.shape}")
    
    # 运行所有基线
    results = run_all_baselines(df, target_col='power', horizon=24, seasonal_period=96)
    
    print("\n✅ 经典基线模型评估完成!")
    print(f"结果保存至: baseline_results/classical_baselines_*.csv")
    
    # 可选：生成工况分层分析
    print("\n生成工况分层分析...")
    # 定义工况条件
    conditions = {}
    if 'windspeed' in df.columns:
        wind = df['windspeed'].values
        conditions['low_wind'] = wind < 5
        conditions['medium_wind'] = (wind >= 5) & (wind < 10)
        conditions['high_wind'] = wind >= 10
        conditions['ramp_up'] = np.concatenate([ [False], (wind[1:] - wind[:-1]) > 2 ])  # 简化
        conditions['ramp_down'] = np.concatenate([ [False], (wind[1:] - wind[:-1]) < -2 ])
    
    # 需要预测值，这里简单用持久性模型生成示例
    # 实际应用中应从各模型获取预测值
    if conditions:
        # 为演示，使用持久性模型的预测（仅示例）
        y_train = df['power'].values[:int(0.7*len(df))]
        persistence = PersistenceModel(horizon=24)
        persistence.fit(y_train)
        y_pred_pers = persistence.forecast(df['power'].values)
        predictions = {'Persistence': y_pred_pers}
        
        stratified_df = stratified_performance_analysis(
            df, predictions, df['power'].values, conditions
        )
        if not stratified_df.empty:
            print(stratified_df.head(20))
            stratified_df.to_csv('baseline_results/stratified_analysis.csv', index=False)
            print("分层分析保存至 baseline_results/stratified_analysis.csv")