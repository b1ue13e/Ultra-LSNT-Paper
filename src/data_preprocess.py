"""
Ultra-LSNT 数据预处理脚本
==========================

支持你已有的数据格式：
1. 风电机组运行数据集 (xlsx)
2. 电力负荷数据集 (xlsx)  
3. GEFCom2012 竞赛数据
4. 城市级空气质量/气象数据 (xls)
5. 时序论文负荷数据 (xlsx)

使用方法：
    # 查看数据信息
    python data_preprocess.py --info --data your_data.xlsx
    
    # 预处理风电数据
    python data_preprocess.py --data wind_data.xlsx --target power --output processed_wind.csv
    
    # 预处理负荷数据
    python data_preprocess.py --data load_data.xlsx --target load --output processed_load.csv
    
    # 合并多个城市数据
    python data_preprocess.py --merge_dir ./city_data/ --output merged_data.csv
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import sys
import subprocess
warnings.filterwarnings('ignore')

# 导入force_fix中的功能
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from force_fix import check_and_fix_power_column
except ImportError:
    print("[WARNING] 无法导入force_fix模块，将使用简化的功率检查")
    def check_and_fix_power_column(df, power_col='power'):
        """简化的功率列检查"""
        if power_col not in df.columns:
            return df
        # 确保非负
        df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
        df = df.dropna(subset=[power_col])
        negative_count = (df[power_col] < 0).sum()
        if negative_count > 0:
            df.loc[df[power_col] < 0, power_col] = 0
        return df


def print_data_info(filepath: str):
    """打印数据文件信息"""
    print("=" * 70)
    print(f"[DATA] 数据文件信息: {filepath}")
    print("=" * 70)
    
    # 读取数据
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        # Excel 文件可能有多个 sheet
        xls = pd.ExcelFile(filepath)
        print(f"\n[SHEETS] Sheet 列表: {xls.sheet_names}")
        
        for sheet in xls.sheet_names[:3]:  # 只显示前3个sheet
            print(f"\n--- Sheet: {sheet} ---")
            df = pd.read_excel(filepath, sheet_name=sheet)
            print(f"形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print(f"\n前5行:")
            print(df.head())
            print(f"\n数据类型:")
            print(df.dtypes)
            print(f"\n缺失值:")
            print(df.isnull().sum())
        return
    
    print(f"\n形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n前5行:")
    print(df.head())
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n统计信息:")
    print(df.describe())
    print(f"\n缺失值:")
    print(df.isnull().sum())


def add_time_features(df: pd.DataFrame, time_col: str = None) -> pd.DataFrame:
    """添加时间特征"""
    
    # 尝试找到时间列
    time_columns = ['time', 'datetime', 'date', 'timestamp', '时间', '日期']
    
    if time_col is None:
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
    
    if time_col is None:
        print("[WARNING] 未找到时间列，尝试使用索引生成时间特征")
        # 假设数据是按时间顺序排列的
        n = len(df)
        hours = np.arange(n) % 24
        days = (np.arange(n) // 24) % 7
    else:
        print(f"[OK] 使用时间列: {time_col}")
        df[time_col] = pd.to_datetime(df[time_col])
        hours = df[time_col].dt.hour
        days = df[time_col].dt.dayofweek
        
        # 删除原始时间列
        df = df.drop(columns=[time_col])
    
    # 添加周期性时间特征
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['day_sin'] = np.sin(2 * np.pi * days / 7)
    df['day_cos'] = np.cos(2 * np.pi * days / 7)
    
    return df


def preprocess_wind_power_data(filepath: str, target_col: str = None, 
                                sheet_name: str = None) -> pd.DataFrame:
    """
    预处理风电数据
    
    常见列名映射：
    - 风速: wind_speed, WS, 风速, v
    - 风向: wind_direction, WD, 风向
    - 功率: power, P, 功率, 发电量, active_power
    - 温度: temperature, T, temp, 温度
    """
    print("\n[WIND] 预处理风电数据...")
    
    # 读取数据
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath)
    
    print(f"   原始形状: {df.shape}")
    print(f"   原始列名: {list(df.columns)}")
    
    # 列名标准化映射
    column_mapping = {
        # 风速
        'WS': 'wind_speed', 'ws': 'wind_speed', '风速': 'wind_speed',
        'Ws': 'wind_speed', 'windspeed': 'wind_speed', 'V': 'wind_speed',
        'wind': 'wind_speed', 'WIND_SPEED': 'wind_speed',
        
        # 风向
        'WD': 'wind_direction', 'wd': 'wind_direction', '风向': 'wind_direction',
        'Wd': 'wind_direction', 'winddirection': 'wind_direction',
        'WIND_DIRECTION': 'wind_direction',
        
        # 功率
        'P': 'power', 'p': 'power', '功率': 'power', '发电量': 'power',
        'Power': 'power', 'POWER': 'power', 'active_power': 'power',
        'ActivePower': 'power', '有功功率': 'power', 'Patv': 'power',
        
        # 温度
        'T': 'temperature', 't': 'temperature', '温度': 'temperature',
        'Temp': 'temperature', 'temp': 'temperature', 'TEMPERATURE': 'temperature',
        'Etmp': 'temperature',
        
        # 湿度
        'humidity': 'humidity', '湿度': 'humidity', 'RH': 'humidity',
        
        # 气压
        'pressure': 'pressure', '气压': 'pressure', 'Pres': 'pressure',
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 添加时间特征
    df = add_time_features(df)
    
    # 确定目标列
    if target_col is None:
        if 'power' in df.columns:
            target_col = 'power'
        else:
            # 使用最后一列
            target_col = df.columns[-1]
            print(f"   [WARNING] 未指定目标列，使用: {target_col}")
    
    # 确保目标列在最后
    if target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    
    # 删除非数值列
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"   [REMOVE] 删除非数值列: {non_numeric}")
        df = df.drop(columns=non_numeric)
    
    # 处理缺失值
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"   [FILL] 填充缺失值: {missing} 个")
        df = df.ffill().bfill()
    
    # 处理异常值（简单的范围裁剪）
    for col in df.columns:
        q1 = df[col].quantile(0.001)
        q99 = df[col].quantile(0.999)
        df[col] = df[col].clip(q1, q99)
    
    print(f"   处理后形状: {df.shape}")
    print(f"   处理后列名: {list(df.columns)}")
    
    return df


def preprocess_load_data(filepath: str, target_col: str = None,
                         sheet_name: str = None) -> pd.DataFrame:
    """
    预处理电力负荷数据
    
    常见列名映射：
    - 负荷: load, Load, 负荷, demand, 用电量
    - 温度: temperature, T, 温度
    """
    print("\n[LOAD] 预处理电力负荷数据...")
    
    # 读取数据
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath)
    
    print(f"   原始形状: {df.shape}")
    print(f"   原始列名: {list(df.columns)}")
    
    # 列名标准化映射
    column_mapping = {
        # 负荷
        'Load': 'load', 'LOAD': 'load', '负荷': 'load',
        'demand': 'load', 'Demand': 'load', '用电量': 'load',
        'consumption': 'load', '电量': 'load', 'power': 'load',
        
        # 温度
        'T': 'temperature', 't': 'temperature', '温度': 'temperature',
        'Temp': 'temperature', 'temp': 'temperature',
        
        # 湿度
        'humidity': 'humidity', '湿度': 'humidity',
    }
    
    df = df.rename(columns=column_mapping)
    
    # 添加时间特征
    df = add_time_features(df)
    
    # 确定目标列
    if target_col is None:
        if 'load' in df.columns:
            target_col = 'load'
        else:
            target_col = df.columns[-1]
    
    # 确保目标列在最后
    if target_col in df.columns:
        cols = [c for c in df.columns if c != target_col] + [target_col]
        df = df[cols]
    
    # 删除非数值列
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"   [REMOVE] 删除非数值列: {non_numeric}")
        df = df.drop(columns=non_numeric)
    
    # 处理缺失值
    df = df.ffill().bfill()
    
    print(f"   处理后形状: {df.shape}")
    print(f"   处理后列名: {list(df.columns)}")
    
    return df


def preprocess_gefcom_data(filepath: str) -> pd.DataFrame:
    """预处理 GEFCom2012 竞赛数据"""
    print("\n[GEFCOM] 预处理 GEFCom2012 数据...")
    
    df = pd.read_excel(filepath) if not filepath.endswith('.csv') else pd.read_csv(filepath)
    
    # GEFCom 特有的列名处理
    column_mapping = {
        'ZONEID': 'zone_id',
        'TIMESTAMP': 'timestamp', 
        'TARGETVAR': 'load',
        'VAR01': 'var01', 'VAR02': 'var02', 'VAR03': 'var03',
    }
    
    df = df.rename(columns=column_mapping)
    df = add_time_features(df, 'timestamp')
    
    # 删除非数值列
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)
    
    df = df.ffill().bfill()
    
    return df


def merge_city_data(data_dir: str, output_file: str = None) -> pd.DataFrame:
    """合并多个城市的数据文件（防弹版：支持 CSV/Excel 混读 + 自动编码检测）"""
    print(f"\n[CITY] 合并城市数据: {data_dir}")

    data_dir = Path(data_dir)
    # 递归查找所有可能的文件格式
    all_files = list(data_dir.rglob('*.csv')) + list(data_dir.rglob('*.xls')) + list(data_dir.rglob('*.xlsx'))

    print(f"   找到 {len(all_files)} 个文件")

    all_dfs = []
    success_count = 0

    # 限制处理文件数量，防止内存爆炸 (如果你内存够大，可以把 [:100] 去掉)
    # 这里我们先处理所有文件，看看进度
    for i, f in enumerate(all_files):
        city_name = f.stem
        # 只打印前几个和每10个的进度，防止刷屏
        if i < 5 or i % 20 == 0:
            print(f"   [PROGRESS] [{i + 1}/{len(all_files)}] 处理: {city_name} ({f.suffix})")

        df = None

        # === 尝试方案 A: 它是 CSV? ===
        try:
            # 优先尝试 UTF-8
            df = pd.read_csv(f)
        except:
            try:
                # 失败了？试试 GBK (中文环境常见)
                df = pd.read_csv(f, encoding='gbk')
            except:
                pass  # 继续尝试方案 B

        # === 尝试方案 B: 它是 Excel? ===
        if df is None:
            try:
                df = pd.read_excel(f)
            except:
                pass

        # === 尝试方案 C: 它是 '假Excel' (HTML表格)? ===
        if df is None:
            try:
                # 有些老旧系统导出的 .xls 其实是 HTML
                dfs = pd.read_html(str(f))
                if dfs: df = dfs[0]
            except:
                pass

        # === 最终判定 ===
        if df is not None:
            # 标准化列名（防止不同文件列名大小写不一致）
            df.columns = [str(c).strip() for c in df.columns]

            # 添加城市标签
            df['city'] = city_name
            all_dfs.append(df)
            success_count += 1
        else:
            print(f"   [ERROR] 彻底读取失败: {f.name} (可能是格式损坏)")

    if all_dfs:
        print(f"\n   [MERGE] 正在合并 {success_count} 个数据框...")
        merged = pd.concat(all_dfs, ignore_index=True)

        # 简单清洗：把所有列名转小写，方便后续处理
        merged.columns = [str(c).lower() for c in merged.columns]

        print(f"   合并后形状: {merged.shape}")

        if output_file:
            # 保存为 CSV，使用 utf-8-sig 确保 Excel 打开不乱码
            merged.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"   [SAVE] 保存到: {output_file}")

        return merged

    print("   [WARNING] 没有成功合并任何数据。")
    return None


def auto_detect_and_process(filepath: str, target_col: str = None) -> pd.DataFrame:
    """自动检测数据类型并处理"""
    print(f"\n[AUTO] 自动检测数据类型: {filepath}")
    
    filename = Path(filepath).stem.lower()
    
    # 根据文件名关键词判断数据类型
    wind_keywords = ['wind', '风电', '风速', '风机', 'turbine']
    load_keywords = ['load', '负荷', '电力', '用电', 'demand', 'consumption']
    gefcom_keywords = ['gefcom', 'gef', 'competition']
    
    if any(kw in filename for kw in wind_keywords):
        print("   [DETECT] 检测为: 风电数据")
        return preprocess_wind_power_data(filepath, target_col)
    
    elif any(kw in filename for kw in load_keywords):
        print("   [DETECT] 检测为: 电力负荷数据")
        return preprocess_load_data(filepath, target_col)
    
    elif any(kw in filename for kw in gefcom_keywords):
        print("   [DETECT] 检测为: GEFCom竞赛数据")
        return preprocess_gefcom_data(filepath)
    
    else:
        print("   [DETECT] 检测为: 通用时序数据")
        # 通用处理
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        df = add_time_features(df)
        
        # 删除非数值列
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            df = df.drop(columns=non_numeric)
        
        df = df.ffill().bfill()
        
        # 移动目标列到最后
        if target_col and target_col in df.columns:
            cols = [c for c in df.columns if c != target_col] + [target_col]
            df = df[cols]
        
        return df


def validate_data(df: pd.DataFrame, min_samples: int = 1000):
    """验证数据是否适合训练"""
    print("\n[VALIDATION] 数据验证:")
    
    issues = []
    
    # 检查样本数
    if len(df) < min_samples:
        issues.append(f"样本数不足: {len(df)} < {min_samples}")
    else:
        print(f"   [OK] 样本数: {len(df)}")
    
    # 检查特征数
    if df.shape[1] < 2:
        issues.append("特征数不足，至少需要1个特征+1个目标")
    else:
        print(f"   [OK] 特征数: {df.shape[1] - 1} + 1个目标")
    
    # 检查缺失值
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"存在缺失值: {missing}")
    else:
        print(f"   [OK] 无缺失值")
    
    # 检查常数列
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        issues.append(f"常数列: {const_cols}")
    else:
        print(f"   [OK] 无常数列")
    
    # 检查无穷值
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"存在无穷值: {inf_count}")
    else:
        print(f"   [OK] 无无穷值")
    
    if issues:
        print("\n[ISSUES] 发现问题:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("\n[SUCCESS] 数据验证通过！可以开始训练。")
    return True


def generate_train_command(output_file: str, target_col: str):
    """生成训练命令"""
    print("\n" + "=" * 70)
    print("[训练] 训练命令")
    print("=" * 70)
    print(f"""
# 快速测试
python ultra_lsnt_timeseries.py --data {output_file} --target {target_col} --quick

# 完整训练
python ultra_lsnt_timeseries.py --data {output_file} --target {target_col} --epochs 100

# 自定义配置
python ultra_lsnt_timeseries.py --data {output_file} --target {target_col} \\
    --seq_len 96 --pred_len 24 \\
    --hidden_dim 128 --num_blocks 3 --num_experts 4
""")


def main():
    parser = argparse.ArgumentParser(description='Ultra-LSNT 数据预处理')
    
    # 输入
    parser.add_argument('--data', type=str, help='输入数据文件路径')
    parser.add_argument('--sheet', type=str, default=None, help='Excel sheet 名称')
    
    # 输出
    parser.add_argument('--output', type=str, default='processed_data.csv', help='输出文件路径')
    parser.add_argument('--target', type=str, default=None, help='目标列名')
    
    # 模式
    parser.add_argument('--info', action='store_true', help='只显示数据信息')
    parser.add_argument('--type', type=str, choices=['wind', 'load', 'gefcom', 'auto'], 
                        default='auto', help='数据类型')
    
    # 合并多文件
    parser.add_argument('--merge_dir', type=str, help='合并目录下所有文件')
    
    args = parser.parse_args()
    
    # 显示信息模式
    if args.info and args.data:
        print_data_info(args.data)
        return
    
    # 合并模式
    if args.merge_dir:
        merge_city_data(args.merge_dir, args.output)
        return
    
    # 预处理模式
    if args.data:
        # 根据类型选择处理方法
        if args.type == 'wind':
            df = preprocess_wind_power_data(args.data, args.target, args.sheet)
        elif args.type == 'load':
            df = preprocess_load_data(args.data, args.target, args.sheet)
        elif args.type == 'gefcom':
            df = preprocess_gefcom_data(args.data)
        else:  # auto
            df = auto_detect_and_process(args.data, args.target)
        
        # 验证数据
        is_valid = validate_data(df)
        
        # 保存处理后的数据（即使验证失败也保存）
        df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] 数据已保存到: {args.output}")
        
        if is_valid:
            # 生成训练命令
            target = args.target or df.columns[-1]
            generate_train_command(args.output, target)
        else:
            print(f"\n[WARNING] 数据验证发现问题，建议检查数据质量后再进行训练")
    
    else:
        print("请指定数据文件: --data your_data.xlsx")
        print("\n使用示例:")
        print("  # 查看数据信息")
        print("  python data_preprocess.py --info --data 风电机组运行数据集/data.xlsx")
        print()
        print("  # 预处理风电数据")
        print("  python data_preprocess.py --data 风电数据.xlsx --type wind --output wind.csv")
        print()
        print("  # 预处理负荷数据")
        print("  python data_preprocess.py --data 负荷数据.xlsx --type load --output load.csv")
        print()
        print("  # 自动检测并处理")
        print("  python data_preprocess.py --data 时序论文负荷数据.xlsx --output processed.csv")


if __name__ == '__main__':
    main()
