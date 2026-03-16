# 数据集说明

## 数据集概览

本项目使用4个公开数据集进行模型训练和评估：

| 数据集 | 任务类型 | 目标变量 | 时间粒度 | 样本规模 |
|--------|----------|----------|----------|----------|
| Wind (CN) | 风电功率预测 | 风电功率 | 15分钟 | ~52,000 |
| Wind (US) | 风电功率预测 | 风电功率 | 5分钟 | ~79,000 |
| Air Quality | 空气质量预测 | AQI指数 | 小时 | ~30M |
| GEFCom Load | 电力负荷预测 | 电力负荷 | 小时 | ~10M |

---

## 1. 中国风电数据 (Wind CN)

### 基本信息
- **文件名**: `wind_final.csv`
- **数据来源**: 新疆某风电场
- **时间范围**: 2019年1月 - 2020年12月
- **时间分辨率**: 15分钟
- **总样本数**: ~52,000

### 特征说明

| 特征名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| timestamp | datetime | 时间戳 | - |
| power | float | 风电功率输出 | kW |
| windspeed | float | 风速 | m/s |
| winddirection | float | 风向 | 度 |
| temperature | float | 温度 | °C |
| humidity | float | 湿度 | % |
| pressure | float | 气压 | hPa |

### 数据特点
- 风电功率范围: 0 - 3000 kW
- 有明显的季节性和日周期性
- 包含大量零功率时段（无风或维护）
- 功率与风速呈立方关系

---

## 2. 美国风电数据 (Wind US)

### 基本信息
- **文件名**: `wind_us.csv`
- **数据来源**: NREL WIND Toolkit
- **时间范围**: 2012年全年
- **时间分辨率**: 5分钟
- **总样本数**: ~79,000

### 特征说明

| 特征名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| timestamp | datetime | 时间戳 | - |
| power (MW) | float | 风电功率输出 | MW |
| wind_speed | float | 风速 | m/s |
| wind_direction | float | 风向 | 度 |
| temperature | float | 温度 | °C |
| air_density | float | 空气密度 | kg/m³ |

### 数据特点
- 风电功率范围: 0 - 150 MW
- 时间分辨率更高（5分钟）
- 包含空气密度特征（影响功率计算）

---

## 3. 空气质量数据 (Air Quality)

### 基本信息
- **文件名**: `air_quality_ready.csv`
- **数据来源**: 中国环境监测总站
- **时间范围**: 多城市多年数据
- **时间分辨率**: 小时
- **总样本数**: ~30,000,000

### 特征说明

| 特征名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| timestamp | datetime | 时间戳 | - |
| city | string | 城市名称 | - |
| AQI | float | 空气质量指数 | - |
| PM2.5 | float | PM2.5浓度 | μg/m³ |
| PM10 | float | PM10浓度 | μg/m³ |
| SO2 | float | 二氧化硫浓度 | μg/m³ |
| NO2 | float | 二氧化氮浓度 | μg/m³ |
| CO | float | 一氧化碳浓度 | mg/m³ |
| O3 | float | 臭氧浓度 | μg/m³ |
| temperature | float | 温度 | °C |
| humidity | float | 湿度 | % |
| wind_speed | float | 风速 | m/s |
| wind_direction | float | 风向 | 度 |

### 数据特点
- 覆盖中国主要城市
- AQI范围: 0 - 500+
- 有明显的季节性污染模式
- 受气象条件影响显著

---

## 4. GEFCom电力负荷数据 (GEFCom Load)

### 基本信息
- **文件名**: `gefcom_ready.csv`
- **数据来源**: GEFCom2012竞赛
- **时间范围**: 2004年 - 2008年
- **时间分辨率**: 小时
- **总样本数**: ~10,000,000

### 特征说明

| 特征名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| timestamp | datetime | 时间戳 | - |
| zone_id | int | 负荷区域ID | - |
| load | float | 电力负荷 | MW |
| temperature | float | 温度 | °C |
| day_of_week | int | 星期几 | 0-6 |
| hour | int | 小时 | 0-23 |
| is_holiday | bool | 是否节假日 | - |

### 数据特点
- 多区域负荷数据（20个区域）
- 有明显的日周期性和周周期性
- 温度敏感性强
- 节假日负荷模式不同

---

## 数据分割

所有数据集采用 **80/20** 时间序列分割：

```
训练集: 80% (前80%时间)
测试集: 20% (后20%时间)
```

**注意**: 时间序列数据不使用随机分割，以保持时间依赖性。

---

## 数据预处理

### 预处理流程

1. **缺失值处理**
   - 线性插值填充短时缺失
   - 删除长时缺失（>6小时）

2. **异常值检测**
   - 基于3-sigma原则
   - 物理约束检查（如功率非负）

3. **特征工程**
   - 时间特征：hour, day_of_week, month, is_weekend
   - 周期性编码：sin/cos变换
   - 滞后特征：过去24小时滞后

4. **归一化**
   - Z-score标准化
   - 保存scaler用于逆变换

### 预处理脚本

```bash
# 预处理风电数据
python src/data_preprocess.py \
    --data data/raw/wind_data.xlsx \
    --target power \
    --output data/processed/wind_final.csv

# 查看数据信息
python src/data_preprocess.py --info --data data/processed/wind_final.csv
```

---

## 专家激活数据

### 专家激活日志
- **文件名**: `wind_expert_activation.csv`, `air_quality_expert_activation.csv`
- **内容**: 记录每个样本的专家选择情况
- **用途**: 分析专家特化行为

### 数据格式

| 字段 | 说明 |
|------|------|
| timestamp | 时间戳 |
| expert_1 | 专家1激活概率 |
| expert_2 | 专家2激活概率 |
| expert_3 | 专家3激活概率 |
| expert_4 | 专家4激活概率 |
| selected_experts | 被选中的专家索引 |
| input_type | 输入类型（高/中/低风速） |

---

## 数据使用示例

```python
import pandas as pd

# 加载数据
df = pd.read_csv('data/processed/wind_final.csv', parse_dates=['timestamp'])

# 查看基本信息
print(df.head())
print(df.describe())

# 时间范围
print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

# 特征统计
print(f"功率范围: {df['power'].min():.2f} - {df['power'].max():.2f} kW")
print(f"风速范围: {df['windspeed'].min():.2f} - {df['windspeed'].max():.2f} m/s")
```

---

## 引用数据

如果使用本项目的数据，请引用原始数据源：

### Wind (CN)
```bibtex
@data{wind_cn_2019,
  title={新疆风电场运行数据},
  author={[Data Provider]},
  year={2019}
}
```

### Wind (US)
```bibtex
@dataset{nrel_wind_toolkit_2012,
  title={NREL WIND Toolkit},
  author={National Renewable Energy Laboratory},
  year={2012},
  url={https://www.nrel.gov/grid/wind-toolkit.html}
}
```

### Air Quality
```bibtex
@data{china_air_quality,
  title={中国环境监测总站空气质量数据},
  author={中国环境监测总站},
  url={http://www.cnemc.cn/}
}
```

### GEFCom Load
```bibtex
@inproceedings{hong2014global,
  title={Global energy forecasting competition 2012},
  author={Hong, Tao and Pinson, Pierre and Fan, Shu},
  booktitle={International Journal of Forecasting},
  year={2014}
}
```
