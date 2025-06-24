# API 参考文档
# API Reference Documentation

本文档提供了BTC Qlib量化策略项目的完整API参考。

## 目录
- [数据收集模块](#数据收集模块)
- [因子工程模块](#因子工程模块)
- [建模系统模块](#建模系统模块)
- [回测分析模块](#回测分析模块)
- [技术改进模块](#技术改进模块)
- [工具函数模块](#工具函数模块)

## 数据收集模块

### ExtendedOKXDataCollector

专业的OKX数据收集器，支持多币种、多时间周期的历史数据收集。

#### 初始化

```python
from src.data_collection.extended_okx_data_collector import ExtendedOKXDataCollector

collector = ExtendedOKXDataCollector(
    api_key=None,           # API密钥（可选）
    secret_key=None,        # 密钥（可选）
    passphrase=None,        # 口令（可选）
    sandbox=False           # 是否使用沙盒环境
)
```

#### 主要方法

##### collect_historical_data()

收集历史数据

```python
data = collector.collect_historical_data(
    symbol="BTC/USDT",      # 交易对
    timeframe="1d",         # 时间周期
    start_date="2023-01-01", # 开始日期
    end_date="2024-01-01",  # 结束日期
    limit=1000              # 数据条数限制
)
```

**参数:**
- `symbol` (str): 交易对符号，如"BTC/USDT"
- `timeframe` (str): 时间周期，支持"1m", "5m", "15m", "1h", "4h", "1d", "1w"
- `start_date` (str): 开始日期，格式"YYYY-MM-DD"
- `end_date` (str): 结束日期，格式"YYYY-MM-DD"
- `limit` (int): 单次请求的最大数据条数

**返回:**
- `pd.DataFrame`: 包含OHLCV数据的DataFrame

##### collect_realtime_data()

收集实时数据

```python
data = collector.collect_realtime_data(
    symbol="BTC/USDT",
    timeframe="1m",
    count=100
)
```

**参数:**
- `symbol` (str): 交易对符号
- `timeframe` (str): 时间周期
- `count` (int): 获取的数据条数

**返回:**
- `pd.DataFrame`: 最新的OHLCV数据

## 因子工程模块

### ProfessionalFactorEngineering

专业因子工程系统，支持174个专业金融因子的生成和分析。

#### 初始化

```python
from src.factor_engineering.professional_factor_engineering import ProfessionalFactorEngineering

factor_engineer = ProfessionalFactorEngineering()
```

#### 主要方法

##### generate_all_factors()

生成所有因子

```python
factors = factor_engineer.generate_all_factors(
    data,                   # 原始OHLCV数据
    include_advanced=True   # 是否包含高级因子
)
```

**参数:**
- `data` (pd.DataFrame): 包含OHLCV列的原始数据
- `include_advanced` (bool): 是否生成高级因子

**返回:**
- `pd.DataFrame`: 包含所有因子的DataFrame

##### generate_technical_indicators()

生成技术指标因子

```python
technical_factors = factor_engineer.generate_technical_indicators(data)
```

**参数:**
- `data` (pd.DataFrame): 原始OHLCV数据

**返回:**
- `pd.DataFrame`: 技术指标因子

##### generate_statistical_factors()

生成统计因子

```python
statistical_factors = factor_engineer.generate_statistical_factors(data)
```

**参数:**
- `data` (pd.DataFrame): 原始OHLCV数据

**返回:**
- `pd.DataFrame`: 统计因子

## 建模系统模块

### ProfessionalModelingSystem

专业建模系统，支持多种机器学习模型的训练和集成。

#### 初始化

```python
from src.modeling.professional_modeling_system import ProfessionalModelingSystem

modeling_system = ProfessionalModelingSystem(
    random_state=42,        # 随机种子
    n_jobs=-1              # 并行作业数
)
```

#### 主要方法

##### train_ensemble_model()

训练集成模型

```python
results = modeling_system.train_ensemble_model(
    X,                      # 特征数据
    y,                      # 目标变量
    models=['rf', 'gbm', 'ridge'],  # 使用的模型
    cv_folds=5             # 交叉验证折数
)
```

**参数:**
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series): 目标变量
- `models` (list): 要使用的模型列表
- `cv_folds` (int): 交叉验证折数

**返回:**
- `dict`: 包含模型、性能指标等的结果字典

##### optimize_hyperparameters()

优化超参数

```python
best_params = modeling_system.optimize_hyperparameters(
    X, y,
    model_type='random_forest',
    n_iter=50,
    cv_folds=5
)
```

**参数:**
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series): 目标变量
- `model_type` (str): 模型类型
- `n_iter` (int): 搜索迭代次数
- `cv_folds` (int): 交叉验证折数

**返回:**
- `dict`: 最佳超参数

## 回测分析模块

### ProfessionalBacktestAnalysis

专业回测分析系统，提供完整的策略回测和性能分析。

#### 初始化

```python
from src.backtest.professional_backtest_analysis import ProfessionalBacktestAnalysis

backtest_analyzer = ProfessionalBacktestAnalysis(
    initial_capital=100000,  # 初始资金
    commission=0.001,        # 手续费率
    slippage=0.0005         # 滑点
)
```

#### 主要方法

##### run_backtest()

运行回测

```python
results = backtest_analyzer.run_backtest(
    data,                   # 价格数据
    signals,                # 交易信号
    strategy_name="BTC_Strategy"
)
```

**参数:**
- `data` (pd.DataFrame): 包含价格数据的DataFrame
- `signals` (pd.Series): 交易信号序列
- `strategy_name` (str): 策略名称

**返回:**
- `dict`: 回测结果，包含收益、风险指标等

##### calculate_performance_metrics()

计算性能指标

```python
metrics = backtest_analyzer.calculate_performance_metrics(returns)
```

**参数:**
- `returns` (pd.Series): 收益率序列

**返回:**
- `dict`: 性能指标字典

## 技术改进模块

### CoreTechnicalImprovements

核心技术改进系统，提供数据质量验证、因子有效性分析等功能。

#### 初始化

```python
from src.improvements.core_technical_improvements import CoreTechnicalImprovements

improvement_system = CoreTechnicalImprovements()
```

#### 主要方法

##### run_comprehensive_improvements()

运行综合技术改进

```python
results = improvement_system.run_comprehensive_improvements(
    data_path="data/processed/bitcoin_data.csv",
    factor_path="data/factors/bitcoin_factors.csv"
)
```

**参数:**
- `data_path` (str): 数据文件路径
- `factor_path` (str): 因子文件路径

**返回:**
- `dict`: 改进结果，包含数据质量、因子分析等

##### validate_data_quality()

验证数据质量

```python
quality_results = improvement_system.validate_data_quality(data)
```

**参数:**
- `data` (pd.DataFrame): 待验证的数据

**返回:**
- `dict`: 数据质量验证结果

##### analyze_factor_effectiveness()

分析因子有效性

```python
factor_results = improvement_system.analyze_factor_effectiveness(
    factors, 
    target_returns
)
```

**参数:**
- `factors` (pd.DataFrame): 因子数据
- `target_returns` (pd.Series): 目标收益率

**返回:**
- `dict`: 因子有效性分析结果

## 工具函数模块

### DataValidator

数据验证工具类

#### 主要方法

##### validate_ohlc_logic()

验证OHLC数据逻辑

```python
from src.utils.data_utils import DataValidator

validation_result = DataValidator.validate_ohlc_logic(df)
```

**参数:**
- `df` (pd.DataFrame): 包含OHLC数据的DataFrame

**返回:**
- `dict`: 验证结果

##### detect_outliers()

检测异常值

```python
outliers = DataValidator.detect_outliers(
    series,
    method='zscore',
    threshold=3.0
)
```

**参数:**
- `series` (pd.Series): 数据序列
- `method` (str): 检测方法，'zscore'或'iqr'
- `threshold` (float): 阈值

**返回:**
- `pd.Series`: 异常值标记

### DataCleaner

数据清洗工具类

#### 主要方法

##### fill_missing_values()

填充缺失值

```python
from src.utils.data_utils import DataCleaner

cleaned_data = DataCleaner.fill_missing_values(
    df,
    method='forward'
)
```

**参数:**
- `df` (pd.DataFrame): 包含缺失值的数据
- `method` (str): 填充方法，'forward', 'backward', 'interpolate', 'mean', 'median'

**返回:**
- `pd.DataFrame`: 填充后的数据

##### remove_outliers()

移除异常值

```python
cleaned_data = DataCleaner.remove_outliers(
    df,
    columns=['close', 'volume'],
    method='zscore',
    threshold=3.0
)
```

**参数:**
- `df` (pd.DataFrame): 原始数据
- `columns` (list): 要处理的列名
- `method` (str): 检测方法
- `threshold` (float): 阈值

**返回:**
- `pd.DataFrame`: 移除异常值后的数据

### DataTransformer

数据转换工具类

#### 主要方法

##### calculate_returns()

计算收益率

```python
from src.utils.data_utils import DataTransformer

returns = DataTransformer.calculate_returns(
    prices,
    method='simple'
)
```

**参数:**
- `prices` (pd.Series): 价格序列
- `method` (str): 计算方法，'simple'或'log'

**返回:**
- `pd.Series`: 收益率序列

##### normalize_data()

标准化数据

```python
normalized_data = DataTransformer.normalize_data(
    df,
    columns=['close', 'volume'],
    method='zscore'
)
```

**参数:**
- `df` (pd.DataFrame): 原始数据
- `columns` (list): 要标准化的列
- `method` (str): 标准化方法，'zscore', 'minmax', 'robust'

**返回:**
- `pd.DataFrame`: 标准化后的数据

## 使用示例

### 完整工作流程示例

```python
# 1. 数据收集
from src.data_collection.extended_okx_data_collector import ExtendedOKXDataCollector

collector = ExtendedOKXDataCollector()
data = collector.collect_historical_data("BTC/USDT", "1d", "2023-01-01", "2024-01-01")

# 2. 因子工程
from src.factor_engineering.professional_factor_engineering import ProfessionalFactorEngineering

factor_engineer = ProfessionalFactorEngineering()
factors = factor_engineer.generate_all_factors(data)

# 3. 模型训练
from src.modeling.professional_modeling_system import ProfessionalModelingSystem

modeling_system = ProfessionalModelingSystem()
target = data['close'].pct_change().shift(-1).fillna(0)
results = modeling_system.train_ensemble_model(factors, target)

# 4. 回测分析
from src.backtest.professional_backtest_analysis import ProfessionalBacktestAnalysis

backtest_analyzer = ProfessionalBacktestAnalysis()
predictions = results['model'].predict(factors)
signals = np.where(predictions > 0.001, 1, np.where(predictions < -0.001, -1, 0))
backtest_results = backtest_analyzer.run_backtest(data, signals)

# 5. 技术改进验证
from src.improvements.core_technical_improvements import CoreTechnicalImprovements

improvement_system = CoreTechnicalImprovements()
improvement_results = improvement_system.run_comprehensive_improvements(
    "data/processed/bitcoin_data.csv",
    "data/factors/bitcoin_factors.csv"
)
```

## 错误处理

所有API方法都包含适当的错误处理机制：

```python
try:
    data = collector.collect_historical_data("BTC/USDT", "1d")
except Exception as e:
    print(f"数据收集失败: {e}")
```

## 配置选项

可以通过配置文件自定义系统行为：

```python
# 加载配置
import yaml

with open('config/data_config.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)
```

## 性能优化建议

1. **并行处理**: 大多数计算密集型操作支持并行处理
2. **内存管理**: 处理大数据集时注意内存使用
3. **缓存机制**: 重复计算的结果会被缓存
4. **批处理**: 大量数据建议分批处理

## 版本兼容性

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- qlib >= 0.8.0

## 更新日志

### v2.0.0 (2025-01-01)
- 添加核心技术改进系统
- 增强数据质量验证
- 改进因子有效性分析
- 优化模型性能

### v1.0.0 (2024-12-01)
- 初始版本发布
- 基础数据收集功能
- 因子工程系统
- 建模和回测框架 