# 🚀 BTC Quantitative Trading Strategy Based on Qlib

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Qlib](https://img.shields.io/badge/Qlib-Framework-green.svg)](https://github.com/microsoft/qlib)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于Microsoft Qlib框架的专业比特币量化交易策略，集成了完整的数据处理、因子工程、模型训练和回测分析系统。

## 🎯 项目特色

### 核心功能
- **📊 专业数据处理**: 完整的数据质量控制和异常值检测
- **🔬 科学因子工程**: 174个专业因子，基于IC测试的有效性验证
- **🤖 智能模型系统**: 多模型集成，系统化超参数优化
- **📈 严格回测验证**: Walk-Forward Analysis，消除前瞻偏差
- **⚙️ 技术风险控制**: 共线性处理，数据质量监控

### 技术亮点
- ✅ **数据质量验证**: OHLC逻辑检查、缺失值处理、异常值检测
- ✅ **因子有效性验证**: IC测试、滚动IC分析、因子筛选
- ✅ **前瞻偏差消除**: 严格的时间序列分割验证
- ✅ **多重共线性处理**: 自动识别和移除冗余因子
- ✅ **系统化超参数优化**: RandomizedSearchCV + TimeSeriesSplit

## 📦 安装和环境配置

### 环境要求
```bash
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
qlib >= 0.8.0
```

### 快速安装
```bash
# 克隆项目
git clone https://github.com/your-username/btc-qlib-strategy.git
cd btc-qlib-strategy

# 安装依赖
pip install -r requirements.txt

# 安装Qlib
pip install pyqlib
```

## 🚀 快速开始

### 1. 数据收集
```bash
# 收集比特币历史数据
python src/data_collection/extended_okx_data_collector.py
```

### 2. 因子工程
```bash
# 生成专业因子
python src/factor_engineering/professional_factor_engineering.py
```

### 3. 模型训练
```bash
# 训练量化模型
python src/modeling/professional_modeling_system.py
```

### 4. 策略回测
```bash
# 执行策略回测
python src/backtest/professional_backtest_analysis.py
```

### 5. 技术改进验证
```bash
# 运行核心技术改进系统
python src/improvements/core_technical_improvements.py
```

## 📁 项目结构

```
btc-qlib-strategy/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包列表
├── LICENSE                            # 开源协议
├── .gitignore                         # Git忽略文件
├── config/                            # 配置文件
│   ├── data_config.yaml              # 数据配置
│   ├── model_config.yaml             # 模型配置
│   └── strategy_config.yaml          # 策略配置
├── src/                              # 源代码
│   ├── data_collection/              # 数据收集
│   │   ├── extended_okx_data_collector.py
│   │   └── data_quality_validator.py
│   ├── factor_engineering/           # 因子工程
│   │   ├── professional_factor_engineering.py
│   │   └── factor_effectiveness_validator.py
│   ├── modeling/                     # 模型系统
│   │   ├── professional_modeling_system.py
│   │   └── ensemble_model_system.py
│   ├── backtest/                     # 回测分析
│   │   ├── professional_backtest_analysis.py
│   │   └── strategy_performance_analyzer.py
│   ├── improvements/                 # 技术改进
│   │   ├── core_technical_improvements.py
│   │   └── enhanced_model_system.py
│   └── utils/                        # 工具函数
│       ├── data_utils.py
│       ├── model_utils.py
│       └── visualization_utils.py
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后数据
│   └── factors/                      # 因子数据
├── models/                           # 训练好的模型
├── results/                          # 结果输出
│   ├── backtest/                     # 回测结果
│   ├── reports/                      # 分析报告
│   └── visualizations/              # 可视化图表
├── docs/                             # 文档
│   ├── technical_analysis.md         # 技术分析报告
│   ├── factor_analysis.md            # 因子分析报告
│   ├── model_performance.md          # 模型性能报告
│   └── api_reference.md              # API参考
├── tests/                            # 测试代码
│   ├── test_data_collection.py
│   ├── test_factor_engineering.py
│   └── test_modeling.py
└── examples/                         # 示例代码
    ├── basic_usage.py
    ├── advanced_features.py
    └── custom_strategy.py
```

## 📊 性能表现

### 技术改进成果
| 指标 | 改进前 | 改进后 | 提升幅度 |
|------|--------|--------|----------|
| 数据质量控制 | 无系统 | 完整体系 | 质的飞跃 |
| 因子有效率 | 未验证 | 23.88% | 新增能力 |
| 前瞻偏差风险 | 存在 | 已消除 | 关键改进 |
| 专业性评分 | 5.5/10 | 7.5/10 | +36.4% |

### 因子分析结果
- **总因子数量**: 174个专业因子
- **有效因子数量**: 16个（通过IC测试）
- **Top 5最佳因子**: max_drawdown_60, ma_5, return_20, momentum_20, reversal_20

### 模型验证结果
- **验证方法**: Walk-Forward Analysis
- **训练窗口**: 252个交易日
- **测试窗口**: 21个交易日
- **前瞻偏差**: 已完全消除

## 🔧 核心技术特性

### 1. 数据质量控制系统
```python
class DataQualityValidator:
    """专业数据质量验证器"""
    
    def comprehensive_quality_check(self, df):
        # OHLC逻辑检查
        # 缺失值检测
        # 异常值检测
        # 时间序列连续性检查
        return quality_issues
```

### 2. 因子有效性验证系统
```python
class FactorEffectivenessValidator:
    """因子有效性验证器"""
    
    def ic_analysis(self, factors, target_returns):
        # 信息系数计算
        # 滚动IC分析
        # 因子有效性筛选
        return ic_results
```

### 3. 前瞻偏差检查系统
```python
class WalkForwardValidator:
    """步进式验证器"""
    
    def validate_model_performance(self, X, y, models):
        # Walk-Forward Analysis
        # 严格时间序列分割
        # 无前瞻偏差验证
        return validation_results
```

## 📈 使用示例

### 基础使用
```python
from src.modeling import ProfessionalModelingSystem

# 初始化建模系统
modeling_system = ProfessionalModelingSystem()

# 加载数据和因子
data_path = "data/processed/bitcoin_data.csv"
factor_path = "data/factors/bitcoin_factors.csv"

# 运行完整建模流程
results = modeling_system.run_comprehensive_modeling(
    data_path=data_path,
    factor_path=factor_path
)

# 查看结果
print(f"最佳模型: {results['best_model']}")
print(f"模型性能: {results['performance_metrics']}")
```

### 高级功能
```python
from src.improvements import CoreTechnicalImprovements

# 运行技术改进系统
improvement_system = CoreTechnicalImprovements()
results = improvement_system.run_comprehensive_improvements(
    data_path="data/processed/bitcoin_data.csv",
    factor_path="data/factors/bitcoin_factors.csv"
)

# 查看改进效果
print(f"数据质量问题: {len(results['data_quality']['issues'])}")
print(f"有效因子数量: {results['factor_analysis']['effective_factors']}")
print(f"最佳模型性能: {results['validation_results']['best_performance']}")
```

## 📚 文档和教程

- [技术分析报告](docs/technical_analysis.md) - 详细的技术改进分析
- [因子工程指南](docs/factor_analysis.md) - 因子开发和验证方法
- [模型训练教程](docs/model_performance.md) - 模型训练和优化指南
- [API参考文档](docs/api_reference.md) - 完整的API文档


### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

## ⚠️ 风险提示

**重要声明**: 本项目仅供学习和研究使用，不构成投资建议。

- 量化交易存在风险，过往表现不代表未来收益
- 请在充分理解策略逻辑后谨慎使用
- 建议先在模拟环境中验证策略有效性
- 实盘交易请控制仓位，做好风险管理

## 📄 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Microsoft Qlib](https://github.com/microsoft/qlib) - 优秀的量化投资平台
- [OKX API](https://www.okx.com/docs-v5/) - 可靠的数据来源
- 所有为开源量化社区做出贡献的开发者们

## 📞 联系方式

- **邮箱**: 953534947@qq.com

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

**最后更新**: 2025年6月24日 
