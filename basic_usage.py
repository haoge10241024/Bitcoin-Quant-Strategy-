#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例 - BTC Qlib 量化策略
Basic Usage Example - BTC Qlib Quantitative Strategy

这个示例展示了如何使用BTC Qlib策略的基本功能
This example demonstrates the basic functionality of the BTC Qlib strategy
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.extended_okx_data_collector import ExtendedOKXDataCollector
from factor_engineering.professional_factor_engineering import ProfessionalFactorEngineering
from modeling.professional_modeling_system import ProfessionalModelingSystem
from backtest.professional_backtest_analysis import ProfessionalBacktestAnalysis


def basic_workflow_example():
    """
    基础工作流程示例
    Basic workflow example
    """
    print("🚀 BTC Qlib 策略基础使用示例")
    print("=" * 50)
    
    # 1. 数据收集示例
    print("\n📊 步骤1: 数据收集")
    print("-" * 30)
    
    # 注意：这里只是示例，实际使用时需要配置API密钥
    # Note: This is just an example, you need to configure API keys for actual use
    try:
        data_collector = ExtendedOKXDataCollector()
        
        # 收集最近7天的数据作为示例
        # Collect recent 7 days data as example
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"收集数据时间范围: {start_date.date()} 到 {end_date.date()}")
        
        # 这里使用模拟数据，实际使用时会从API获取
        # Using simulated data here, actual use would fetch from API
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start=start_date, end=end_date, freq='1H'),
            'open': np.random.uniform(40000, 45000, 169),
            'high': np.random.uniform(40500, 45500, 169),
            'low': np.random.uniform(39500, 44500, 169),
            'close': np.random.uniform(40000, 45000, 169),
            'volume': np.random.uniform(100, 1000, 169)
        })
        
        print(f"✅ 数据收集完成，共 {len(sample_data)} 条记录")
        
    except Exception as e:
        print(f"❌ 数据收集失败: {e}")
        # 使用备用数据继续示例
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(40500, 45500, 100),
            'low': np.random.uniform(39500, 44500, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        print("✅ 使用模拟数据继续示例")
    
    # 2. 因子工程示例
    print("\n🔬 步骤2: 因子工程")
    print("-" * 30)
    
    try:
        factor_engineer = ProfessionalFactorEngineering()
        
        # 生成基础技术因子
        factors = factor_engineer.generate_basic_factors(sample_data)
        print(f"✅ 因子生成完成，共 {factors.shape[1]} 个因子")
        print(f"因子样本数: {len(factors)}")
        
        # 显示前几个因子名称
        factor_names = factors.columns.tolist()[:10]
        print(f"前10个因子: {factor_names}")
        
    except Exception as e:
        print(f"❌ 因子工程失败: {e}")
        return
    
    # 3. 模型训练示例
    print("\n🤖 步骤3: 模型训练")
    print("-" * 30)
    
    try:
        modeling_system = ProfessionalModelingSystem()
        
        # 准备目标变量（未来1日收益率）
        target = factors['close'].pct_change().shift(-1).fillna(0)
        target.name = 'future_return_1d'
        
        # 选择部分因子进行训练
        feature_cols = [col for col in factors.columns if col not in ['close', 'timestamp']]
        X = factors[feature_cols].fillna(0)
        y = target
        
        # 训练模型
        model_results = modeling_system.train_ensemble_model(X, y)
        print(f"✅ 模型训练完成")
        print(f"最佳模型: {model_results.get('best_model', 'N/A')}")
        print(f"训练得分: {model_results.get('train_score', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    # 4. 简单回测示例
    print("\n📈 步骤4: 简单回测")
    print("-" * 30)
    
    try:
        # 生成简单的交易信号
        predictions = model_results['model'].predict(X)
        signals = np.where(predictions > 0.001, 1, np.where(predictions < -0.001, -1, 0))
        
        # 计算策略收益
        strategy_returns = signals[:-1] * target[1:].values
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(24)  # 假设小时数据
        
        print(f"✅ 回测完成")
        print(f"总收益率: {total_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.4f}")
        print(f"交易次数: {np.sum(np.abs(np.diff(signals)))}")
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        return
    
    print("\n🎉 基础工作流程示例完成！")
    print("=" * 50)
    
    return {
        'data': sample_data,
        'factors': factors,
        'model_results': model_results,
        'backtest_results': {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'signals': signals
        }
    }


def data_analysis_example():
    """
    数据分析示例
    Data analysis example
    """
    print("\n📊 数据分析示例")
    print("=" * 30)
    
    # 生成示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    price_data = pd.DataFrame({
        'date': dates,
        'close': 40000 + np.cumsum(np.random.randn(100) * 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # 基础统计分析
    print("基础统计信息:")
    print(f"价格均值: ${price_data['close'].mean():.2f}")
    print(f"价格标准差: ${price_data['close'].std():.2f}")
    print(f"最高价: ${price_data['close'].max():.2f}")
    print(f"最低价: ${price_data['close'].min():.2f}")
    
    # 收益率分析
    returns = price_data['close'].pct_change().dropna()
    print(f"\n收益率分析:")
    print(f"平均日收益率: {returns.mean():.4f}")
    print(f"收益率波动率: {returns.std():.4f}")
    print(f"最大单日涨幅: {returns.max():.4f}")
    print(f"最大单日跌幅: {returns.min():.4f}")
    
    return price_data


def factor_analysis_example():
    """
    因子分析示例
    Factor analysis example
    """
    print("\n🔬 因子分析示例")
    print("=" * 30)
    
    # 生成示例因子数据
    n_samples = 100
    n_factors = 10
    
    np.random.seed(42)
    factors = pd.DataFrame(
        np.random.randn(n_samples, n_factors),
        columns=[f'factor_{i+1}' for i in range(n_factors)]
    )
    
    # 生成目标变量（与某些因子相关）
    target = (0.3 * factors['factor_1'] + 
              0.2 * factors['factor_2'] + 
              0.1 * factors['factor_3'] + 
              np.random.randn(n_samples) * 0.1)
    
    # 计算因子与目标的相关性
    correlations = factors.corrwith(target).sort_values(key=abs, ascending=False)
    
    print("因子相关性分析:")
    for factor, corr in correlations.head(5).items():
        print(f"{factor}: {corr:.4f}")
    
    # 因子稳定性分析
    print(f"\n因子稳定性分析:")
    for factor in factors.columns[:5]:
        stability = factors[factor].rolling(20).std().mean()
        print(f"{factor} 滚动标准差: {stability:.4f}")
    
    return factors, target, correlations


if __name__ == "__main__":
    """
    主函数 - 运行所有示例
    Main function - run all examples
    """
    print("🎯 BTC Qlib 策略 - 基础使用示例集合")
    print("=" * 60)
    
    try:
        # 运行基础工作流程示例
        workflow_results = basic_workflow_example()
        
        # 运行数据分析示例
        price_data = data_analysis_example()
        
        # 运行因子分析示例
        factors, target, correlations = factor_analysis_example()
        
        print("\n✅ 所有示例运行完成！")
        print("\n📚 更多高级功能请参考:")
        print("- advanced_features.py: 高级功能示例")
        print("- custom_strategy.py: 自定义策略示例")
        print("- 项目文档: docs/ 目录")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        print("请检查依赖包是否正确安装")
        
    print("\n" + "=" * 60) 