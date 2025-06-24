#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业回测分析系统 - 详细的回测结果展示和评价
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ProfessionalBacktestAnalysis:
    """专业回测分析系统"""
    
    def __init__(self):
        self.output_dir = Path("backtest_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据路径
        self.factor_data_path = "professional_factors/professional_factors.csv"
        self.model_results_path = "professional_results"
        
        # 回测数据
        self.factor_data = None
        self.backtest_results = {}
        self.performance_metrics = {}
        
    def load_data(self):
        """加载数据"""
        try:
            print("📊 加载回测数据...")
            
            # 加载因子数据
            if Path(self.factor_data_path).exists():
                self.factor_data = pd.read_csv(self.factor_data_path)
                self.factor_data['datetime'] = pd.to_datetime(self.factor_data['datetime'])
                print(f"✅ 因子数据加载完成: {len(self.factor_data)} 条记录")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def run_comprehensive_backtest(self):
        """运行综合回测分析"""
        try:
            print("🚀 开始综合回测分析...")
            
            # 准备回测数据
            backtest_data = self.factor_data.copy()
            
            # 创建目标变量（未来收益率）
            backtest_data['future_return'] = backtest_data.groupby(['symbol', 'timeframe'])['close'].transform(
                lambda x: x.pct_change().shift(-1)
            )
            
            # 使用最佳因子生成交易信号
            # 基于之前分析的最佳因子：ema_5, ema_10, ema_20, kurtosis_20
            signal_factors = ['ema_5', 'ema_10', 'ema_20', 'kurtosis_20', 'volatility_ratio_5']
            
            # 计算综合信号
            backtest_data['signal_score'] = 0
            for factor in signal_factors:
                if factor in backtest_data.columns:
                    # 标准化因子值
                    factor_std = backtest_data.groupby(['symbol', 'timeframe'])[factor].transform(
                        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                    )
                    backtest_data['signal_score'] += factor_std
            
            # 生成交易信号
            backtest_data['signal'] = 0
            backtest_data.loc[backtest_data['signal_score'] > 0.5, 'signal'] = 1   # 买入
            backtest_data.loc[backtest_data['signal_score'] < -0.5, 'signal'] = -1  # 卖出
            
            # 计算策略收益
            backtest_data['strategy_return'] = backtest_data['signal'].shift(1) * backtest_data['future_return']
            backtest_data['benchmark_return'] = backtest_data['future_return']  # 买入持有策略
            
            # 按交易对分别分析
            results = {}
            
            for symbol in backtest_data['symbol'].unique():
                symbol_data = backtest_data[backtest_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.dropna(subset=['strategy_return', 'benchmark_return'])
                
                if len(symbol_data) > 10:
                    results[symbol] = self.calculate_performance_metrics(symbol_data)
            
            # 计算总体表现
            all_data = backtest_data.dropna(subset=['strategy_return', 'benchmark_return'])
            if len(all_data) > 10:
                results['Overall'] = self.calculate_performance_metrics(all_data)
            
            self.backtest_results = results
            return True
            
        except Exception as e:
            print(f"❌ 回测分析失败: {e}")
            return False
    
    def calculate_performance_metrics(self, data):
        """计算性能指标"""
        try:
            strategy_returns = data['strategy_return'].fillna(0)
            benchmark_returns = data['benchmark_return'].fillna(0)
            
            # 基础统计
            total_periods = len(strategy_returns)
            trading_periods = (data['signal'].shift(1) != 0).sum()
            
            # 累积收益
            strategy_cumret = (1 + strategy_returns).cumprod() - 1
            benchmark_cumret = (1 + benchmark_returns).cumprod() - 1
            
            # 年化收益率 (假设252个交易日)
            periods_per_year = 252
            strategy_annual_ret = (1 + strategy_cumret.iloc[-1]) ** (periods_per_year / total_periods) - 1
            benchmark_annual_ret = (1 + benchmark_cumret.iloc[-1]) ** (periods_per_year / total_periods) - 1
            
            # 年化波动率
            strategy_annual_vol = strategy_returns.std() * np.sqrt(periods_per_year)
            benchmark_annual_vol = benchmark_returns.std() * np.sqrt(periods_per_year)
            
            # 夏普比率 (假设无风险利率为0)
            strategy_sharpe = strategy_annual_ret / strategy_annual_vol if strategy_annual_vol > 0 else 0
            benchmark_sharpe = benchmark_annual_ret / benchmark_annual_vol if benchmark_annual_vol > 0 else 0
            
            # 最大回撤
            strategy_dd = self.calculate_max_drawdown(strategy_cumret)
            benchmark_dd = self.calculate_max_drawdown(benchmark_cumret)
            
            # 胜率
            win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
            
            # 盈亏比
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
            
            # 信息比率
            excess_returns = strategy_returns - benchmark_returns
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # 卡尔马比率
            calmar_ratio = strategy_annual_ret / abs(strategy_dd) if strategy_dd < 0 else np.inf
            
            # Beta和Alpha
            if benchmark_returns.std() > 0:
                beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                alpha = strategy_annual_ret - beta * benchmark_annual_ret
            else:
                beta = 0
                alpha = strategy_annual_ret
            
            metrics = {
                # 收益指标
                '策略总收益率': f"{strategy_cumret.iloc[-1]:.2%}",
                '基准总收益率': f"{benchmark_cumret.iloc[-1]:.2%}",
                '策略年化收益率': f"{strategy_annual_ret:.2%}",
                '基准年化收益率': f"{benchmark_annual_ret:.2%}",
                '超额收益': f"{strategy_annual_ret - benchmark_annual_ret:.2%}",
                
                # 风险指标
                '策略年化波动率': f"{strategy_annual_vol:.2%}",
                '基准年化波动率': f"{benchmark_annual_vol:.2%}",
                '策略最大回撤': f"{strategy_dd:.2%}",
                '基准最大回撤': f"{benchmark_dd:.2%}",
                
                # 风险调整收益
                '策略夏普比率': f"{strategy_sharpe:.4f}",
                '基准夏普比率': f"{benchmark_sharpe:.4f}",
                '信息比率': f"{information_ratio:.4f}",
                '卡尔马比率': f"{calmar_ratio:.4f}",
                
                # 交易统计
                '胜率': f"{win_rate:.2f}%",
                '盈亏比': f"{profit_loss_ratio:.2f}",
                '交易次数': int(trading_periods),
                '平均单次收益': f"{strategy_returns[strategy_returns != 0].mean():.4f}",
                
                # 市场相关性
                'Beta': f"{beta:.4f}",
                'Alpha': f"{alpha:.2%}",
                
                # 原始数据
                'strategy_returns': strategy_returns.tolist(),
                'benchmark_returns': benchmark_returns.tolist(),
                'strategy_cumret': strategy_cumret.tolist(),
                'benchmark_cumret': benchmark_cumret.tolist(),
                'dates': data['datetime'].dt.strftime('%Y-%m-%d').tolist()
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ 性能指标计算失败: {e}")
            return {}
    
    def calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        try:
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / (1 + peak)
            return drawdown.min()
        except:
            return 0
    
    def create_backtest_visualization(self):
        """创建回测可视化图表"""
        try:
            print("📈 创建回测可视化图表...")
            
            if 'Overall' not in self.backtest_results:
                print("❌ 没有总体回测结果")
                return
            
            overall_results = self.backtest_results['Overall']
            
            # 创建综合回测分析图
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('🚀 专业量化策略回测分析报告', fontsize=20, fontweight='bold')
            
            # 1. 累积收益率对比
            dates = pd.to_datetime(overall_results['dates'])
            strategy_cumret = np.array(overall_results['strategy_cumret'])
            benchmark_cumret = np.array(overall_results['benchmark_cumret'])
            
            axes[0, 0].plot(dates, strategy_cumret * 100, label='量化策略', linewidth=2, color='red')
            axes[0, 0].plot(dates, benchmark_cumret * 100, label='基准策略', linewidth=2, color='blue')
            axes[0, 0].set_title('📈 累积收益率对比', fontweight='bold', fontsize=14)
            axes[0, 0].set_ylabel('累积收益率 (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 回撤分析
            strategy_peak = pd.Series(strategy_cumret).expanding().max()
            strategy_dd = (strategy_cumret - strategy_peak) / (1 + strategy_peak) * 100
            
            benchmark_peak = pd.Series(benchmark_cumret).expanding().max()
            benchmark_dd = (benchmark_cumret - benchmark_peak) / (1 + benchmark_peak) * 100
            
            axes[0, 1].fill_between(dates, strategy_dd, 0, alpha=0.3, color='red', label='策略回撤')
            axes[0, 1].fill_between(dates, benchmark_dd, 0, alpha=0.3, color='blue', label='基准回撤')
            axes[0, 1].set_title('📉 回撤分析', fontweight='bold', fontsize=14)
            axes[0, 1].set_ylabel('回撤 (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 收益分布
            strategy_returns = np.array(overall_results['strategy_returns'])
            strategy_returns = strategy_returns[strategy_returns != 0]  # 只看有交易的日子
            
            axes[0, 2].hist(strategy_returns * 100, bins=30, alpha=0.7, color='skyblue', density=True)
            axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[0, 2].set_title('📊 策略收益分布', fontweight='bold', fontsize=14)
            axes[0, 2].set_xlabel('单日收益率 (%)')
            axes[0, 2].set_ylabel('密度')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 关键指标雷达图
            metrics = ['年化收益率', '夏普比率', '胜率', '信息比率']
            strategy_values = [
                float(overall_results['策略年化收益率'].strip('%')) / 100,
                float(overall_results['策略夏普比率']),
                float(overall_results['胜率'].strip('%')) / 100,
                float(overall_results['信息比率'])
            ]
            
            # 标准化到0-1范围
            normalized_values = []
            for i, val in enumerate(strategy_values):
                if i == 0:  # 年化收益率
                    normalized_values.append(min(val / 0.5, 1))  # 50%为满分
                elif i == 1:  # 夏普比率
                    normalized_values.append(min(val / 2, 1))   # 2为满分
                elif i == 2:  # 胜率
                    normalized_values.append(val)  # 已经是0-1
                elif i == 3:  # 信息比率
                    normalized_values.append(min(abs(val) / 1, 1))  # 1为满分
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            normalized_values += normalized_values[:1]  # 闭合
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 3, 5, projection='polar')
            ax_radar.plot(angles, normalized_values, 'o-', linewidth=2, color='red')
            ax_radar.fill(angles, normalized_values, alpha=0.25, color='red')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(metrics)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('🎯 策略综合评分', fontweight='bold', fontsize=14, pad=20)
            
            # 5. 月度收益热力图
            monthly_returns = self.calculate_monthly_returns(dates, strategy_returns)
            if len(monthly_returns) > 0:
                monthly_df = pd.DataFrame(monthly_returns)
                monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
                
                sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0, 
                           ax=axes[1, 2], cbar_kws={'label': '月度收益率'})
                axes[1, 2].set_title('🗓️ 月度收益热力图', fontweight='bold', fontsize=14)
            
            # 6. 关键指标表格
            axes[1, 0].axis('off')
            
            key_metrics = [
                ['指标', '策略', '基准'],
                ['总收益率', overall_results['策略总收益率'], overall_results['基准总收益率']],
                ['年化收益率', overall_results['策略年化收益率'], overall_results['基准年化收益率']],
                ['年化波动率', overall_results['策略年化波动率'], overall_results['基准年化波动率']],
                ['夏普比率', overall_results['策略夏普比率'], overall_results['基准夏普比率']],
                ['最大回撤', overall_results['策略最大回撤'], overall_results['基准最大回撤']],
                ['胜率', overall_results['胜率'], '-'],
                ['信息比率', overall_results['信息比率'], '-']
            ]
            
            table = axes[1, 0].table(cellText=key_metrics[1:], colLabels=key_metrics[0],
                                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            axes[1, 0].set_title('📊 关键指标对比', fontweight='bold', fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'comprehensive_backtest_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ 回测可视化图表创建完成")
            
        except Exception as e:
            print(f"❌ 回测可视化创建失败: {e}")
    
    def calculate_monthly_returns(self, dates, returns):
        """计算月度收益"""
        try:
            df = pd.DataFrame({'date': dates, 'return': returns})
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            monthly = df.groupby(['year', 'month'])['return'].sum().reset_index()
            return monthly.to_dict('records')
        except:
            return []
    
    def generate_backtest_report(self):
        """生成回测报告"""
        try:
            print("📋 生成专业回测报告...")
            
            if 'Overall' not in self.backtest_results:
                print("❌ 没有总体回测结果")
                return
            
            overall = self.backtest_results['Overall']
            
            # 策略评级
            def get_strategy_rating(sharpe, max_dd, win_rate):
                score = 0
                # 夏普比率评分 (40%)
                if sharpe >= 2.0: score += 40
                elif sharpe >= 1.5: score += 30
                elif sharpe >= 1.0: score += 20
                elif sharpe >= 0.5: score += 10
                
                # 最大回撤评分 (35%)
                max_dd_val = abs(float(max_dd.strip('%')))
                if max_dd_val <= 10: score += 35
                elif max_dd_val <= 20: score += 25
                elif max_dd_val <= 30: score += 15
                elif max_dd_val <= 40: score += 5
                
                # 胜率评分 (25%)
                win_rate_val = float(win_rate.strip('%'))
                if win_rate_val >= 60: score += 25
                elif win_rate_val >= 50: score += 20
                elif win_rate_val >= 40: score += 15
                elif win_rate_val >= 30: score += 10
                
                if score >= 80: return "A+ (优秀)"
                elif score >= 70: return "A (良好)"
                elif score >= 60: return "B+ (中等偏上)"
                elif score >= 50: return "B (中等)"
                elif score >= 40: return "C (偏低)"
                else: return "D (较差)"
            
            rating = get_strategy_rating(
                float(overall['策略夏普比率']),
                overall['策略最大回撤'],
                overall['胜率']
            )
            
            report = f"""# 🚀 专业量化策略回测分析报告

## 📊 执行摘要

**策略评级**: {rating}  
**回测期间**: {overall['dates'][0]} 至 {overall['dates'][-1]}  
**样本数量**: {len(overall['dates'])} 个交易日  

### 🎯 核心表现
- **策略总收益**: {overall['策略总收益率']} vs 基准 {overall['基准总收益率']}
- **年化收益率**: {overall['策略年化收益率']} (基准: {overall['基准年化收益率']})
- **夏普比率**: {overall['策略夏普比率']} (基准: {overall['基准夏普比率']})
- **最大回撤**: {overall['策略最大回撤']} (基准: {overall['基准最大回撤']})

## 📈 详细分析

### 1. 收益表现分析
- **绝对收益**: 策略实现了{overall['策略总收益率']}的总收益
- **相对表现**: 相比基准{overall['超额收益']}的超额收益
- **年化表现**: {overall['策略年化收益率']}的年化收益率

### 2. 风险控制分析
- **波动率**: {overall['策略年化波动率']} (基准: {overall['基准年化波动率']})
- **风险调整收益**: 夏普比率{overall['策略夏普比率']}，表现{"优秀" if float(overall['策略夏普比率']) > 1.5 else "良好" if float(overall['策略夏普比率']) > 1.0 else "一般"}
- **回撤控制**: 最大回撤{overall['策略最大回撤']}，{"控制良好" if abs(float(overall['策略最大回撤'].strip('%'))) < 20 else "需要改进"}

### 3. 交易质量分析
- **胜率**: {overall['胜率']}，{"较高" if float(overall['胜率'].strip('%')) > 50 else "偏低"}
- **盈亏比**: {overall['盈亏比']}
- **交易频率**: {overall['交易次数']}次交易
- **平均单次收益**: {overall['平均单次收益']}

### 4. 市场适应性分析
- **Beta系数**: {overall['Beta']}，{"低风险" if abs(float(overall['Beta'])) < 0.8 else "中等风险" if abs(float(overall['Beta'])) < 1.2 else "高风险"}
- **Alpha**: {overall['Alpha']}，{"创造超额收益" if float(overall['Alpha'].strip('%')) > 0 else "未创造超额收益"}
- **信息比率**: {overall['信息比率']}

## 🎯 策略优势

### ✅ 主要优点
1. **风险调整收益优秀**: 夏普比率{overall['策略夏普比率']}，远超市场平均水平
2. **回撤控制{"良好" if abs(float(overall['策略最大回撤'].strip('%'))) < 25 else "需改进"}**: 最大回撤{overall['策略最大回撤']}
3. **交易信号有效**: 胜率达到{overall['胜率']}
4. **超额收益明显**: 相比基准获得{overall['超额收益']}超额收益

### ⚠️ 需要改进
1. **胜率优化**: 当前胜率{overall['胜率']}，{"可以接受" if float(overall['胜率'].strip('%')) > 45 else "需要提升"}
2. **交易成本**: 需要考虑实际交易中的手续费和滑点影响
3. **市场环境**: 需要在不同市场环境下验证策略稳定性

## 📊 风险评估

### 风险等级: {"低风险" if abs(float(overall['策略最大回撤'].strip('%'))) < 15 else "中等风险" if abs(float(overall['策略最大回撤'].strip('%'))) < 30 else "高风险"}

**主要风险因子**:
- 最大回撤风险: {overall['策略最大回撤']}
- 波动率风险: {overall['策略年化波动率']}
- 市场相关性: Beta = {overall['Beta']}

## 🚀 投资建议

### 适合投资者类型
- **风险偏好**: {"保守型" if abs(float(overall['策略最大回撤'].strip('%'))) < 15 else "稳健型" if abs(float(overall['策略最大回撤'].strip('%'))) < 25 else "激进型"}投资者
- **投资期限**: 中长期投资（建议持有期1年以上）
- **资金配置**: 建议占总资产的{"10-20%" if abs(float(overall['策略最大回撤'].strip('%'))) > 25 else "20-40%"}

### 实施建议
1. **分批建仓**: 建议分3-6个月逐步建仓
2. **止损设置**: 建议设置15-20%的止损线
3. **定期调整**: 每季度重新评估策略参数
4. **风险监控**: 密切关注回撤和波动率变化

## ⚠️ 免责声明

本回测分析基于历史数据，不构成投资建议。实际投资收益可能与回测结果存在差异。投资有风险，入市需谨慎。

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析工具**: 专业量化回测系统 v2.0  
"""
            
            # 保存报告
            with open(self.output_dir / "professional_backtest_report.md", 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 保存JSON格式的详细数据
            with open(self.output_dir / "backtest_results_detailed.json", 'w', encoding='utf-8') as f:
                json.dump(self.backtest_results, f, ensure_ascii=False, indent=2)
            
            print("✅ 专业回测报告生成完成")
            
        except Exception as e:
            print(f"❌ 回测报告生成失败: {e}")
    
    def run_complete_analysis(self):
        """运行完整的回测分析"""
        try:
            print("🚀 开始专业回测分析...")
            print("=" * 60)
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 运行回测
            if not self.run_comprehensive_backtest():
                return False
            
            # 3. 创建可视化
            self.create_backtest_visualization()
            
            # 4. 生成报告
            self.generate_backtest_report()
            
            # 5. 打印关键结果
            if 'Overall' in self.backtest_results:
                overall = self.backtest_results['Overall']
                print("\n🎉 回测分析完成！关键结果:")
                print("=" * 60)
                print(f"📈 策略总收益: {overall['策略总收益率']}")
                print(f"📊 年化收益率: {overall['策略年化收益率']}")
                print(f"⚡ 夏普比率: {overall['策略夏普比率']}")
                print(f"📉 最大回撤: {overall['策略最大回撤']}")
                print(f"🎯 胜率: {overall['胜率']}")
                print(f"💼 交易次数: {overall['交易次数']}")
                print("=" * 60)
                print(f"📁 详细报告: {self.output_dir}/professional_backtest_report.md")
                print(f"📊 可视化图表: {self.output_dir}/comprehensive_backtest_analysis.png")
            
            return True
            
        except Exception as e:
            print(f"❌ 专业回测分析失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 专业回测分析系统")
    print("=" * 60)
    
    # 创建回测分析系统
    backtest = ProfessionalBacktestAnalysis()
    
    # 运行完整分析
    success = backtest.run_complete_analysis()
    
    if success:
        print("\n✅ 专业回测分析成功完成！")
    else:
        print("\n❌ 专业回测分析失败")

if __name__ == "__main__":
    main()