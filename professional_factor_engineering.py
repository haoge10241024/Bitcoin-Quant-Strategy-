#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业因子工程系统 - 包含完整的预处理、筛选和分析方法
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 统计和机器学习
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ProfessionalFactorEngineering:
    """专业因子工程系统"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "data/extended/extended_all_data_3years_20250623_215123.csv"
        self.output_dir = Path("professional_factors")
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化数据
        self.raw_data = None
        self.factor_data = None
        self.processed_factors = None
        
        # 因子分析结果
        self.factor_analysis = {}
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / f"professional_factor_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """加载数据"""
        try:
            self.logger.info("加载原始数据...")
            self.raw_data = pd.read_csv(self.data_path)
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'])
            
            self.logger.info(f"数据加载完成: {len(self.raw_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return False
    
    def calculate_advanced_factors(self):
        """计算高级因子"""
        try:
            self.logger.info("计算高级因子...")
            
            processed_data = []
            
            for (symbol, timeframe), group in self.raw_data.groupby(['symbol', 'timeframe']):
                self.logger.info(f"处理 {symbol} {timeframe} 数据...")
                
                group = group.reset_index(drop=True).copy()
                
                # 基础价格因子
                group = self._calculate_price_factors(group)
                
                # 高级技术因子
                group = self._calculate_advanced_technical_factors(group)
                
                # 统计因子
                group = self._calculate_statistical_factors(group)
                
                # 微观结构因子
                group = self._calculate_microstructure_factors(group)
                
                # 时间序列因子
                group = self._calculate_time_series_factors(group)
                
                processed_data.append(group)
            
            self.factor_data = pd.concat(processed_data, ignore_index=True)
            self.logger.info(f"因子计算完成，共 {len(self.factor_data.columns)} 列")
            
            return True
            
        except Exception as e:
            self.logger.error(f"因子计算失败: {e}")
            return False
    
    def _calculate_price_factors(self, df):
        """计算价格相关因子"""
        # 基础收益率
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # 价格位置
        for period in [20, 60]:
            df[f'price_position_{period}'] = (df['close'] - df['close'].rolling(period).min()) / \
                                           (df['close'].rolling(period).max() - df['close'].rolling(period).min())
        
        # 相对强弱
        for period in [10, 20]:
            df[f'relative_strength_{period}'] = df['close'] / df['close'].rolling(period).mean()
        
        return df
    
    def _calculate_advanced_technical_factors(self, df):
        """计算高级技术因子"""
        # 移动平均系统
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']
        
        # MACD系统
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI系统
        for period in [6, 14, 24]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 布林带系统
        for period in [20]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = ma + 2 * std
            df[f'bb_lower_{period}'] = ma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                        (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # ATR系统
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        for period in [14, 21]:
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        return df
    
    def _calculate_statistical_factors(self, df):
        """计算统计因子"""
        # 波动率系统
        for period in [5, 10, 20, 60]:
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(period).mean()
        
        # 高阶矩
        for period in [20, 60]:
            returns = df['close'].pct_change()
            df[f'skewness_{period}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(period).kurt()
        
        # 风险指标
        for period in [20, 60]:
            returns = df['close'].pct_change()
            df[f'downside_deviation_{period}'] = returns[returns < 0].rolling(period).std()
            df[f'upside_deviation_{period}'] = returns[returns > 0].rolling(period).std()
            df[f'var_95_{period}'] = returns.rolling(period).quantile(0.05)
            df[f'cvar_95_{period}'] = returns[returns <= returns.rolling(period).quantile(0.05)].rolling(period).mean()
        
        return df
    
    def _calculate_microstructure_factors(self, df):
        """计算微观结构因子"""
        # 成交量因子
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
            df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
        
        # 价量关系
        for period in [10, 20]:
            df[f'price_volume_corr_{period}'] = df['close'].rolling(period).corr(df['volume'])
            df[f'return_volume_corr_{period}'] = df['close'].pct_change().rolling(period).corr(df['volume'])
        
        # 资金流向
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        for period in [14]:
            df[f'mfi_{period}'] = 100 - (100 / (1 + positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()))
        
        return df
    
    def _calculate_time_series_factors(self, df):
        """计算时间序列因子"""
        # 趋势因子
        for period in [5, 10, 20]:
            # 线性回归斜率
            def calc_slope(y):
                if len(y) < 2:
                    return np.nan
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            
            df[f'trend_slope_{period}'] = df['close'].rolling(period).apply(calc_slope, raw=False)
            df[f'trend_r2_{period}'] = df['close'].rolling(period).apply(
                lambda y: stats.linregress(np.arange(len(y)), y)[2]**2 if len(y) >= 2 else np.nan, 
                raw=False
            )
        
        # 动量和反转
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'reversal_{period}'] = -df[f'momentum_{period}']
            
            # 风险调整动量
            vol = df['close'].pct_change().rolling(period).std()
            df[f'momentum_vol_adj_{period}'] = df[f'momentum_{period}'] / vol
        
        return df
    
    def professional_preprocessing(self):
        """专业预处理"""
        try:
            self.logger.info("开始专业预处理...")
            
            # 获取因子列
            factor_columns = [col for col in self.factor_data.columns 
                            if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                         'open', 'high', 'low', 'close', 'volume']]
            
            processed_data = self.factor_data.copy()
            
            # 1. 异常值处理（去极值）
            self.logger.info("1. 异常值处理...")
            for col in factor_columns:
                if processed_data[col].dtype in ['float64', 'int64']:
                    # 使用3倍标准差方法
                    mean_val = processed_data[col].mean()
                    std_val = processed_data[col].std()
                    
                    # 去极值
                    processed_data[col] = processed_data[col].clip(
                        lower=mean_val - 3*std_val,
                        upper=mean_val + 3*std_val
                    )
            
            # 2. 标准化处理
            self.logger.info("2. 标准化处理...")
            scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
            
            # 按symbol和timeframe分组标准化
            standardized_data = []
            for (symbol, timeframe), group in processed_data.groupby(['symbol', 'timeframe']):
                group_copy = group.copy()
                
                # 只对因子列进行标准化
                factor_values = group_copy[factor_columns].values
                factor_values_clean = np.nan_to_num(factor_values, nan=0.0)
                
                if len(factor_values_clean) > 1:
                    factor_values_scaled = scaler.fit_transform(factor_values_clean)
                    group_copy[factor_columns] = factor_values_scaled
                
                standardized_data.append(group_copy)
            
            processed_data = pd.concat(standardized_data, ignore_index=True)
            
            # 3. 缺失值处理
            self.logger.info("3. 缺失值处理...")
            # 使用前向填充和后向填充
            processed_data[factor_columns] = processed_data[factor_columns].fillna(method='ffill').fillna(method='bfill')
            
            # 4. 中性化处理（行业中性化，这里用symbol代替）
            self.logger.info("4. 中性化处理...")
            for col in factor_columns:
                if processed_data[col].dtype in ['float64', 'int64']:
                    # 按symbol进行中性化
                    processed_data[f'{col}_neutral'] = processed_data.groupby('symbol')[col].transform(
                        lambda x: x - x.mean()
                    )
            
            self.processed_factors = processed_data
            self.logger.info("专业预处理完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"专业预处理失败: {e}")
            return False
    
    def professional_factor_selection(self, target_col: str = 'return_1', n_features: int = 20):
        """专业因子筛选"""
        try:
            self.logger.info("开始专业因子筛选...")
            
            # 准备数据
            factor_columns = [col for col in self.processed_factors.columns 
                            if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                         'open', 'high', 'low', 'close', 'volume']]
            
            # 创建目标变量
            self.processed_factors['target'] = self.processed_factors.groupby(['symbol', 'timeframe'])['close'].transform(
                lambda x: x.pct_change().shift(-1)
            )
            
            # 移除缺失值
            clean_data = self.processed_factors.dropna(subset=['target'] + factor_columns[:50])  # 限制因子数量避免内存问题
            
            X = clean_data[factor_columns[:50]]
            y = clean_data['target']
            
            selection_results = {}
            
            # 1. 相关性分析
            self.logger.info("1. 相关性分析...")
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selection_results['correlation'] = correlations.head(n_features).to_dict()
            
            # 2. 互信息分析
            self.logger.info("2. 互信息分析...")
            mi_scores = mutual_info_regression(X.fillna(0), y)
            mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            selection_results['mutual_info'] = mi_df.head(n_features).to_dict()
            
            # 3. F统计量分析
            self.logger.info("3. F统计量分析...")
            f_selector = SelectKBest(score_func=f_regression, k=n_features)
            f_selector.fit(X.fillna(0), y)
            f_scores = pd.Series(f_selector.scores_, index=X.columns).sort_values(ascending=False)
            selection_results['f_statistic'] = f_scores.head(n_features).to_dict()
            
            # 4. 随机森林重要性
            self.logger.info("4. 随机森林重要性...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X.fillna(0), y)
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            selection_results['random_forest'] = rf_importance.head(n_features).to_dict()
            
            # 5. IC分析（信息系数）
            self.logger.info("5. IC分析...")
            ic_results = {}
            for factor in factor_columns[:30]:  # 限制因子数量
                factor_data = clean_data[[factor, 'target', 'symbol', 'timeframe']].dropna()
                if len(factor_data) > 10:
                    ic = factor_data.groupby(['symbol', 'timeframe']).apply(
                        lambda x: x[factor].corr(x['target']) if len(x) > 2 else np.nan
                    ).mean()
                    ic_results[factor] = ic
            
            ic_series = pd.Series(ic_results).abs().sort_values(ascending=False)
            selection_results['ic_analysis'] = ic_series.head(n_features).to_dict()
            
            # 综合评分
            self.logger.info("6. 综合评分...")
            all_factors = set()
            for method_results in selection_results.values():
                all_factors.update(method_results.keys())
            
            composite_scores = {}
            for factor in all_factors:
                score = 0
                count = 0
                for method, results in selection_results.items():
                    if factor in results:
                        # 标准化分数
                        max_score = max(results.values())
                        if max_score > 0:
                            score += results[factor] / max_score
                            count += 1
                
                if count > 0:
                    composite_scores[factor] = score / count
            
            composite_ranking = pd.Series(composite_scores).sort_values(ascending=False)
            selection_results['composite'] = composite_ranking.head(n_features).to_dict()
            
            # 保存结果
            self.factor_analysis['selection_results'] = selection_results
            
            with open(self.output_dir / "professional_factor_selection.json", 'w', encoding='utf-8') as f:
                json.dump(selection_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info("专业因子筛选完成")
            return selection_results
            
        except Exception as e:
            self.logger.error(f"专业因子筛选失败: {e}")
            return None
    
    def generate_professional_analysis(self):
        """生成专业分析报告"""
        try:
            self.logger.info("生成专业分析报告...")
            
            # 创建综合分析图表
            self._create_factor_analysis_plots()
            
            # 生成分析报告
            report = {
                "专业因子工程分析报告": {
                    "生成时间": datetime.now().isoformat(),
                    "数据概况": {
                        "原始数据": len(self.raw_data),
                        "处理后数据": len(self.processed_factors),
                        "因子总数": len([col for col in self.processed_factors.columns 
                                      if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                                   'open', 'high', 'low', 'close', 'volume', 'target']]),
                        "交易对": self.processed_factors['symbol'].unique().tolist(),
                        "时间框架": self.processed_factors['timeframe'].unique().tolist()
                    },
                    "预处理结果": {
                        "异常值处理": "3倍标准差去极值",
                        "标准化方法": "RobustScaler",
                        "缺失值处理": "前向后向填充",
                        "中性化处理": "按symbol中性化"
                    },
                    "因子筛选结果": self.factor_analysis.get('selection_results', {})
                }
            }
            
            # 保存报告
            with open(self.output_dir / "professional_analysis_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 保存处理后的因子数据
            self.processed_factors.to_csv(self.output_dir / "professional_factors.csv", index=False)
            
            self.logger.info("专业分析报告生成完成")
            return report
            
        except Exception as e:
            self.logger.error(f"生成专业分析报告失败: {e}")
            return None
    
    def _create_factor_analysis_plots(self):
        """创建因子分析图表"""
        try:
            # 1. 因子筛选结果对比
            if 'selection_results' in self.factor_analysis:
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle('🔍 专业因子筛选结果对比', fontsize=16, fontweight='bold')
                
                methods = ['correlation', 'mutual_info', 'f_statistic', 'random_forest', 'ic_analysis', 'composite']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                
                for i, method in enumerate(methods):
                    if method in self.factor_analysis['selection_results']:
                        ax = axes[i//3, i%3]
                        data = self.factor_analysis['selection_results'][method]
                        
                        factors = list(data.keys())[:10]  # 取前10个
                        scores = [data[f] for f in factors]
                        
                        bars = ax.barh(range(len(factors)), scores, color=colors[i])
                        ax.set_yticks(range(len(factors)))
                        ax.set_yticklabels(factors, fontsize=8)
                        ax.set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        # 添加数值标签
                        for j, (bar, score) in enumerate(zip(bars, scores)):
                            ax.text(bar.get_width() + max(scores)*0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{score:.3f}', va='center', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'factor_selection_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("因子分析图表创建完成")
            
        except Exception as e:
            self.logger.error(f"创建因子分析图表失败: {e}")
    
    def run_complete_analysis(self):
        """运行完整的专业分析"""
        try:
            self.logger.info("🚀 开始专业因子工程分析...")
            
            # 1. 加载数据
            if not self.load_data():
                return False
            
            # 2. 计算高级因子
            if not self.calculate_advanced_factors():
                return False
            
            # 3. 专业预处理
            if not self.professional_preprocessing():
                return False
            
            # 4. 专业因子筛选
            selection_results = self.professional_factor_selection()
            if not selection_results:
                return False
            
            # 5. 生成专业分析报告
            report = self.generate_professional_analysis()
            if not report:
                return False
            
            self.logger.info("🎉 专业因子工程分析完成！")
            self.logger.info(f"📁 结果保存在: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"专业分析失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 专业因子工程系统")
    print("=" * 60)
    
    # 创建专业因子工程系统
    pfe = ProfessionalFactorEngineering()
    
    # 运行完整分析
    success = pfe.run_complete_analysis()
    
    if success:
        print("\n✅ 专业因子工程分析成功完成！")
        print("📊 主要改进:")
        print("  • 专业的异常值处理（3倍标准差去极值）")
        print("  • 鲁棒的标准化处理（RobustScaler）")
        print("  • 多维度因子筛选（相关性、互信息、F统计量、随机森林、IC分析）")
        print("  • 中性化处理减少偏差")
        print("  • 综合评分系统")
    else:
        print("\n❌ 专业因子工程分析失败，请检查日志")

if __name__ == "__main__":
    main() 