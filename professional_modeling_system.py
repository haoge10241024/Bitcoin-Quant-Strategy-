#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业建模系统 - 包含详细的目标变量定义、模型分析和评估
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 机器学习和统计
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ProfessionalModelingSystem:
    """专业建模系统"""
    
    def __init__(self, factor_data_path: str = None):
        self.factor_data_path = factor_data_path or "professional_factors/professional_factors.csv"
        self.results_dir = Path("professional_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 数据和模型
        self.data = None
        self.features = None
        self.targets = {}  # 多个目标变量
        self.models = {}
        self.model_results = {}
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.results_dir / f"professional_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_factor_data(self):
        """加载因子数据"""
        try:
            self.logger.info("加载因子数据...")
            self.data = pd.read_csv(self.factor_data_path)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
            self.logger.info(f"因子数据加载完成: {len(self.data)} 条记录, {len(self.data.columns)} 列")
            return True
            
        except Exception as e:
            self.logger.error(f"因子数据加载失败: {e}")
            return False
    
    def prepare_target_variables(self):
        """准备多个目标变量"""
        try:
            self.logger.info("准备目标变量...")
            
            # 按交易对和时间框架分组处理
            processed_data = []
            
            for (symbol, timeframe), group in self.data.groupby(['symbol', 'timeframe']):
                group = group.reset_index(drop=True).copy()
                
                # 1. 未来收益率（不同期数）
                for period in [1, 3, 5, 10]:
                    group[f'future_return_{period}'] = group['close'].pct_change(period).shift(-period)
                    group[f'future_log_return_{period}'] = (np.log(group['close']) - np.log(group['close'].shift(period))).shift(-period)
                
                # 2. 未来波动率
                for period in [5, 10, 20]:
                    future_vol = group['close'].pct_change().rolling(period).std().shift(-period)
                    group[f'future_volatility_{period}'] = future_vol * np.sqrt(252)  # 年化
                
                # 3. 未来最大回撤
                for period in [10, 20]:
                    def calc_future_drawdown(series, window):
                        result = []
                        for i in range(len(series)):
                            if i + window < len(series):
                                future_prices = series.iloc[i:i+window]
                                cumulative = (1 + future_prices.pct_change()).cumprod()
                                rolling_max = cumulative.expanding().max()
                                drawdown = (cumulative - rolling_max) / rolling_max
                                result.append(drawdown.min())
                            else:
                                result.append(np.nan)
                        return pd.Series(result, index=series.index)
                    
                    group[f'future_max_drawdown_{period}'] = calc_future_drawdown(group['close'], period)
                
                # 4. 未来夏普比率
                for period in [20]:
                    def calc_future_sharpe(series, window):
                        result = []
                        for i in range(len(series)):
                            if i + window < len(series):
                                future_returns = series.iloc[i:i+window].pct_change().dropna()
                                if len(future_returns) > 1 and future_returns.std() > 0:
                                    sharpe = future_returns.mean() / future_returns.std() * np.sqrt(252)
                                    result.append(sharpe)
                                else:
                                    result.append(np.nan)
                            else:
                                result.append(np.nan)
                        return pd.Series(result, index=series.index)
                    
                    group[f'future_sharpe_{period}'] = calc_future_sharpe(group['close'], period)
                
                # 5. 分类目标（涨跌方向）
                for period in [1, 5]:
                    future_return = group['close'].pct_change(period).shift(-period)
                    group[f'future_direction_{period}'] = (future_return > 0).astype(int)
                    
                    # 三分类：涨/平/跌
                    group[f'future_trend_{period}'] = pd.cut(
                        future_return, 
                        bins=[-np.inf, -0.01, 0.01, np.inf], 
                        labels=[0, 1, 2]  # 0:跌, 1:平, 2:涨
                    ).astype(float)
                
                processed_data.append(group)
            
            self.data = pd.concat(processed_data, ignore_index=True)
            
            # 准备特征和目标变量
            feature_columns = [col for col in self.data.columns 
                             if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                          'open', 'high', 'low', 'close', 'volume'] 
                             and not col.startswith('future_')]
            
            target_columns = [col for col in self.data.columns if col.startswith('future_')]
            
            self.features = self.data[feature_columns].copy()
            
            for target_col in target_columns:
                self.targets[target_col] = self.data[target_col].copy()
            
            self.logger.info(f"目标变量准备完成: {len(feature_columns)} 个特征, {len(target_columns)} 个目标变量")
            self.logger.info(f"目标变量类型: {list(self.targets.keys())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"准备目标变量失败: {e}")
            return False
    
    def train_professional_models(self, target_name: str = 'future_return_1'):
        """训练专业模型"""
        try:
            self.logger.info(f"训练专业模型 - 目标变量: {target_name}")
            
            if target_name not in self.targets:
                self.logger.error(f"目标变量 {target_name} 不存在")
                return False
            
            # 准备数据
            X = self.features
            y = self.targets[target_name]
            
            # 移除缺失值
            valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_indices].copy()
            y_clean = y[valid_indices].copy()
            
            if len(X_clean) < 100:
                self.logger.error("有效数据不足，无法进行模型训练")
                return False
            
            # 填充剩余缺失值
            X_clean = X_clean.fillna(X_clean.median())
            
            # 特征选择
            selector = SelectKBest(score_func=f_regression, k=min(50, X_clean.shape[1]))
            X_selected = selector.fit_transform(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            self.logger.info(f"特征选择完成，选择了 {len(selected_features)} 个特征")
            
            # 数据标准化
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 定义专业模型
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            model_results = {}
            
            # 训练和评估每个模型
            for name, model in models.items():
                self.logger.info(f"训练 {name} 模型...")
                
                cv_scores = {
                    'mse': [],
                    'mae': [],
                    'r2': [],
                    'explained_variance': []
                }
                
                predictions = []
                actuals = []
                
                # 交叉验证
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    try:
                        # 训练模型
                        model.fit(X_train, y_train)
                        
                        # 预测
                        y_pred = model.predict(X_val)
                        
                        # 计算评估指标
                        mse = mean_squared_error(y_val, y_pred)
                        mae = mean_absolute_error(y_val, y_pred)
                        r2 = r2_score(y_val, y_pred)
                        ev = explained_variance_score(y_val, y_pred)
                        
                        cv_scores['mse'].append(mse)
                        cv_scores['mae'].append(mae)
                        cv_scores['r2'].append(r2)
                        cv_scores['explained_variance'].append(ev)
                        
                        predictions.extend(y_pred)
                        actuals.extend(y_val)
                        
                    except Exception as e:
                        self.logger.warning(f"{name} 第{fold+1}折训练失败: {e}")
                        continue
                
                if cv_scores['mse']:
                    # 计算平均分数
                    avg_scores = {
                        'mse': np.mean(cv_scores['mse']),
                        'mae': np.mean(cv_scores['mae']),
                        'r2': np.mean(cv_scores['r2']),
                        'explained_variance': np.mean(cv_scores['explained_variance']),
                        'rmse': np.sqrt(np.mean(cv_scores['mse'])),
                        'mse_std': np.std(cv_scores['mse']),
                        'r2_std': np.std(cv_scores['r2'])
                    }
                    
                    # 在全部数据上训练最终模型
                    try:
                        model.fit(X_scaled, y_clean)
                        
                        model_results[name] = {
                            'model': model,
                            'scaler': scaler,
                            'selector': selector,
                            'selected_features': selected_features,
                            'scores': avg_scores,
                            'predictions': predictions,
                            'actuals': actuals
                        }
                        
                        self.logger.info(f"{name} 训练完成 - R²: {avg_scores['r2']:.4f}±{avg_scores['r2_std']:.4f}, RMSE: {avg_scores['rmse']:.6f}")
                        
                    except Exception as e:
                        self.logger.warning(f"{name} 最终训练失败: {e}")
                        continue
                else:
                    self.logger.warning(f"{name} 所有折都训练失败")
            
            self.models[target_name] = model_results
            
            self.logger.info(f"模型训练完成，成功训练 {len(model_results)} 个模型")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return False
    
    def comprehensive_model_evaluation(self, target_name: str = 'future_return_1'):
        """综合模型评估"""
        try:
            self.logger.info(f"综合模型评估 - {target_name}")
            
            if target_name not in self.models or not self.models[target_name]:
                self.logger.error(f"没有找到 {target_name} 的训练模型")
                return None
            
            model_results = self.models[target_name]
            
            # 创建评估结果DataFrame
            evaluation_data = []
            
            for name, result in model_results.items():
                scores = result['scores']
                evaluation_data.append({
                    'Model': name,
                    'R²': scores['r2'],
                    'R²_std': scores['r2_std'],
                    'MSE': scores['mse'],
                    'MSE_std': scores['mse_std'],
                    'MAE': scores['mae'],
                    'RMSE': scores['rmse'],
                    'Explained_Variance': scores['explained_variance']
                })
            
            evaluation_df = pd.DataFrame(evaluation_data)
            evaluation_df = evaluation_df.sort_values('R²', ascending=False)
            
            # 保存评估结果
            evaluation_df.to_csv(self.results_dir / f"model_evaluation_{target_name}.csv", index=False)
            
            # 生成详细的可视化分析
            self._create_comprehensive_plots(target_name, evaluation_df, model_results)
            
            # 模型稳定性分析
            stability_analysis = self._analyze_model_stability(model_results)
            
            # 预测能力分析
            prediction_analysis = self._analyze_prediction_capability(model_results)
            
            # 综合分析结果
            comprehensive_results = {
                'target_variable': target_name,
                'evaluation_summary': evaluation_df.to_dict('records'),
                'best_model': evaluation_df.iloc[0]['Model'],
                'best_model_metrics': {
                    'R²': float(evaluation_df.iloc[0]['R²']),
                    'RMSE': float(evaluation_df.iloc[0]['RMSE']),
                    'MAE': float(evaluation_df.iloc[0]['MAE'])
                },
                'stability_analysis': stability_analysis,
                'prediction_analysis': prediction_analysis,
                'analysis_time': datetime.now().isoformat()
            }
            
            # 保存综合分析结果
            with open(self.results_dir / f"comprehensive_analysis_{target_name}.json", 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info("综合模型评估完成")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"综合模型评估失败: {e}")
            return None
    
    def _create_comprehensive_plots(self, target_name: str, evaluation_df: pd.DataFrame, model_results: dict):
        """创建综合分析图表"""
        try:
            # 1. 模型性能对比图
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'📊 专业模型分析 - {target_name}', fontsize=16, fontweight='bold')
            
            # R²对比（带误差条）
            models = evaluation_df['Model']
            r2_scores = evaluation_df['R²']
            r2_errors = evaluation_df['R²_std']
            
            bars1 = axes[0, 0].bar(range(len(models)), r2_scores, yerr=r2_errors, capsize=5)
            axes[0, 0].set_title('R² Score (±std)', fontweight='bold')
            axes[0, 0].set_xticks(range(len(models)))
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_errors)*0.1, 
                               f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # RMSE对比
            rmse_scores = evaluation_df['RMSE']
            bars2 = axes[0, 1].bar(range(len(models)), rmse_scores)
            axes[0, 1].set_title('RMSE', fontweight='bold')
            axes[0, 1].set_xticks(range(len(models)))
            axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # MAE对比
            mae_scores = evaluation_df['MAE']
            bars3 = axes[0, 2].bar(range(len(models)), mae_scores)
            axes[0, 2].set_title('MAE', fontweight='bold')
            axes[0, 2].set_xticks(range(len(models)))
            axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 预测vs实际散点图（最佳模型）
            best_model_name = evaluation_df.iloc[0]['Model']
            best_model_result = model_results[best_model_name]
            
            predictions = best_model_result['predictions']
            actuals = best_model_result['actuals']
            
            axes[1, 0].scatter(actuals, predictions, alpha=0.6, s=20)
            axes[1, 0].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', linewidth=2)
            axes[1, 0].set_xlabel('实际值')
            axes[1, 0].set_ylabel('预测值')
            axes[1, 0].set_title(f'预测精度 - {best_model_name}', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 残差分析
            residuals = np.array(predictions) - np.array(actuals)
            axes[1, 1].scatter(predictions, residuals, alpha=0.6, s=20)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('预测值')
            axes[1, 1].set_ylabel('残差')
            axes[1, 1].set_title('残差分析', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 残差分布
            axes[1, 2].hist(residuals, bins=30, alpha=0.7, density=True)
            axes[1, 2].axvline(x=0, color='r', linestyle='--')
            axes[1, 2].set_xlabel('残差')
            axes[1, 2].set_ylabel('密度')
            axes[1, 2].set_title('残差分布', fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'comprehensive_model_analysis_{target_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"创建综合分析图表失败: {e}")
    
    def _analyze_model_stability(self, model_results: dict) -> dict:
        """分析模型稳定性"""
        stability_analysis = {}
        
        for name, result in model_results.items():
            scores = result['scores']
            stability_analysis[name] = {
                'r2_coefficient_of_variation': scores['r2_std'] / abs(scores['r2']) if scores['r2'] != 0 else np.inf,
                'mse_coefficient_of_variation': scores['mse_std'] / scores['mse'] if scores['mse'] != 0 else np.inf,
                'stability_score': 1 / (1 + scores['r2_std']) if scores['r2_std'] > 0 else 1.0
            }
        
        return stability_analysis
    
    def _analyze_prediction_capability(self, model_results: dict) -> dict:
        """分析预测能力"""
        prediction_analysis = {}
        
        for name, result in model_results.items():
            predictions = np.array(result['predictions'])
            actuals = np.array(result['actuals'])
            
            # 方向准确率
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            direction_accuracy = np.mean(pred_direction == actual_direction)
            
            # 预测范围覆盖率
            pred_range = np.max(predictions) - np.min(predictions)
            actual_range = np.max(actuals) - np.min(actuals)
            range_coverage = min(pred_range / actual_range, actual_range / pred_range) if actual_range != 0 else 0
            
            prediction_analysis[name] = {
                'direction_accuracy': float(direction_accuracy),
                'range_coverage': float(range_coverage),
                'prediction_variance': float(np.var(predictions)),
                'actual_variance': float(np.var(actuals))
            }
        
        return prediction_analysis
    
    def run_complete_modeling(self):
        """运行完整的专业建模流程"""
        try:
            self.logger.info("🚀 开始专业建模分析...")
            
            # 1. 加载数据
            if not self.load_factor_data():
                return False
            
            # 2. 准备目标变量
            if not self.prepare_target_variables():
                return False
            
            # 3. 对主要目标变量进行建模
            main_targets = ['future_return_1', 'future_return_5', 'future_volatility_10']
            
            for target in main_targets:
                if target in self.targets:
                    self.logger.info(f"开始建模: {target}")
                    
                    # 训练模型
                    if self.train_professional_models(target):
                        # 综合评估
                        self.comprehensive_model_evaluation(target)
                    else:
                        self.logger.warning(f"{target} 建模失败")
            
            # 4. 生成总结报告
            self._generate_modeling_summary()
            
            self.logger.info("🎉 专业建模分析完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"专业建模分析失败: {e}")
            return False
    
    def _generate_modeling_summary(self):
        """生成建模总结报告"""
        try:
            summary = {
                "专业建模分析总结": {
                    "分析时间": datetime.now().isoformat(),
                    "目标变量说明": {
                        "future_return_1": "未来1期收益率（主要预测目标）",
                        "future_return_5": "未来5期收益率（中期趋势）",
                        "future_volatility_10": "未来10期波动率（风险预测）",
                        "future_direction_1": "未来1期涨跌方向（分类目标）",
                        "future_max_drawdown_10": "未来10期最大回撤（风险指标）"
                    },
                    "模型改进": [
                        "使用RobustScaler进行数据标准化，对异常值更鲁棒",
                        "采用TimeSeriesSplit进行时间序列交叉验证",
                        "增加了9种专业机器学习模型",
                        "引入模型稳定性分析",
                        "添加预测能力分析（方向准确率、范围覆盖率）",
                        "使用SelectKBest进行特征选择",
                        "增加残差分析和分布检验"
                    ],
                    "评估指标": [
                        "R²决定系数（解释方差比例）",
                        "RMSE均方根误差（预测精度）",
                        "MAE平均绝对误差（鲁棒性指标）",
                        "方向准确率（趋势预测能力）",
                        "模型稳定性系数（跨时间稳定性）"
                    ]
                }
            }
            
            with open(self.results_dir / "modeling_summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info("建模总结报告生成完成")
            
        except Exception as e:
            self.logger.error(f"生成建模总结报告失败: {e}")

def main():
    """主函数"""
    print("🚀 专业建模系统")
    print("=" * 60)
    
    # 创建专业建模系统
    pms = ProfessionalModelingSystem()
    
    # 运行完整建模
    success = pms.run_complete_modeling()
    
    if success:
        print("\n✅ 专业建模分析成功完成！")
        print("📊 主要改进:")
        print("  • 多个目标变量（收益率、波动率、方向、回撤）")
        print("  • 9种专业机器学习模型")
        print("  • 时间序列交叉验证")
        print("  • 模型稳定性分析")
        print("  • 预测能力分析")
        print("  • 残差分析和分布检验")
    else:
        print("\n❌ 专业建模分析失败，请检查日志")

if __name__ == "__main__":
    main() 