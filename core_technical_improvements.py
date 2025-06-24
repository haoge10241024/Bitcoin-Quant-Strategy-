#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心技术改进系统
重点解决：数据质量验证、异常值检测、OHLC逻辑检查、IC测试、因子有效性验证、共线性处理、前瞻偏差检查、系统化超参数优化
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

# 统计和机器学习
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self, missing_threshold=0.05, outlier_threshold=3.0):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.quality_issues = []
        
    def comprehensive_quality_check(self, df):
        """全面数据质量检查"""
        print("🔍 开始数据质量检查...")
        self.quality_issues = []
        
        # 1. 缺失值检查
        missing_issues = self._check_missing_values(df)
        self.quality_issues.extend(missing_issues)
        
        # 2. OHLC逻辑检查
        ohlc_issues = self._check_ohlc_logic(df)
        self.quality_issues.extend(ohlc_issues)
        
        # 3. 异常值检查
        outlier_issues = self._check_outliers(df)
        self.quality_issues.extend(outlier_issues)
        
        # 4. 时间序列连续性检查
        time_issues = self._check_time_continuity(df)
        self.quality_issues.extend(time_issues)
        
        # 5. 价格跳跃检查
        jump_issues = self._check_price_jumps(df)
        self.quality_issues.extend(jump_issues)
        
        print(f"✅ 数据质量检查完成，发现 {len(self.quality_issues)} 个问题")
        return self.quality_issues
    
    def _check_missing_values(self, df):
        """检查缺失值"""
        issues = []
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in price_cols:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > self.missing_threshold:
                    issues.append({
                        'type': 'missing_values',
                        'column': col,
                        'percentage': missing_pct,
                        'severity': 'high' if missing_pct > 0.1 else 'medium'
                    })
        
        return issues
    
    def _check_ohlc_logic(self, df):
        """检查OHLC逻辑"""
        issues = []
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 检查 high >= max(open, close)
            invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
            if invalid_high.any():
                issues.append({
                    'type': 'ohlc_logic',
                    'problem': 'high_too_low',
                    'count': invalid_high.sum(),
                    'severity': 'high'
                })
            
            # 检查 low <= min(open, close)
            invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
            if invalid_low.any():
                issues.append({
                    'type': 'ohlc_logic',
                    'problem': 'low_too_high',
                    'count': invalid_low.sum(),
                    'severity': 'high'
                })
            
            # 检查非正价格
            for col in ['open', 'high', 'low', 'close']:
                non_positive = (df[col] <= 0).any()
                if non_positive:
                    issues.append({
                        'type': 'ohlc_logic',
                        'problem': 'non_positive_price',
                        'column': col,
                        'severity': 'critical'
                    })
        
        return issues
    
    def _check_outliers(self, df):
        """检查异常值"""
        issues = []
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                # 计算收益率
                returns = df[col].pct_change().dropna()
                
                # 使用3σ原则检测异常值
                mean_return = returns.mean()
                std_return = returns.std()
                
                outliers = np.abs(returns - mean_return) > self.outlier_threshold * std_return
                outlier_count = outliers.sum()
                
                if outlier_count > len(returns) * 0.01:  # 超过1%的数据是异常值
                    issues.append({
                        'type': 'outliers',
                        'column': col,
                        'count': outlier_count,
                        'percentage': outlier_count / len(returns),
                        'severity': 'medium'
                    })
        
        return issues
    
    def _check_time_continuity(self, df):
        """检查时间序列连续性"""
        issues = []
        
        if 'datetime' in df.columns:
            df_sorted = df.sort_values('datetime')
            time_diffs = df_sorted['datetime'].diff().dropna()
            
            # 检查时间间隔的一致性
            mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
            
            if mode_diff:
                # 找出时间间隔异常的记录
                abnormal_gaps = time_diffs != mode_diff
                if abnormal_gaps.any():
                    issues.append({
                        'type': 'time_continuity',
                        'problem': 'irregular_intervals',
                        'count': abnormal_gaps.sum(),
                        'severity': 'medium'
                    })
        
        return issues
    
    def _check_price_jumps(self, df):
        """检查价格跳跃"""
        issues = []
        
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            
            # 检查极端价格跳跃（超过20%的单日变化）
            extreme_jumps = np.abs(returns) > 0.2
            jump_count = extreme_jumps.sum()
            
            if jump_count > 0:
                issues.append({
                    'type': 'price_jumps',
                    'count': jump_count,
                    'max_jump': np.abs(returns).max(),
                    'severity': 'medium'
                })
        
        return issues
    
    def auto_fix_data(self, df):
        """自动修复数据问题"""
        print("🔧 开始自动修复数据问题...")
        fixed_df = df.copy()
        
        # 1. 修复OHLC逻辑错误
        fixed_df = self._fix_ohlc_logic(fixed_df)
        
        # 2. 填充缺失值
        fixed_df = self._fill_missing_values(fixed_df)
        
        # 3. 平滑异常值
        fixed_df = self._smooth_outliers(fixed_df)
        
        print("✅ 数据修复完成")
        return fixed_df
    
    def _fix_ohlc_logic(self, df):
        """修复OHLC逻辑错误"""
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 修复high值：确保high >= max(open, close)
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            
            # 修复low值：确保low <= min(open, close)
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        return df
    
    def _fill_missing_values(self, df):
        """填充缺失值"""
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                # 使用前向填充
                df[col] = df[col].fillna(method='ffill')
                # 如果开头有缺失值，使用后向填充
                df[col] = df[col].fillna(method='bfill')
        
        # 成交量使用0填充
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def _smooth_outliers(self, df):
        """平滑异常值"""
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                # 使用分位数方法处理异常值
                Q1 = df[col].quantile(0.01)
                Q99 = df[col].quantile(0.99)
                
                df[col] = df[col].clip(lower=Q1, upper=Q99)
        
        return df


class FactorEffectivenessValidator:
    """因子有效性验证器"""
    
    def __init__(self, min_ic=0.02, min_ir=0.5, ic_window=20):
        self.min_ic = min_ic
        self.min_ir = min_ir
        self.ic_window = ic_window
        
    def comprehensive_factor_analysis(self, factors, target_returns):
        """全面因子分析"""
        print("📊 开始因子有效性分析...")
        
        results = {
            'ic_analysis': self._ic_analysis(factors, target_returns),
            'correlation_analysis': self._correlation_analysis(factors),
            'factor_selection': None,
            'multicollinearity_check': None
        }
        
        # 因子筛选
        effective_factors = self._select_effective_factors(
            factors, target_returns, results['ic_analysis']
        )
        results['factor_selection'] = effective_factors
        
        # 共线性检查
        if len(effective_factors['selected_factors']) > 1:
            multicollinearity = self._check_multicollinearity(
                factors[effective_factors['selected_factors']]
            )
            results['multicollinearity_check'] = multicollinearity
        
        print(f"✅ 因子分析完成，从 {len(factors.columns)} 个因子中筛选出 {len(effective_factors['selected_factors'])} 个有效因子")
        
        return results
    
    def _ic_analysis(self, factors, target_returns):
        """IC分析（信息系数分析）"""
        # 只选择数值列进行分析
        numeric_factors = factors.select_dtypes(include=[np.number])
        
        ic_results = {}
        
        for factor_name in numeric_factors.columns:
            if factor_name in target_returns.index:
                continue
                
            ic_series = []
            
            # 滚动计算IC
            for i in range(self.ic_window, len(numeric_factors)):
                try:
                    factor_window = numeric_factors[factor_name].iloc[i-self.ic_window:i]
                    return_window = target_returns.iloc[i-self.ic_window:i]
                    
                    # 移除缺失值
                    valid_mask = ~(factor_window.isnull() | return_window.isnull())
                    if valid_mask.sum() < 10:  # 至少需要10个有效观测
                        ic_series.append(np.nan)
                        continue
                    
                    factor_clean = factor_window[valid_mask]
                    return_clean = return_window[valid_mask]
                    
                    # 计算Spearman相关系数（更稳健）
                    ic, _ = stats.spearmanr(factor_clean, return_clean)
                    ic_series.append(ic)
                    
                except Exception as e:
                    ic_series.append(np.nan)
            
            ic_series = pd.Series(ic_series).dropna()
            
            if len(ic_series) > 0:
                ic_results[factor_name] = {
                    'ic_mean': ic_series.mean(),
                    'ic_std': ic_series.std(),
                    'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                    'ic_positive_rate': (ic_series > 0).mean(),
                    'ic_t_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))) if ic_series.std() > 0 else 0,
                    'is_effective': (abs(ic_series.mean()) > self.min_ic and 
                                   abs(ic_series.mean() / ic_series.std()) > self.min_ir)
                }
            else:
                ic_results[factor_name] = {
                    'ic_mean': 0,
                    'ic_std': 0,
                    'ic_ir': 0,
                    'ic_positive_rate': 0.5,
                    'ic_t_stat': 0,
                    'is_effective': False
                }
        
        return ic_results
    
    def _correlation_analysis(self, factors):
        """因子相关性分析"""
        # 只选择数值列进行相关性分析
        numeric_factors = factors.select_dtypes(include=[np.number])
        
        if len(numeric_factors.columns) == 0:
            return {
                'correlation_matrix': pd.DataFrame(),
                'high_correlation_pairs': [],
                'max_correlation': 0
            }
        
        correlation_matrix = numeric_factors.corr()
        
        # 找出高相关性的因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.8:  # 高相关性阈值
                    high_corr_pairs.append({
                        'factor1': correlation_matrix.columns[i],
                        'factor2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'max_correlation': correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].max() if len(correlation_matrix) > 1 else 0
        }
    
    def _select_effective_factors(self, factors, target_returns, ic_analysis):
        """选择有效因子"""
        # 基于IC分析选择因子
        effective_factors = [
            factor for factor, result in ic_analysis.items()
            if result['is_effective']
        ]
        
        # 按IC_IR排序
        factor_scores = [(factor, ic_analysis[factor]['ic_ir']) 
                        for factor in effective_factors]
        factor_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        selected_factors = [factor for factor, score in factor_scores]
        
        return {
            'selected_factors': selected_factors,
            'factor_scores': dict(factor_scores),
            'selection_summary': {
                'total_factors': len(factors.columns),
                'effective_factors': len(effective_factors),
                'selection_rate': len(effective_factors) / len(factors.columns)
            }
        }
    
    def _check_multicollinearity(self, factors, threshold=0.8):
        """检查多重共线性"""
        # 只选择数值列
        numeric_factors = factors.select_dtypes(include=[np.number])
        
        if len(numeric_factors.columns) <= 1:
            return {
                'removed_factors': [],
                'final_factors': list(numeric_factors.columns),
                'multicollinearity_resolved': False
            }
        
        correlation_matrix = numeric_factors.corr().abs()
        
        # 找出需要移除的因子
        to_remove = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    # 移除相关性较高的因子中IC_IR较低的那个
                    factor1 = correlation_matrix.columns[i]
                    factor2 = correlation_matrix.columns[j]
                    
                    if factor1 not in to_remove and factor2 not in to_remove:
                        to_remove.add(factor2)  # 简单策略：移除后面的因子
        
        final_factors = [f for f in numeric_factors.columns if f not in to_remove]
        
        return {
            'removed_factors': list(to_remove),
            'final_factors': final_factors,
            'multicollinearity_resolved': len(to_remove) > 0
        }


class WalkForwardValidator:
    """步进式验证器 - 避免前瞻偏差"""
    
    def __init__(self, train_window=252, test_window=21, min_train_size=100):
        self.train_window = train_window
        self.test_window = test_window
        self.min_train_size = min_train_size
        
    def validate_model_performance(self, X, y, model_configs):
        """验证模型性能 - 无前瞻偏差"""
        print("🔄 开始步进式模型验证...")
        
        results = {}
        
        for model_name, model_config in model_configs.items():
            print(f"   验证模型: {model_name}")
            
            model_class = model_config['class']
            model_params = model_config.get('params', {})
            
            validation_result = self._walk_forward_validation(
                X, y, model_class, model_params
            )
            
            results[model_name] = validation_result
        
        print("✅ 步进式验证完成")
        return results
    
    def _walk_forward_validation(self, X, y, model_class, model_params):
        """步进式验证"""
        predictions = []
        actual_values = []
        model_scores = []
        feature_importance_history = []
        
        # 确保数据对齐
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # 步进式验证
        for i in range(self.train_window, len(X_aligned) - self.test_window, self.test_window):
            try:
                # 训练集：只使用历史数据
                train_start = max(0, i - self.train_window)
                train_end = i
                
                X_train = X_aligned.iloc[train_start:train_end]
                y_train = y_aligned.iloc[train_start:train_end]
                
                # 测试集：未来数据
                test_start = i
                test_end = min(i + self.test_window, len(X_aligned))
                
                X_test = X_aligned.iloc[test_start:test_end]
                y_test = y_aligned.iloc[test_start:test_end]
                
                # 检查数据质量
                if len(X_train) < self.min_train_size or len(X_test) == 0:
                    continue
                
                # 处理缺失值
                X_train_clean = X_train.fillna(X_train.median())
                X_test_clean = X_test.fillna(X_train.median())  # 用训练集的中位数填充测试集
                
                # 训练模型
                model = model_class(**model_params)
                model.fit(X_train_clean, y_train)
                
                # 预测
                pred = model.predict(X_test_clean)
                
                # 记录结果
                predictions.extend(pred)
                actual_values.extend(y_test.values)
                
                # 评估模型
                if len(y_test) > 1:
                    score = r2_score(y_test, pred)
                    model_scores.append(score)
                
                # 记录特征重要性（如果模型支持）
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X_train_clean.columns, model.feature_importances_))
                    feature_importance_history.append(importance)
                
            except Exception as e:
                print(f"   警告: 验证过程中出现错误: {e}")
                continue
        
        # 计算总体性能指标
        if len(predictions) > 0 and len(actual_values) > 0:
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            
            mse = mean_squared_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
            
            # 计算方向准确率
            actual_direction = np.sign(actual_values)
            pred_direction = np.sign(predictions)
            direction_accuracy = np.mean(actual_direction == pred_direction)
            
            return {
                'predictions': predictions,
                'actual': actual_values,
                'mse': mse,
                'r2_score': r2,
                'direction_accuracy': direction_accuracy,
                'model_scores': model_scores,
                'mean_score': np.mean(model_scores) if model_scores else 0,
                'score_std': np.std(model_scores) if model_scores else 0,
                'feature_importance_history': feature_importance_history,
                'validation_periods': len(model_scores)
            }
        else:
            return {
                'predictions': np.array([]),
                'actual': np.array([]),
                'mse': np.inf,
                'r2_score': -np.inf,
                'direction_accuracy': 0.5,
                'model_scores': [],
                'mean_score': 0,
                'score_std': 0,
                'feature_importance_history': [],
                'validation_periods': 0
            }


class HyperparameterOptimizer:
    """系统化超参数优化器"""
    
    def __init__(self, cv_folds=5, n_iter=50, random_state=42):
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.random_state = random_state
        
    def optimize_hyperparameters(self, X, y, model_configs):
        """系统化超参数优化"""
        print("⚙️ 开始超参数优化...")
        
        optimized_models = {}
        
        for model_name, config in model_configs.items():
            print(f"   优化模型: {model_name}")
            
            model_class = config['class']
            param_distributions = config.get('param_distributions', {})
            
            if param_distributions:
                best_model = self._optimize_single_model(
                    X, y, model_class, param_distributions
                )
                optimized_models[model_name] = best_model
            else:
                # 如果没有参数分布，使用默认参数
                optimized_models[model_name] = {
                    'model': model_class(),
                    'best_params': {},
                    'best_score': 0
                }
        
        print("✅ 超参数优化完成")
        return optimized_models
    
    def _optimize_single_model(self, X, y, model_class, param_distributions):
        """优化单个模型"""
        try:
            # 处理缺失值
            X_clean = X.fillna(X.median())
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            # 随机搜索
            model = model_class()
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=self.n_iter,
                cv=tscv,
                scoring='r2',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            search.fit(X_clean, y)
            
            return {
                'model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
        except Exception as e:
            print(f"   警告: 优化过程中出现错误: {e}")
            return {
                'model': model_class(),
                'best_params': {},
                'best_score': 0,
                'cv_results': {}
            }


class CoreTechnicalImprovementSystem:
    """核心技术改进系统"""
    
    def __init__(self):
        self.data_validator = DataQualityValidator()
        self.factor_validator = FactorEffectivenessValidator()
        self.walk_forward_validator = WalkForwardValidator()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        self.results = {}
        
    def run_comprehensive_improvements(self, data_path, factor_path=None):
        """运行全面技术改进"""
        print("🚀 开始核心技术改进...")
        
        # 1. 数据质量验证和修复
        print("\n" + "="*50)
        print("第1步: 数据质量验证和修复")
        print("="*50)
        
        data = pd.read_csv(data_path)
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        quality_issues = self.data_validator.comprehensive_quality_check(data)
        fixed_data = self.data_validator.auto_fix_data(data)
        
        self.results['data_quality'] = {
            'issues': quality_issues,
            'fixed_data_shape': fixed_data.shape
        }
        
        # 2. 因子有效性验证
        print("\n" + "="*50)
        print("第2步: 因子有效性验证")
        print("="*50)
        
        if factor_path:
            factors = pd.read_csv(factor_path)
        else:
            # 如果没有提供因子文件，从数据中计算基础因子
            factors = self._calculate_basic_factors(fixed_data)
        
        # 计算目标变量（未来收益率）
        target_returns = fixed_data.groupby(['symbol', 'timeframe'])['close'].pct_change().shift(-1)
        target_returns = target_returns.dropna()
        
        factor_analysis = self.factor_validator.comprehensive_factor_analysis(
            factors, target_returns
        )
        
        self.results['factor_analysis'] = factor_analysis
        
        # 3. 获取有效因子
        effective_factors = factors[factor_analysis['factor_selection']['selected_factors']]
        
        # 4. 处理多重共线性
        if factor_analysis['multicollinearity_check']:
            final_factors = effective_factors[
                factor_analysis['multicollinearity_check']['final_factors']
            ]
        else:
            final_factors = effective_factors
        
        # 5. 步进式模型验证
        print("\n" + "="*50)
        print("第3步: 步进式模型验证（避免前瞻偏差）")
        print("="*50)
        
        model_configs = self._get_model_configs()
        
        validation_results = self.walk_forward_validator.validate_model_performance(
            final_factors, target_returns, model_configs
        )
        
        self.results['validation_results'] = validation_results
        
        # 6. 超参数优化
        print("\n" + "="*50)
        print("第4步: 系统化超参数优化")
        print("="*50)
        
        optimization_configs = self._get_optimization_configs()
        
        optimized_models = self.hyperparameter_optimizer.optimize_hyperparameters(
            final_factors, target_returns, optimization_configs
        )
        
        self.results['optimized_models'] = optimized_models
        
        # 7. 生成改进报告
        self._generate_improvement_report()
        
        print("\n🎉 核心技术改进完成！")
        return self.results
    
    def _calculate_basic_factors(self, data):
        """计算基础因子"""
        print("   计算基础因子...")
        
        factors_list = []
        
        for (symbol, timeframe), group in data.groupby(['symbol', 'timeframe']):
            group = group.reset_index(drop=True).copy()
            
            # 价格因子
            for period in [5, 10, 20]:
                group[f'return_{period}'] = group['close'].pct_change(period)
                group[f'ma_{period}'] = group['close'].rolling(period).mean()
                group[f'volatility_{period}'] = group['close'].pct_change().rolling(period).std()
            
            # 技术指标
            group['rsi_14'] = self._calculate_rsi(group['close'], 14)
            group['macd'] = group['close'].ewm(span=12).mean() - group['close'].ewm(span=26).mean()
            
            factors_list.append(group)
        
        all_factors = pd.concat(factors_list, ignore_index=True)
        
        # 选择因子列
        factor_columns = [col for col in all_factors.columns 
                         if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                      'open', 'high', 'low', 'close', 'volume']]
        
        return all_factors[factor_columns].fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_model_configs(self):
        """获取模型配置"""
        return {
            'LinearRegression': {
                'class': LinearRegression,
                'params': {}
            },
            'Ridge': {
                'class': Ridge,
                'params': {'alpha': 1.0}
            },
            'RandomForest': {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'GradientBoosting': {
                'class': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'random_state': 42}
            }
        }
    
    def _get_optimization_configs(self):
        """获取优化配置"""
        return {
            'RandomForest': {
                'class': RandomForestRegressor,
                'param_distributions': {
                    'n_estimators': randint(50, 200),
                    'max_depth': randint(3, 15),
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingRegressor,
                'param_distributions': {
                    'n_estimators': randint(50, 200),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8),
                    'subsample': uniform(0.8, 0.2)
                }
            }
        }
    
    def _generate_improvement_report(self):
        """生成改进报告"""
        print("\n📊 生成改进报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'improvements_summary': {
                'data_quality_issues_found': len(self.results['data_quality']['issues']),
                'factors_analyzed': len(self.results['factor_analysis']['ic_analysis']),
                'effective_factors_selected': len(self.results['factor_analysis']['factor_selection']['selected_factors']),
                'models_validated': len(self.results['validation_results']),
                'models_optimized': len(self.results['optimized_models'])
            },
            'best_model_performance': self._get_best_model_performance(),
            'factor_effectiveness_summary': self._get_factor_effectiveness_summary()
        }
        
        # 保存报告
        with open('core_technical_improvement_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 改进报告已保存到 core_technical_improvement_report.json")
    
    def _get_best_model_performance(self):
        """获取最佳模型性能"""
        best_model = None
        best_score = -np.inf
        
        for model_name, results in self.results['validation_results'].items():
            if results['r2_score'] > best_score:
                best_score = results['r2_score']
                best_model = model_name
        
        return {
            'best_model': best_model,
            'best_r2_score': best_score,
            'best_direction_accuracy': self.results['validation_results'][best_model]['direction_accuracy'] if best_model else 0
        }
    
    def _get_factor_effectiveness_summary(self):
        """获取因子有效性总结"""
        ic_analysis = self.results['factor_analysis']['ic_analysis']
        
        effective_count = sum(1 for result in ic_analysis.values() if result['is_effective'])
        
        # 找出最佳因子
        best_factors = sorted(
            [(factor, result['ic_ir']) for factor, result in ic_analysis.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        return {
            'total_factors': len(ic_analysis),
            'effective_factors': effective_count,
            'effectiveness_rate': effective_count / len(ic_analysis) if ic_analysis else 0,
            'top_5_factors': [factor for factor, score in best_factors]
        }


def main():
    """主函数"""
    print("🔧 核心技术改进系统启动")
    
    # 初始化改进系统
    improvement_system = CoreTechnicalImprovementSystem()
    
    # 运行改进
    data_path = "data/extended/extended_all_data_3years_20250623_215123.csv"
    factor_path = "factors/bitcoin_factors_20250623_215658.csv"
    
    results = improvement_system.run_comprehensive_improvements(
        data_path=data_path,
        factor_path=factor_path
    )
    
    print("\n🎯 改进完成！主要成果:")
    print(f"   - 数据质量问题: {len(results['data_quality']['issues'])} 个")
    print(f"   - 有效因子筛选: {len(results['factor_analysis']['factor_selection']['selected_factors'])} 个")
    print(f"   - 模型验证完成: {len(results['validation_results'])} 个模型")
    print(f"   - 最佳模型: {results.get('best_model_performance', {}).get('best_model', 'N/A')}")


if __name__ == "__main__":
    main() 