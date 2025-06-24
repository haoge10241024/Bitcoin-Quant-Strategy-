#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强模型系统 - 解决缺失值处理和模型性能问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedDataPreprocessor:
    """高级数据预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        
    def advanced_missing_value_handling(self, factors):
        """高级缺失值处理"""
        print("🔧 执行高级缺失值处理...")
        
        # 只选择数值列
        numeric_factors = factors.select_dtypes(include=[np.number])
        processed_factors = numeric_factors.copy()
        
        if len(processed_factors.columns) == 0:
            print("⚠️ 警告: 没有找到数值列")
            return factors
        
        # 按因子类型分别处理
        factor_types = self._classify_factors(processed_factors)
        
        for factor_type, factor_list in factor_types.items():
            if len(factor_list) > 0:
                if factor_type == 'price':
                    # 价格因子用前向填充
                    processed_factors[factor_list] = processed_factors[factor_list].fillna(method='ffill')
                    processed_factors[factor_list] = processed_factors[factor_list].fillna(method='bfill')
                    
                elif factor_type == 'volume':
                    # 成交量因子用0填充
                    processed_factors[factor_list] = processed_factors[factor_list].fillna(0)
                    
                elif factor_type == 'technical':
                    # 技术指标用KNN填充
                    if len(factor_list) > 1:
                        try:
                            imputer = KNNImputer(n_neighbors=5)
                            processed_factors[factor_list] = imputer.fit_transform(processed_factors[factor_list])
                        except:
                            # 如果KNN失败，使用中位数填充
                            processed_factors[factor_list] = processed_factors[factor_list].fillna(
                                processed_factors[factor_list].median()
                            )
                    else:
                        processed_factors[factor_list] = processed_factors[factor_list].fillna(
                            processed_factors[factor_list].median()
                        )
                
                else:
                    # 其他因子用中位数填充
                    processed_factors[factor_list] = processed_factors[factor_list].fillna(
                        processed_factors[factor_list].median()
                    )
        
        print(f"✅ 缺失值处理完成，处理前缺失值: {numeric_factors.isnull().sum().sum()}, 处理后: {processed_factors.isnull().sum().sum()}")
        
        return processed_factors
    
    def _classify_factors(self, factors):
        """因子分类"""
        factor_types = {
            'price': [],
            'volume': [],
            'technical': [],
            'statistical': [],
            'other': []
        }
        
        for col in factors.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['price', 'ma_', 'ema_', 'return']):
                factor_types['price'].append(col)
            elif any(keyword in col_lower for keyword in ['volume', 'vol_']):
                factor_types['volume'].append(col)
            elif any(keyword in col_lower for keyword in ['rsi', 'macd', 'bb_', 'kdj', 'williams']):
                factor_types['technical'].append(col)
            elif any(keyword in col_lower for keyword in ['volatility', 'skew', 'kurt', 'sharpe']):
                factor_types['statistical'].append(col)
            else:
                factor_types['other'].append(col)
        
        return factor_types
    
    def robust_scaling(self, factors):
        """稳健标准化"""
        print("📏 执行稳健标准化...")
        
        scaler = RobustScaler()
        scaled_factors = pd.DataFrame(
            scaler.fit_transform(factors),
            index=factors.index,
            columns=factors.columns
        )
        
        self.scalers['robust'] = scaler
        
        print("✅ 标准化完成")
        return scaled_factors


class EnhancedFeatureEngineering:
    """增强特征工程"""
    
    def __init__(self):
        pass
    
    def create_advanced_features(self, data):
        """创建高级特征"""
        print("🔬 创建高级特征...")
        
        enhanced_data = data.copy()
        
        # 1. 技术指标增强
        enhanced_data = self._add_advanced_technical_indicators(enhanced_data)
        
        # 2. 交互特征
        enhanced_data = self._add_interaction_features(enhanced_data)
        
        # 3. 时间特征
        enhanced_data = self._add_time_features(enhanced_data)
        
        # 4. 统计特征
        enhanced_data = self._add_statistical_features(enhanced_data)
        
        print(f"✅ 特征工程完成，新增特征数量: {len(enhanced_data.columns) - len(data.columns)}")
        
        return enhanced_data
    
    def _add_advanced_technical_indicators(self, data):
        """添加高级技术指标"""
        # 布林带位置
        required_bb_cols = ['bb_upper_20', 'bb_lower_20', 'close']
        if all(col in data.columns for col in required_bb_cols):
            data['bollinger_position'] = (data['close'] - data['bb_lower_20']) / (data['bb_upper_20'] - data['bb_lower_20'])
        
        # RSI背离
        if 'rsi_14' in data.columns:
            data['rsi_divergence'] = data['rsi_14'].diff()
            data['rsi_momentum'] = data['rsi_14'].rolling(5).mean() - data['rsi_14'].rolling(20).mean()
        
        # MACD增强
        if 'macd' in data.columns:
            data['macd_momentum'] = data['macd'].diff()
            data['macd_acceleration'] = data['macd_momentum'].diff()
        
        # 成交量价格趋势
        if all(col in data.columns for col in ['volume', 'close']):
            data['volume_price_trend'] = data['volume'] * data['close'].pct_change()
            volume_ma = data['volume'].rolling(20).mean()
            data['volume_ma_ratio'] = data['volume'] / volume_ma.replace(0, np.nan)
        
        return data
    
    def _add_interaction_features(self, data):
        """添加交互特征"""
        # 移动平均交叉
        if all(col in data.columns for col in ['ma_5', 'ma_20']):
            data['ma_cross_5_20'] = (data['ma_5'] > data['ma_20']).astype(int)
            data['ma_distance_5_20'] = (data['ma_5'] - data['ma_20']) / data['ma_20'].replace(0, np.nan)
        
        # 波动率制度
        if 'volatility_20' in data.columns:
            try:
                data['volatility_regime'] = pd.qcut(
                    data['volatility_20'].rank(method='first'), 
                    3, 
                    labels=[0, 1, 2],
                    duplicates='drop'
                ).astype(float)
            except:
                # 如果分位数切分失败，使用简单的三分法
                vol_33 = data['volatility_20'].quantile(0.33)
                vol_67 = data['volatility_20'].quantile(0.67)
                data['volatility_regime'] = np.where(
                    data['volatility_20'] <= vol_33, 0,
                    np.where(data['volatility_20'] <= vol_67, 1, 2)
                )
        
        # 价格动量组合
        if all(col in data.columns for col in ['return_5', 'return_20']):
            data['momentum_combination'] = data['return_5'] * data['return_20']
        
        return data
    
    def _add_time_features(self, data):
        """添加时间特征"""
        if 'datetime' in data.columns:
            data['hour'] = pd.to_datetime(data['datetime']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek
            data['month'] = pd.to_datetime(data['datetime']).dt.month
        
        return data
    
    def _add_statistical_features(self, data):
        """添加统计特征"""
        # 收益率的高阶矩
        if 'return_1' in data.columns:
            data['return_skew_20'] = data['return_1'].rolling(20).skew()
            data['return_kurt_20'] = data['return_1'].rolling(20).kurt()
        
        # 价格分位数位置
        if 'close' in data.columns:
            data['price_percentile_60'] = data['close'].rolling(60).rank(pct=True)
        
        return data


class EnsembleModelSystem:
    """集成模型系统"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        
    def initialize_models(self):
        """初始化模型"""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=0.1)
        }
    
    def train_ensemble(self, X, y):
        """训练集成模型"""
        print("🤖 训练集成模型...")
        
        # 确保数据清洁
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())
        
        # 训练各个模型
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                # 时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_clean):
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    score = r2_score(y_val, pred)
                    scores.append(score)
                
                model_scores[name] = np.mean(scores)
                print(f"   {name}: R² = {model_scores[name]:.4f}")
                
            except Exception as e:
                print(f"   警告: {name} 训练失败: {e}")
                model_scores[name] = -1.0
        
        # 计算权重（基于性能）
        self._calculate_ensemble_weights(model_scores)
        
        # 在全部数据上重新训练
        for name, model in self.models.items():
            try:
                model.fit(X_clean, y_clean)
            except Exception as e:
                print(f"   警告: {name} 最终训练失败: {e}")
        
        print("✅ 集成模型训练完成")
        
        return model_scores
    
    def _calculate_ensemble_weights(self, model_scores):
        """计算集成权重"""
        # 将负分数设为0
        positive_scores = {name: max(0, score) for name, score in model_scores.items()}
        
        # 归一化权重
        total_score = sum(positive_scores.values())
        
        if total_score > 0:
            self.weights = {name: score / total_score for name, score in positive_scores.items()}
        else:
            # 如果所有模型都表现不好，使用均等权重
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        print(f"   集成权重: {self.weights}")
    
    def predict_ensemble(self, X):
        """集成预测"""
        X_clean = X.fillna(X.median())
        
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X_clean)
            except Exception as e:
                print(f"   警告: {name} 预测失败: {e}")
                predictions[name] = np.zeros(len(X_clean))
        
        # 加权平均
        ensemble_pred = np.zeros(len(X_clean))
        for name, pred in predictions.items():
            ensemble_pred += self.weights.get(name, 0) * pred
        
        return ensemble_pred


class ImprovedTargetDefinition:
    """改进的目标变量定义"""
    
    def __init__(self):
        pass
    
    def create_multiple_targets(self, data):
        """创建多个目标变量"""
        print("🎯 创建改进的目标变量...")
        
        targets = {}
        
        # 按分组处理
        for (symbol, timeframe), group in data.groupby(['symbol', 'timeframe']):
            group = group.reset_index(drop=True).copy()
            
            # 1. 多期收益率
            group['target_1d'] = group['close'].pct_change().shift(-1)
            group['target_3d'] = group['close'].pct_change(3).shift(-3)
            group['target_5d'] = group['close'].pct_change(5).shift(-5)
            
            # 2. 风险调整收益
            volatility = group['close'].pct_change().rolling(20).std()
            group['risk_adjusted_return_1d'] = group['target_1d'] / (volatility + 1e-6)
            
            # 3. 方向性目标
            group['direction_1d'] = (group['target_1d'] > 0).astype(int)
            
            # 4. 分位数目标
            group['return_quantile'] = pd.qcut(
                group['target_1d'].rank(method='first'), 
                5, 
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
            
            targets[(symbol, timeframe)] = group
        
        combined_data = pd.concat(targets.values(), ignore_index=True)
        
        print("✅ 目标变量创建完成")
        return combined_data


class EnhancedModelSystem:
    """增强模型系统"""
    
    def __init__(self):
        self.preprocessor = AdvancedDataPreprocessor()
        self.feature_engineer = EnhancedFeatureEngineering()
        self.ensemble_system = EnsembleModelSystem()
        self.target_creator = ImprovedTargetDefinition()
        
        self.results = {}
        
    def run_enhanced_modeling(self, data_path, factor_path=None):
        """运行增强建模"""
        print("🚀 启动增强模型系统...")
        
        # 1. 加载数据
        print("\n" + "="*50)
        print("第1步: 数据加载和预处理")
        print("="*50)
        
        data = pd.read_csv(data_path)
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        if factor_path:
            factors = pd.read_csv(factor_path)
        else:
            factors = self._create_basic_factors(data)
        
        # 2. 高级数据预处理
        factors_processed = self.preprocessor.advanced_missing_value_handling(factors)
        factors_scaled = self.preprocessor.robust_scaling(factors_processed)
        
        # 3. 特征工程
        print("\n" + "="*50)
        print("第2步: 增强特征工程")
        print("="*50)
        
        # 合并数据进行特征工程 - 重置索引避免重复
        data_reset = data.reset_index(drop=True)
        factors_reset = factors_processed.reset_index(drop=True)
        
        # 确保数据长度一致
        min_length = min(len(data_reset), len(factors_reset))
        data_truncated = data_reset.iloc[:min_length]
        factors_truncated = factors_reset.iloc[:min_length]
        
        combined_data = pd.concat([data_truncated, factors_truncated], axis=1)
        enhanced_data = self.feature_engineer.create_advanced_features(combined_data)
        
        # 4. 目标变量创建
        print("\n" + "="*50)
        print("第3步: 改进目标变量定义")
        print("="*50)
        
        data_with_targets = self.target_creator.create_multiple_targets(enhanced_data)
        
        # 5. 准备建模数据
        feature_columns = [col for col in data_with_targets.columns 
                          if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                       'open', 'high', 'low', 'close', 'volume'] 
                          and not col.startswith('target_') 
                          and not col.startswith('direction_')
                          and not col.startswith('risk_adjusted_')
                          and not col.startswith('return_quantile')]
        
        X = data_with_targets[feature_columns].select_dtypes(include=[np.number])
        y = data_with_targets['target_1d'].dropna()
        
        # 确保索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 6. 集成模型训练
        print("\n" + "="*50)
        print("第4步: 集成模型训练")
        print("="*50)
        
        self.ensemble_system.initialize_models()
        model_scores = self.ensemble_system.train_ensemble(X, y)
        
        # 7. 模型评估
        print("\n" + "="*50)
        print("第5步: 模型性能评估")
        print("="*50)
        
        evaluation_results = self._evaluate_models(X, y)
        
        # 8. 保存结果
        self.results = {
            'model_scores': model_scores,
            'evaluation_results': evaluation_results,
            'feature_count': len(feature_columns),
            'data_shape': X.shape,
            'ensemble_weights': self.ensemble_system.weights
        }
        
        self._generate_enhanced_report()
        
        print("\n🎉 增强建模完成！")
        return self.results
    
    def _create_basic_factors(self, data):
        """创建基础因子"""
        print("   计算基础因子...")
        
        factors_list = []
        
        for (symbol, timeframe), group in data.groupby(['symbol', 'timeframe']):
            group = group.reset_index(drop=True).copy()
            
            # 基础因子
            for period in [5, 10, 20, 60]:
                group[f'return_{period}'] = group['close'].pct_change(period)
                group[f'ma_{period}'] = group['close'].rolling(period).mean()
                group[f'volatility_{period}'] = group['close'].pct_change().rolling(period).std()
            
            # 技术指标
            group['rsi_14'] = self._calculate_rsi(group['close'], 14)
            group['macd'] = group['close'].ewm(span=12).mean() - group['close'].ewm(span=26).mean()
            
            # 布林带
            ma20 = group['close'].rolling(20).mean()
            std20 = group['close'].rolling(20).std()
            group['bb_upper_20'] = ma20 + 2 * std20
            group['bb_lower_20'] = ma20 - 2 * std20
            
            factors_list.append(group)
        
        all_factors = pd.concat(factors_list, ignore_index=True)
        
        factor_columns = [col for col in all_factors.columns 
                         if col not in ['timestamp', 'datetime', 'symbol', 'timeframe', 
                                      'open', 'high', 'low', 'close', 'volume']]
        
        return all_factors[factor_columns]
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _evaluate_models(self, X, y):
        """评估模型性能"""
        print("📊 评估模型性能...")
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        # 评估集成模型
        ensemble_scores = []
        ensemble_direction_acc = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练集成模型
            self.ensemble_system.train_ensemble(X_train, y_train)
            
            # 预测
            pred = self.ensemble_system.predict_ensemble(X_test)
            
            # 评估
            r2 = r2_score(y_test, pred)
            direction_acc = np.mean(np.sign(y_test) == np.sign(pred))
            
            ensemble_scores.append(r2)
            ensemble_direction_acc.append(direction_acc)
        
        results['ensemble'] = {
            'r2_mean': np.mean(ensemble_scores),
            'r2_std': np.std(ensemble_scores),
            'direction_accuracy_mean': np.mean(ensemble_direction_acc),
            'direction_accuracy_std': np.std(ensemble_direction_acc)
        }
        
        print(f"   集成模型 R²: {results['ensemble']['r2_mean']:.4f} ± {results['ensemble']['r2_std']:.4f}")
        print(f"   方向准确率: {results['ensemble']['direction_accuracy_mean']:.4f} ± {results['ensemble']['direction_accuracy_std']:.4f}")
        
        return results
    
    def _generate_enhanced_report(self):
        """生成增强报告"""
        print("\n📊 生成增强模型报告...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_improvements': {
                'advanced_preprocessing': True,
                'enhanced_feature_engineering': True,
                'ensemble_modeling': True,
                'improved_target_definition': True
            },
            'model_performance': self.results['evaluation_results'],
            'ensemble_configuration': {
                'models': list(self.ensemble_system.models.keys()),
                'weights': self.ensemble_system.weights,
                'feature_count': self.results['feature_count']
            },
            'data_statistics': {
                'shape': self.results['data_shape'],
                'features': self.results['feature_count']
            }
        }
        
        with open('enhanced_model_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 增强模型报告已保存到 enhanced_model_report.json")


def main():
    """主函数"""
    print("🔧 增强模型系统启动")
    
    # 初始化系统
    enhanced_system = EnhancedModelSystem()
    
    # 运行增强建模
    data_path = "data/extended/extended_all_data_3years_20250623_215123.csv"
    factor_path = "factors/bitcoin_factors_20250623_215658.csv"
    
    results = enhanced_system.run_enhanced_modeling(
        data_path=data_path,
        factor_path=factor_path
    )
    
    print("\n🎯 增强建模完成！主要改进:")
    print(f"   - 特征数量: {results['feature_count']}")
    print(f"   - 数据形状: {results['data_shape']}")
    print(f"   - 集成模型: {len(results['ensemble_weights'])} 个")
    
    # 显示性能提升
    if 'ensemble' in results['evaluation_results']:
        ensemble_perf = results['evaluation_results']['ensemble']
        print(f"   - 集成R²: {ensemble_perf['r2_mean']:.4f}")
        print(f"   - 方向准确率: {ensemble_perf['direction_accuracy_mean']:.4f}")


if __name__ == "__main__":
    main() 