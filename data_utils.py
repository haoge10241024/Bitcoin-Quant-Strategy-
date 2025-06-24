#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据工具函数
Data Utility Functions

提供数据处理、验证和转换的通用工具函数
Provides common utility functions for data processing, validation and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_ohlc_logic(df: pd.DataFrame) -> Dict[str, any]:
        """
        验证OHLC数据逻辑
        Validate OHLC data logic
        """
        results = {
            'valid': True,
            'issues': [],
            'statistics': {}
        }
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            results['valid'] = False
            results['issues'].append("Missing required OHLC columns")
            return results
        
        # 检查High >= max(Open, Close)
        high_violations = ~(df['high'] >= df[['open', 'close']].max(axis=1))
        if high_violations.any():
            results['valid'] = False
            results['issues'].append(f"High price violations: {high_violations.sum()} records")
        
        # 检查Low <= min(Open, Close)
        low_violations = ~(df['low'] <= df[['open', 'close']].min(axis=1))
        if low_violations.any():
            results['valid'] = False
            results['issues'].append(f"Low price violations: {low_violations.sum()} records")
        
        results['statistics'] = {
            'total_records': len(df),
            'high_violations': high_violations.sum() if not high_violations.empty else 0,
            'low_violations': low_violations.sum() if not low_violations.empty else 0
        }
        
        return results
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
        """
        检测异常值
        Detect outliers
        """
        if method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, any]:
        """
        检查缺失值
        Check missing values
        """
        missing_stats = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / total_rows
            missing_stats[col] = {
                'count': missing_count,
                'ratio': missing_ratio,
                'exceeds_threshold': missing_ratio > threshold
            }
        
        return {
            'statistics': missing_stats,
            'total_missing': df.isnull().sum().sum(),
            'columns_exceeding_threshold': [col for col, stats in missing_stats.items() 
                                          if stats['exceeds_threshold']]
        }


class DataCleaner:
    """数据清洗器"""
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """
        填充缺失值
        Fill missing values
        """
        df_cleaned = df.copy()
        
        if method == 'forward':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif method == 'backward':
            df_cleaned = df_cleaned.fillna(method='bfill')
        elif method == 'interpolate':
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].interpolate()
        elif method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df_cleaned[numeric_columns].mean()
            )
        elif method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df_cleaned[numeric_columns].median()
            )
        
        return df_cleaned
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'zscore', 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        移除异常值
        Remove outliers
        """
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                outliers = DataValidator.detect_outliers(df_cleaned[col], method, threshold)
                df_cleaned = df_cleaned[~outliers]
        
        return df_cleaned
    
    @staticmethod
    def winsorize_data(df: pd.DataFrame, columns: List[str], 
                      lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        缩尾处理
        Winsorize data
        """
        df_winsorized = df.copy()
        
        for col in columns:
            if col in df_winsorized.columns:
                lower_bound = df_winsorized[col].quantile(lower_percentile)
                upper_bound = df_winsorized[col].quantile(upper_percentile)
                df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_winsorized


class DataTransformer:
    """数据转换器"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        计算收益率
        Calculate returns
        """
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
    
    @staticmethod
    def calculate_moving_average(series: pd.Series, window: int, method: str = 'simple') -> pd.Series:
        """
        计算移动平均
        Calculate moving average
        """
        if method == 'simple':
            return series.rolling(window=window).mean()
        elif method == 'exponential':
            return series.ewm(span=window).mean()
        else:
            raise ValueError(f"Unknown moving average method: {method}")
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """
        计算波动率
        Calculate volatility
        """
        return returns.rolling(window=window).std()
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'zscore') -> pd.DataFrame:
        """
        标准化数据
        Normalize data
        """
        df_normalized = df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    mad = np.median(np.abs(df_normalized[col] - median))
                    df_normalized[col] = (df_normalized[col] - median) / mad
        
        return df_normalized


class TimeSeriesUtils:
    """时间序列工具"""
    
    @staticmethod
    def resample_data(df: pd.DataFrame, timestamp_col: str, freq: str, 
                     agg_methods: Dict[str, str] = None) -> pd.DataFrame:
        """
        重采样数据
        Resample data
        """
        if agg_methods is None:
            agg_methods = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        df_resampled = df.copy()
        df_resampled[timestamp_col] = pd.to_datetime(df_resampled[timestamp_col])
        df_resampled = df_resampled.set_index(timestamp_col)
        
        # 应用聚合方法
        resampled_data = {}
        for col, method in agg_methods.items():
            if col in df_resampled.columns:
                if method == 'first':
                    resampled_data[col] = df_resampled[col].resample(freq).first()
                elif method == 'last':
                    resampled_data[col] = df_resampled[col].resample(freq).last()
                elif method == 'max':
                    resampled_data[col] = df_resampled[col].resample(freq).max()
                elif method == 'min':
                    resampled_data[col] = df_resampled[col].resample(freq).min()
                elif method == 'sum':
                    resampled_data[col] = df_resampled[col].resample(freq).sum()
                elif method == 'mean':
                    resampled_data[col] = df_resampled[col].resample(freq).mean()
        
        result_df = pd.DataFrame(resampled_data)
        result_df.reset_index(inplace=True)
        
        return result_df
    
    @staticmethod
    def align_timestamps(df1: pd.DataFrame, df2: pd.DataFrame, 
                        timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        对齐时间戳
        Align timestamps
        """
        # 转换为datetime
        df1_aligned = df1.copy()
        df2_aligned = df2.copy()
        
        df1_aligned[timestamp_col] = pd.to_datetime(df1_aligned[timestamp_col])
        df2_aligned[timestamp_col] = pd.to_datetime(df2_aligned[timestamp_col])
        
        # 找到共同的时间范围
        common_start = max(df1_aligned[timestamp_col].min(), df2_aligned[timestamp_col].min())
        common_end = min(df1_aligned[timestamp_col].max(), df2_aligned[timestamp_col].max())
        
        # 过滤数据
        df1_aligned = df1_aligned[
            (df1_aligned[timestamp_col] >= common_start) & 
            (df1_aligned[timestamp_col] <= common_end)
        ]
        df2_aligned = df2_aligned[
            (df2_aligned[timestamp_col] >= common_start) & 
            (df2_aligned[timestamp_col] <= common_end)
        ]
        
        return df1_aligned, df2_aligned
    
    @staticmethod
    def create_lagged_features(df: pd.DataFrame, columns: List[str], 
                             lags: List[int]) -> pd.DataFrame:
        """
        创建滞后特征
        Create lagged features
        """
        df_lagged = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged


def load_data_with_validation(file_path: str, validation: bool = True) -> pd.DataFrame:
    """
    加载数据并进行验证
    Load data with validation
    """
    try:
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        if validation:
            # 基础验证
            if df.empty:
                warnings.warn("Loaded data is empty")
            
            # 检查缺失值
            missing_info = DataValidator.check_missing_values(df)
            if missing_info['total_missing'] > 0:
                print(f"Warning: Found {missing_info['total_missing']} missing values")
            
            # 如果包含OHLC数据，进行OHLC验证
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_validation = DataValidator.validate_ohlc_logic(df)
                if not ohlc_validation['valid']:
                    print(f"Warning: OHLC validation failed: {ohlc_validation['issues']}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load data from {file_path}: {str(e)}")


def save_data_with_metadata(df: pd.DataFrame, file_path: str, metadata: Dict = None):
    """
    保存数据并添加元数据
    Save data with metadata
    """
    try:
        # 保存数据
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        elif file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # 保存元数据
        if metadata:
            metadata_path = file_path.rsplit('.', 1)[0] + '_metadata.json'
            import json
            
            # 添加基础元数据
            metadata.update({
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'created_at': datetime.now().isoformat(),
                'file_path': file_path
            })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved successfully to {file_path}")
        
    except Exception as e:
        raise Exception(f"Failed to save data to {file_path}: {str(e)}")


# 导出主要函数和类
__all__ = [
    'DataValidator',
    'DataCleaner', 
    'DataTransformer',
    'TimeSeriesUtils',
    'load_data_with_validation',
    'save_data_with_metadata'
] 