#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展OKX数据收集器 - Extended OKX Data Collector
=============================================

获取更长时间跨度的比特币历史数据（2-3年）
为量化分析提供充足的数据基础

Author: Professional Quantitative System
Version: 5.0 Extended Edition
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple, Optional
import os
import sys
from tqdm import tqdm

class ExtendedOKXDataCollector:
    """扩展OKX数据收集器 - 获取长期历史数据"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, passphrase: str = None):
        """
        初始化扩展数据收集器
        
        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key  
            passphrase: OKX API Passphrase
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.okx = None
        
        # 创建数据目录
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "extended").mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('extended_okx_collector.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.initialize_okx()
    
    def initialize_okx(self):
        """初始化OKX连接"""
        try:
            # 使用公共API模式（更稳定）
            self.okx = ccxt.okx({
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # 测试连接
            markets = self.okx.load_markets()
            self.logger.info(f"OKX连接成功，支持 {len(markets)} 个交易对")
            
        except Exception as e:
            self.logger.error(f"OKX初始化失败: {e}")
            raise
    
    def get_extended_ohlcv_data(self, symbol: str = 'BTC/USDT', timeframe: str = '1d', 
                               start_date: str = None, end_date: str = None):
        """
        获取扩展的OHLCV历史数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        try:
            # 设置默认时间范围（3年）
            if not end_date:
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            if not start_date:
                start_date = end_date - timedelta(days=3*365)  # 3年前
            else:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            
            self.logger.info(f"获取 {symbol} {timeframe} 数据，时间范围: {start_date.date()} 到 {end_date.date()}")
            
            # 转换为时间戳
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_timestamp = start_timestamp
            
            # 根据时间周期确定每次请求的数量
            limit_map = {
                '1m': 1000,   # 1分钟
                '5m': 1000,   # 5分钟
                '15m': 1000,  # 15分钟
                '1h': 1000,   # 1小时
                '4h': 1000,   # 4小时
                '1d': 1000,   # 1天
                '1w': 1000,   # 1周
            }
            
            limit = limit_map.get(timeframe, 1000)
            
            # 计算总的预期请求次数
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '1h': 60, 
                '4h': 240, '1d': 1440, '1w': 10080
            }
            
            total_minutes = (end_timestamp - start_timestamp) / (1000 * 60)
            expected_bars = int(total_minutes / timeframe_minutes.get(timeframe, 1440))
            expected_requests = max(1, expected_bars // limit)
            
            self.logger.info(f"预计需要 {expected_requests} 次请求获取约 {expected_bars} 条数据")
            
            # 使用进度条
            with tqdm(total=expected_requests, desc=f"获取{symbol} {timeframe}数据") as pbar:
                request_count = 0
                
                while current_timestamp < end_timestamp:
                    try:
                        # 获取数据
                        ohlcv = self.okx.fetch_ohlcv(
                            symbol, 
                            timeframe, 
                            since=current_timestamp, 
                            limit=limit
                        )
                        
                        if not ohlcv:
                            self.logger.warning(f"没有获取到数据，时间戳: {current_timestamp}")
                            break
                        
                        # 添加到总数据中
                        all_data.extend(ohlcv)
                        
                        # 更新时间戳到最后一条数据的时间
                        if ohlcv:
                            current_timestamp = ohlcv[-1][0] + 1
                        
                        request_count += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'Records': len(all_data),
                            'Latest': datetime.fromtimestamp(ohlcv[-1][0]/1000).strftime('%Y-%m-%d') if ohlcv else 'N/A'
                        })
                        
                        # 避免请求过快
                        time.sleep(0.1)
                        
                        # 如果获取的数据少于limit，说明已经到了最新数据
                        if len(ohlcv) < limit:
                            break
                            
                    except Exception as e:
                        self.logger.error(f"获取数据时出错: {e}")
                        time.sleep(1)  # 出错时等待更长时间
                        continue
            
            if all_data:
                # 去重并排序
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # 添加其他字段
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                self.logger.info(f"成功获取 {len(df)} 条 {symbol} {timeframe} 数据")
                self.logger.info(f"数据时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
                
                return df
            else:
                self.logger.warning(f"未获取到任何 {symbol} {timeframe} 数据")
                return None
                
        except Exception as e:
            self.logger.error(f"获取扩展数据失败: {e}")
            return None
    
    def collect_multi_year_data(self, symbols: List[str] = None, timeframes: List[str] = None, 
                               years: int = 3):
        """
        收集多年历史数据
        
        Args:
            symbols: 交易对列表
            timeframes: 时间周期列表
            years: 历史年数
        """
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT']
        
        if timeframes is None:
            timeframes = ['1d', '4h', '1h']  # 从大到小的时间周期
        
        self.logger.info(f"开始收集 {years} 年历史数据...")
        
        # 计算开始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        all_extended_data = {}
        
        for symbol in symbols:
            self.logger.info(f"处理交易对: {symbol}")
            all_extended_data[symbol] = {}
            
            for timeframe in timeframes:
                self.logger.info(f"获取 {symbol} {timeframe} 数据...")
                
                df = self.get_extended_ohlcv_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if df is not None and not df.empty:
                    all_extended_data[symbol][timeframe] = df
                    
                    # 保存单个文件
                    filename = f"extended_{symbol.replace('/', '_')}_{timeframe}_{years}years.csv"
                    filepath = self.data_dir / "extended" / filename
                    df.to_csv(filepath, index=False)
                    self.logger.info(f"已保存: {filepath}")
                else:
                    self.logger.warning(f"未能获取 {symbol} {timeframe} 数据")
                
                # 请求间隔
                time.sleep(0.5)
        
        # 保存综合数据
        self.save_extended_data(all_extended_data, years)
        
        return all_extended_data
    
    def save_extended_data(self, data: Dict, years: int):
        """保存扩展数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 合并所有数据
            all_dfs = []
            total_records = 0
            
            for symbol, timeframes in data.items():
                for timeframe, df in timeframes.items():
                    if df is not None and not df.empty:
                        all_dfs.append(df)
                        total_records += len(df)
            
            if all_dfs:
                # 合并所有数据
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # 保存合并的CSV文件
                combined_file = self.data_dir / "extended" / f"extended_all_data_{years}years_{timestamp}.csv"
                combined_df.to_csv(combined_file, index=False)
                self.logger.info(f"合并数据已保存: {combined_file}")
                
                # 保存到SQLite数据库
                db_file = self.data_dir / "processed" / f"extended_crypto_data_{years}years.db"
                with sqlite3.connect(db_file) as conn:
                    combined_df.to_sql('extended_ohlcv', conn, if_exists='replace', index=False)
                    self.logger.info(f"数据已保存到数据库: {db_file}")
                
                # 生成数据统计报告
                self.generate_extended_report(data, years, total_records, timestamp)
            
        except Exception as e:
            self.logger.error(f"保存扩展数据失败: {e}")
    
    def generate_extended_report(self, data: Dict, years: int, total_records: int, timestamp: str):
        """生成扩展数据报告"""
        try:
            report = {
                "收集时间": datetime.now().isoformat(),
                "数据源": "OKX API - Extended Collection",
                "历史年数": years,
                "总记录数": total_records,
                "数据统计": {},
                "时间范围": {},
                "数据质量": {}
            }
            
            for symbol, timeframes in data.items():
                report["数据统计"][symbol] = {}
                report["时间范围"][symbol] = {}
                report["数据质量"][symbol] = {}
                
                for timeframe, df in timeframes.items():
                    if df is not None and not df.empty:
                        # 基础统计
                        report["数据统计"][symbol][timeframe] = {
                            "记录数": len(df),
                            "开始时间": df['datetime'].min().isoformat(),
                            "结束时间": df['datetime'].max().isoformat(),
                            "时间跨度天数": (df['datetime'].max() - df['datetime'].min()).days
                        }
                        
                        # 价格统计
                        report["时间范围"][symbol][timeframe] = {
                            "最高价": f"${df['high'].max():,.2f}",
                            "最低价": f"${df['low'].min():,.2f}",
                            "最新价": f"${df['close'].iloc[-1]:,.2f}",
                            "平均价": f"${df['close'].mean():,.2f}"
                        }
                        
                        # 数据质量
                        missing_count = df.isnull().sum().sum()
                        report["数据质量"][symbol][timeframe] = {
                            "完整率": f"{(1 - missing_count/len(df)) * 100:.2f}%",
                            "缺失值": int(missing_count),
                            "重复值": int(df.duplicated().sum())
                        }
            
            # 保存报告
            report_file = f"extended_data_report_{years}years_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"扩展数据报告已保存: {report_file}")
            
            # 打印摘要
            self.logger.info("=" * 60)
            self.logger.info(f"扩展数据收集完成！({years}年历史数据)")
            self.logger.info(f"总记录数: {total_records:,} 条")
            for symbol in data.keys():
                symbol_total = sum(len(df) for df in data[symbol].values() if df is not None)
                self.logger.info(f"{symbol}: {symbol_total:,} 条记录")
            self.logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成扩展报告失败: {e}")
            return None

def main():
    """主函数"""
    print("🚀 扩展OKX数据收集器 - 获取长期历史数据")
    print("=" * 60)
    
    # 询问历史年数
    try:
        years_input = input("请输入要收集的历史年数 (建议3-5年，默认3年): ").strip()
        years = int(years_input) if years_input else 3
        years = max(1, min(years, 10))  # 限制在1-10年之间
    except ValueError:
        years = 3
        print("输入无效，使用默认值3年")
    
    print(f"将收集 {years} 年的历史数据...")
    
    # 创建收集器
    collector = ExtendedOKXDataCollector()
    
    try:
        # 收集扩展数据
        print(f"\n📊 开始收集 {years} 年历史数据...")
        data = collector.collect_multi_year_data(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframes=['1d', '4h', '1h'],  # 从日线开始，避免数据量过大
            years=years
        )
        
        print(f"\n🎉 {years}年历史数据收集完成！")
        print("📁 请查看 data/extended/ 目录下的文件")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 收集失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 程序执行成功！")
    else:
        print("\n❌ 程序执行失败！") 