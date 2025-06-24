# 🚀 部署指南
# Deployment Guide

本文档提供了BTC Qlib量化策略项目的完整部署指南。

## 📋 目录
- [环境要求](#环境要求)
- [快速部署](#快速部署)
- [详细安装步骤](#详细安装步骤)
- [配置说明](#配置说明)
- [验证部署](#验证部署)
- [生产环境部署](#生产环境部署)
- [故障排除](#故障排除)

## 🔧 环境要求

### 系统要求
- **操作系统**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python版本**: 3.8 - 3.11
- **内存**: 最少8GB，推荐16GB+
- **存储**: 最少10GB可用空间
- **网络**: 稳定的互联网连接（用于数据获取）

### 软件依赖
- Git
- Python 3.8+
- pip 或 conda
- (可选) Docker

## ⚡ 快速部署

### 方式一：使用pip安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/btc-qlib-strategy.git
cd btc-qlib-strategy

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python examples/basic_usage.py
```

### 方式二：使用conda安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/btc-qlib-strategy.git
cd btc-qlib-strategy

# 2. 创建conda环境
conda create -n btc-qlib python=3.9
conda activate btc-qlib

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python examples/basic_usage.py
```

### 方式三：使用Docker部署

```bash
# 1. 克隆项目
git clone https://github.com/your-username/btc-qlib-strategy.git
cd btc-qlib-strategy

# 2. 构建Docker镜像
docker build -t btc-qlib-strategy .

# 3. 运行容器
docker run -it --rm -v $(pwd)/data:/app/data btc-qlib-strategy
```

## 📝 详细安装步骤

### 步骤1：环境准备

#### Linux (Ubuntu/Debian)
```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装Python和相关工具
sudo apt install python3 python3-pip python3-venv git -y

# 安装系统依赖
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
```

#### macOS
```bash
# 安装Homebrew (如果尚未安装)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Python
brew install python@3.9 git
```

#### Windows
1. 从 [python.org](https://www.python.org/downloads/) 下载并安装Python 3.9+
2. 从 [git-scm.com](https://git-scm.com/download/win) 下载并安装Git
3. 确保Python和Git已添加到系统PATH

### 步骤2：项目克隆和设置

```bash
# 克隆项目
git clone https://github.com/your-username/btc-qlib-strategy.git
cd btc-qlib-strategy

# 检查项目结构
ls -la
```

### 步骤3：虚拟环境设置

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 升级pip
pip install --upgrade pip
```

### 步骤4：依赖安装

```bash
# 安装核心依赖
pip install -r requirements.txt

# 验证关键包安装
python -c "import pandas, numpy, sklearn, qlib; print('核心依赖安装成功')"
```

### 步骤5：Qlib框架配置

```bash
# 初始化Qlib
python -c "
import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)
print('Qlib初始化成功')
"
```

## ⚙️ 配置说明

### 数据配置

编辑 `config/data_config.yaml`：

```yaml
# 数据源配置
data_sources:
  primary: "okx"
  backup: "binance"

# OKX API配置（可选）
okx:
  api_key: "your_api_key"        # 替换为您的API密钥
  secret_key: "your_secret"      # 替换为您的密钥
  passphrase: "your_passphrase"  # 替换为您的口令
  sandbox: false                 # 生产环境设为false
```

### 模型配置

编辑 `config/model_config.yaml`：

```yaml
# 模型选择
models:
  ensemble:
    enabled: true
    method: "weighted_average"
  
  individual_models:
    random_forest:
      enabled: true
      n_estimators: 200
    gradient_boosting:
      enabled: true
      n_estimators: 150
```

### 策略配置

编辑 `config/strategy_config.yaml`：

```yaml
# 策略参数
strategy:
  name: "BTC_Multi_Factor_Strategy"
  version: "2.0"

# 信号生成
signal:
  prediction_threshold: 0.6
  signal_smoothing: true
```

## ✅ 验证部署

### 基础验证

```bash
# 1. 运行基础测试
python tests/test_data_collection.py

# 2. 运行示例程序
python examples/basic_usage.py

# 3. 验证核心功能
python -c "
from src.data_collection.extended_okx_data_collector import ExtendedOKXDataCollector
from src.factor_engineering.professional_factor_engineering import ProfessionalFactorEngineering
print('所有核心模块导入成功')
"
```

### 完整功能测试

```bash
# 运行完整测试套件
python -m pytest tests/ -v

# 运行性能测试
python examples/advanced_features.py
```

### 数据收集测试

```bash
# 测试数据收集功能
python -c "
from src.data_collection.extended_okx_data_collector import ExtendedOKXDataCollector
collector = ExtendedOKXDataCollector()
# 注意：需要配置API密钥才能实际获取数据
print('数据收集器初始化成功')
"
```

## 🏭 生产环境部署

### 服务器配置

#### 推荐配置
- **CPU**: 4核心以上
- **内存**: 16GB以上
- **存储**: SSD 50GB以上
- **网络**: 1Gbps以上

#### 系统优化

```bash
# 设置系统限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化Python性能
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
```

### 使用systemd服务

创建服务文件 `/etc/systemd/system/btc-qlib.service`：

```ini
[Unit]
Description=BTC Qlib Strategy Service
After=network.target

[Service]
Type=simple
User=btc-qlib
WorkingDirectory=/opt/btc-qlib-strategy
Environment=PATH=/opt/btc-qlib-strategy/venv/bin
ExecStart=/opt/btc-qlib-strategy/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 重新加载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start btc-qlib

# 设置开机自启
sudo systemctl enable btc-qlib

# 查看状态
sudo systemctl status btc-qlib
```

### 使用Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  btc-qlib:
    build: .
    container_name: btc-qlib-strategy
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

部署：

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 监控和日志

#### 日志配置

创建 `config/logging.yaml`：

```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.FileHandler
    filename: logs/btc-qlib.log
    level: DEBUG
    formatter: default
root:
  level: DEBUG
  handlers: [console, file]
```

#### 性能监控

```bash
# 安装监控工具
pip install psutil prometheus-client

# 启动监控
python scripts/monitoring.py
```

## 🔧 故障排除

### 常见问题

#### 1. 依赖安装失败

```bash
# 问题：pip安装失败
# 解决方案：
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

#### 2. Qlib初始化错误

```bash
# 问题：Qlib数据初始化失败
# 解决方案：
python -c "
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
"
```

#### 3. 内存不足

```bash
# 问题：运行时内存不足
# 解决方案：调整批处理大小
export BATCH_SIZE=100
export MAX_WORKERS=2
```

#### 4. API连接失败

```bash
# 问题：无法连接到OKX API
# 解决方案：
# 1. 检查网络连接
# 2. 验证API密钥
# 3. 检查API限制
```

### 日志分析

```bash
# 查看错误日志
tail -f logs/btc-qlib.log | grep ERROR

# 查看性能日志
tail -f logs/performance.log

# 分析内存使用
python -c "
import psutil
print(f'内存使用: {psutil.virtual_memory().percent}%')
"
```

### 性能优化

#### 1. 数据处理优化

```python
# 使用多进程处理
import multiprocessing
n_cores = multiprocessing.cpu_count()
print(f"可用CPU核心: {n_cores}")

# 在配置中设置
# config/model_config.yaml
# n_jobs: -1  # 使用所有核心
```

#### 2. 内存优化

```python
# 分批处理数据
def process_data_in_batches(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]
```

#### 3. 磁盘I/O优化

```bash
# 使用SSD存储
# 启用数据压缩
# 定期清理临时文件
find ./data/temp -type f -mtime +7 -delete
```

## 📊 部署验证清单

- [ ] Python环境正确安装
- [ ] 所有依赖包成功安装
- [ ] 核心模块可以正常导入
- [ ] 配置文件正确设置
- [ ] 基础功能测试通过
- [ ] 数据收集功能正常
- [ ] 因子工程模块运行正常
- [ ] 模型训练功能正常
- [ ] 回测分析功能正常
- [ ] 日志系统配置正确
- [ ] 监控系统运行正常
- [ ] 性能指标在合理范围内

## 🆘 获取帮助

如果遇到部署问题，可以通过以下方式获取帮助：

1. **查看文档**: 详细阅读项目文档
2. **检查日志**: 分析错误日志信息
3. **社区支持**: 在GitHub Issues中提问
4. **联系维护者**: 发送邮件至项目维护者

## 📈 后续维护

### 定期维护任务

```bash
# 每日任务
# 1. 检查系统状态
systemctl status btc-qlib

# 2. 查看日志
tail -n 100 logs/btc-qlib.log

# 3. 检查磁盘空间
df -h

# 每周任务
# 1. 更新依赖包
pip list --outdated

# 2. 清理临时文件
find ./data/temp -type f -mtime +7 -delete

# 3. 备份重要数据
tar -czf backup_$(date +%Y%m%d).tar.gz data/ config/

# 每月任务
# 1. 系统安全更新
sudo apt update && sudo apt upgrade

# 2. 性能分析
python scripts/performance_analysis.py

# 3. 策略效果评估
python scripts/strategy_evaluation.py
```

---

**部署成功后，您就可以开始使用BTC Qlib量化策略系统了！** 🎉 