# BCI Platform

## 项目简介

BCI Platform 是一个基于 Python 开发的国产化脑机接口（Brain-Computer Interface, BCI）信号处理平台。

本项目以**自主开发、模块化设计、可扩展**为目标，尽量避免依赖国外脑机接口专用工具箱（如 MNE、EEGLAB Python 接口等），核心算法均采用 NumPy 等基础科学计算库实现，为后续脑机接口算法研究和国产化软件开发提供基础平台。

目前已完成：

- EDF/BDF 文件读取
- EEG 数据统一格式封装
- 基础数据检查
- 去均值、去趋势
- FFT 带通滤波
- FFT 陷波滤波
- 固定窗口分段
- Marker 分段
- FFT 频谱分析
- 常见脑电频段能量计算
- 脑功能连接分析（Correlation）
- 国际 10-20 / 10-10 电极布局可视化

---

# 项目结构

```text
bci_platform/
│
├── data/                  # 测试数据
│
├── io/
│   └── edf_bdf_reader.py  # EDF/BDF读取
│
├── process/
│   ├── basic_data_process.py    # EEG基础处理
│   └── brain_connectivity.py    # 脑功能连接分析
│
├── demo.ipynb             # Jupyter 示例
│
└── README.md
```

---

# 运行环境

建议 Python 3.10 及以上版本。

目前仅依赖：

```
numpy
matplotlib
```

安装：

```bash
pip install numpy matplotlib
```

---

# 快速开始

读取 EDF 文件：

```python
from bci_platform.io.edf_bdf_reader import EDFBDFReader

reader = EDFBDFReader("data/S001R01.edf")
eeg = reader.read(preload=True)
```

基础滤波：

```python
from bci_platform.process.basic_data_process import basic_filter_pipeline

filtered = basic_filter_pipeline(
    eeg.data[:64],
    fs=eeg.main_sampling_rate
)
```

脑功能连接分析：

```python
from bci_platform.process.brain_connectivity import (
    compute_band_correlation,
    threshold_matrix,
    plot_brain_connectivity,
)

corr = compute_band_correlation(
    filtered,
    eeg.main_sampling_rate,
    band=(8,13)
)

corr = threshold_matrix(corr, threshold=0.4)

plot_brain_connectivity(
    corr,
    eeg.channel_names[:64]
)
```

---

# 测试数据

本项目目前采用 PhysioNet 提供的 **EEG Motor Movement/Imagery Dataset** 作为测试数据。

数据下载地址：

https://physionet.org/content/eegmmidb/1.0.0/

## 数据简介

该数据集由 **Wadsworth Center（New York State Department of Health）** 基于 **BCI2000** 系统采集，是国际上广泛使用的脑机接口公开数据集之一。

主要特点：

- 64 通道 EEG
- 国际 **10-10 电极系统**
- EDF+ 数据格式
- 采样率 **160 Hz**
- 包含 Marker 注释通道
- 支持运动执行（Motor Execution）和运动想象（Motor Imagery）研究

实验包含：

- 睁眼静息
- 闭眼静息
- 左手运动/运动想象
- 右手运动/运动想象
- 双手运动/运动想象
- 双脚运动/运动想象

其中事件标记包括：

| Marker | 含义 |
|---------|------|
| T0 | 静息（Rest） |
| T1 | 左手或双手任务（根据实验类型） |
| T2 | 右手或双脚任务（根据实验类型） |

每个 EDF 文件包含：

- 64 路 EEG 信号
- 1 路 Annotation 通道
- EEG 信号采用国际 10-10 电极布局
- EDF Annotation 与 `.event` 文件具有一致的事件信息

本项目目前主要使用其中的 **S001R01.edf** 文件作为基础功能测试数据，用于验证：

- EDF 文件读取
- EEG 数据解析
- 基础滤波
- 频谱分析
- Marker 分段
- 脑功能连接分析
- 电极拓扑可视化

---

# 开发原则

本项目坚持以下原则：

- 尽可能采用自主实现算法
- 尽量减少国外脑机接口专用工具箱依赖
- 保持模块化设计
- 保证接口统一
- 方便后续扩展更多脑机接口算法

---

# License

仅用于科研学习与算法研究。
