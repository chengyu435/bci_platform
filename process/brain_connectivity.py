# -*- coding: utf-8 -*-
"""
脑功能连接分析模块（国产化轻量版）

不依赖：
    - mne
    - scipy
    - sklearn
仅使用 numpy

功能：
1. Pearson相关矩阵
2. 频段过滤后相关
3. 简化PLV（基于FFT相位）
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Pearson相关连接
# =========================================================

def compute_correlation_matrix(data):
    """
    计算通道间 Pearson 相关系数

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples]

    Returns
    -------
    corr : np.ndarray
        shape = [n_channels, n_channels]
    """

    data = np.asarray(data, dtype=np.float64)

    # 去均值
    data = data - np.mean(data, axis=1, keepdims=True)

    # 标准化
    std = np.std(data, axis=1, keepdims=True) + 1e-12
    data_norm = data / std

    # 相关矩阵
    corr = np.dot(data_norm, data_norm.T) / data.shape[1]

    return corr


# =========================================================
# 2. FFT带通（用于连接前滤波）
# =========================================================

def fft_bandpass(data, fs, low=8, high=13):
    """
    简单FFT带通（alpha/beta等）

    Parameters
    ----------
    data : [channels, samples]
    """

    n_ch, n_s = data.shape

    freqs = np.fft.rfftfreq(n_s, d=1/fs)
    fft_data = np.fft.rfft(data, axis=1)

    mask = (freqs >= low) & (freqs <= high)

    fft_data = fft_data * mask[None, :]

    return np.fft.irfft(fft_data, n=n_s, axis=1)


# =========================================================
# 3. 频段相关连接
# =========================================================

def compute_band_correlation(data, fs, band=(8, 13)):
    """
    计算某频段内的功能连接（alpha/beta等）

    思路：
        先滤波 -> 再相关

    Returns
    -------
    corr_matrix
    """

    filtered = fft_bandpass(data, fs, band[0], band[1])
    return compute_correlation_matrix(filtered)


# =========================================================
# 4. 简化 PLV（Phase Locking Value）
# =========================================================

def compute_plv_matrix(data):
    """
    简化PLV（FFT相位版）

    Parameters
    ----------
    data : [channels, samples]

    Returns
    -------
    plv : [channels, channels]
    """

    data = np.asarray(data)
    n_ch, n_s = data.shape

    fft_data = np.fft.rfft(data, axis=1)

    phase = np.angle(fft_data)

    plv = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(n_ch):

            phase_diff = phase[i] - phase[j]

            plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


# =========================================================
# 5. 结果阈值化（构建脑网络）
# =========================================================

def threshold_matrix(mat, threshold=0.5):
    """
    构建稀疏脑网络
    """
    mat = np.copy(mat)
    mat[np.abs(mat) < threshold] = 0
    return mat



# =========================================================
# 6. 脑网络可视化
# =========================================================
def plot_brain_network(matrix, threshold=0.4, title="Brain Connectivity Network"):
    """
    简化脑网络可视化（国产化版本）

    参数：
        matrix: [n_channels, n_channels]
        threshold: 连接阈值
    """

    n = matrix.shape[0]

    # =========================
    # 1. 构建节点位置（圆形布局）
    # =========================
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = np.cos(angles)
    y = np.sin(angles)

    # =========================
    # 2. 画节点
    # =========================
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=50, c='black')

    # =========================
    # 3. 画连接
    # =========================
    for i in range(n):
        for j in range(i + 1, n):

            w = matrix[i, j]

            if abs(w) > threshold:

                plt.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    linewidth=abs(w) * 2,
                    alpha=0.6
                )

                # 在中点标数值
                mx = (x[i] + x[j]) / 2
                my = (y[i] + y[j]) / 2

                plt.text(
                    mx, my,
                    f"{w:.2f}",
                    fontsize=6,
                    ha='center'
                )

    # =========================
    # 4. 美化
    # =========================
    for i in range(n):
        plt.text(x[i] * 1.1, y[i] * 1.1, str(i), fontsize=8)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()