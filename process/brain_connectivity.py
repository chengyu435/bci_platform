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

def clean_channel_name(name):
    return name.replace(".", "").upper()

def get_1020_positions():
    return {
        "FP1": (-1, 4), "FP2": (1, 4),

        "F7": (-3, 2), "F3": (-1, 2), "FZ": (0, 2), "F4": (1, 2), "F8": (3, 2),

        "T7": (-4, 0), "C3": (-1, 0), "CZ": (0, 0), "C4": (1, 0), "T8": (4, 0),

        "P7": (-3, -2), "P3": (-1, -2), "PZ": (0, -2), "P4": (1, -2), "P8": (3, -2),

        "O1": (-1, -4), "O2": (1, -4),
    }

def build_channel_position(channel_names):
    """
    EDF通道 → 脑区坐标
    """

    pos_map = get_1020_positions()

    channel_pos = {}

    for i, ch in enumerate(channel_names):

        ch = clean_channel_name(ch)

        if ch in pos_map:
            channel_pos[i] = pos_map[ch]

    return channel_pos




def plot_brain_connectivity_simple(matrix, channel_names, threshold=0.4):

    pos_map = get_1020_positions()
    ch_pos = build_channel_position(channel_names)

    plt.figure(figsize=(8, 8))

    # 画节点
    for i, pos in ch_pos.items():
        x, y = pos
        name = clean_channel_name(channel_names[i])

        plt.scatter(x, y, c='black')
        plt.text(x, y+0.1, name, ha='center', fontsize=8)

    # 画连接
    n = matrix.shape[0]

    for i in range(n):
        if i not in ch_pos:
            continue

        for j in range(i+1, n):

            if j not in ch_pos:
                continue

            w = matrix[i, j]

            if abs(w) < threshold:
                continue

            x1, y1 = ch_pos[i]
            x2, y2 = ch_pos[j]

            plt.plot([x1, x2], [y1, y2], linewidth=abs(w)*2)

    plt.title("EEG Functional Connectivity (10-20)")
    plt.axis("off")
    plt.show()

def plot_brain_connectivity(
    matrix,
    channel_names,
    min_threshold=0.4,
    max_threshold=1.0,
    layout="1020",
    show_weight=False,
    cmap_name="viridis",
):
    """
    统一脑网络可视化函数（推荐唯一入口）
    """

    import matplotlib.pyplot as plt
    import numpy as np

    def clean(name):
        return name.replace(".", "").upper()

    # =========================
    # 1. 选择布局
    # =========================
    if layout == "1020":
        pos_map = get_1020_positions()

        ch_pos = {}
        for i, ch in enumerate(channel_names):
            ch = clean(ch)
            if ch in pos_map:
                ch_pos[i] = pos_map[ch]

    elif layout == "circle":
        n = len(channel_names)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        ch_pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    else:
        raise ValueError("layout must be '1020' or 'circle'")

    # =========================
    # 2. 颜色映射
    # =========================
    weights = [
    abs(matrix[i, j])
    for i in range(matrix.shape[0])
    for j in range(i+1, matrix.shape[1])
    if min_threshold <= abs(matrix[i, j]) <= max_threshold
    ]
    if len(weights) == 0:
        weights = [min_threshold, max_threshold]

    if len(weights) == 0:
        weights = [0, 1]

    vmin, vmax = min(weights), max(weights)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps[cmap_name]

    fig, ax = plt.subplots(figsize=(8, 8))

    # ===== 画节点 =====
    for i, pos in ch_pos.items():
        x, y = pos
        name = clean(channel_names[i])

        ax.scatter(x, y, c="black", s=40)
        ax.text(x, y+0.1, name, ha="center", fontsize=8)
    n = matrix.shape[0]         
    # ===== 画连接 =====
    for i in range(n):
        for j in range(i+1, n):

            if i not in ch_pos or j not in ch_pos:
                continue

            w = matrix[i, j]
            weight = abs(w)

            if weight < min_threshold:
                continue

            if weight > max_threshold:
                continue

            x1, y1 = ch_pos[i]
            x2, y2 = ch_pos[j]

            color = cmap(norm(abs(w)))

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

            if show_weight:
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx, my, f"{w:.2f}", fontsize=7)

    # ===== colorbar =====
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Connectivity strength")

    ax.set_title(f"EEG Connectivity ({layout})")
    ax.axis("off")

    plt.tight_layout()
    plt.show()