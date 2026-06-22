# -*- coding: utf-8 -*-
"""
基础脑电数据处理模块

功能：
1. 数据基础检查
2. 去均值
3. 去趋势
4. FFT 带通滤波
5. FFT 陷波滤波
6. 固定时间窗分段
7. 数值 Marker 分段
8. 幅值频谱分析
9. 频段能量计算

说明：
    本模块不依赖 MNE、pyEDFlib、scipy.signal 等现成脑机接口工具箱。
    当前仅依赖 numpy。
"""

import numpy as np


# ============================================================
# 1. 数据基础检查
# ============================================================

def check_eeg_data(data):
    """
    检查脑电数据基本状态。

    Parameters
    ----------
    data : np.ndarray
        脑电数据，shape = [n_channels, n_samples]

    Returns
    -------
    info : dict
        数据检查结果
    """

    data = np.asarray(data)

    info = {
        "shape": data.shape,
        "ndim": data.ndim,
        "dtype": data.dtype,
        "has_nan": np.isnan(data).any(),
        "has_inf": np.isinf(data).any(),
        "min": np.nanmin(data),
        "max": np.nanmax(data),
        "mean": np.nanmean(data),
        "std": np.nanstd(data),
    }

    return info


def print_eeg_data_info(data, name="EEG data"):
    """
    打印脑电数据基本信息。
    """

    info = check_eeg_data(data)

    print(f"========== {name} ==========")
    print("shape:", info["shape"])
    print("ndim:", info["ndim"])
    print("dtype:", info["dtype"])
    print("has_nan:", info["has_nan"])
    print("has_inf:", info["has_inf"])
    print("min:", info["min"])
    print("max:", info["max"])
    print("mean:", info["mean"])
    print("std:", info["std"])


# ============================================================
# 2. 基础预处理
# ============================================================

def remove_mean(data, axis=-1):
    """
    去均值。

    对脑电数据而言，通常沿时间维度去均值。
    默认 data shape = [n_channels, n_samples]。
    """

    data = np.asarray(data, dtype=np.float64)
    return data - np.mean(data, axis=axis, keepdims=True)


def detrend_linear(data):
    """
    简单线性去趋势。

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples] 或 [n_samples]

    Returns
    -------
    detrended : np.ndarray
        去趋势后的数据
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        x = np.arange(data.shape[0])
        p = np.polyfit(x, data, deg=1)
        trend = np.polyval(p, x)
        return data - trend

    elif data.ndim == 2:
        n_channels, n_samples = data.shape
        x = np.arange(n_samples)

        detrended = np.zeros_like(data, dtype=np.float64)

        for ch in range(n_channels):
            p = np.polyfit(x, data[ch], deg=1)
            trend = np.polyval(p, x)
            detrended[ch] = data[ch] - trend

        return detrended

    else:
        raise ValueError("detrend_linear 仅支持 1维或2维数据")


# ============================================================
# 3. FFT 滤波
# ============================================================

def fft_bandpass_filter(data, fs, low_freq=0.5, high_freq=40.0):
    """
    FFT 带通滤波。

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples] 或 [n_samples]

    fs : float
        采样率

    low_freq : float
        低截止频率

    high_freq : float
        高截止频率

    Returns
    -------
    filtered_data : np.ndarray
        滤波后数据
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data_2d = data[np.newaxis, :]
        squeeze_output = True
    elif data.ndim == 2:
        data_2d = data
        squeeze_output = False
    else:
        raise ValueError("fft_bandpass_filter 仅支持 1维或2维数据")

    n_channels, n_samples = data_2d.shape

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_data = np.fft.rfft(data_2d, axis=1)

    mask = (freqs >= low_freq) & (freqs <= high_freq)

    fft_data_filtered = fft_data * mask[np.newaxis, :]

    filtered_data = np.fft.irfft(
        fft_data_filtered,
        n=n_samples,
        axis=1
    )

    if squeeze_output:
        return filtered_data[0]

    return filtered_data


def fft_notch_filter(data, fs, notch_freq=50.0, bandwidth=2.0):
    """
    FFT 陷波滤波。

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples] 或 [n_samples]

    fs : float
        采样率

    notch_freq : float
        陷波中心频率，例如 50 Hz

    bandwidth : float
        陷波带宽，例如 2 表示去除 49~51 Hz

    Returns
    -------
    filtered_data : np.ndarray
        陷波后数据
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data_2d = data[np.newaxis, :]
        squeeze_output = True
    elif data.ndim == 2:
        data_2d = data
        squeeze_output = False
    else:
        raise ValueError("fft_notch_filter 仅支持 1维或2维数据")

    n_channels, n_samples = data_2d.shape

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_data = np.fft.rfft(data_2d, axis=1)

    low = notch_freq - bandwidth / 2.0
    high = notch_freq + bandwidth / 2.0

    mask = ~((freqs >= low) & (freqs <= high))

    fft_data_filtered = fft_data * mask[np.newaxis, :]

    filtered_data = np.fft.irfft(
        fft_data_filtered,
        n=n_samples,
        axis=1
    )

    if squeeze_output:
        return filtered_data[0]

    return filtered_data


def basic_filter_pipeline(
    data,
    fs,
    low_freq=0.5,
    high_freq=40.0,
    notch_freq=50.0,
    notch_bandwidth=2.0,
    do_detrend=True,
):
    """
    基础滤波流程。

    处理顺序：
        1. 去均值
        2. 可选线性去趋势
        3. 陷波
        4. 带通

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples]

    fs : float
        采样率

    Returns
    -------
    filtered_data : np.ndarray
        滤波后数据
    """

    processed = remove_mean(data, axis=-1)

    if do_detrend:
        processed = detrend_linear(processed)

    processed = fft_notch_filter(
        processed,
        fs,
        notch_freq=notch_freq,
        bandwidth=notch_bandwidth
    )

    processed = fft_bandpass_filter(
        processed,
        fs,
        low_freq=low_freq,
        high_freq=high_freq
    )

    return processed


# ============================================================
# 4. 数据分段
# ============================================================

def segment_by_fixed_window(data, fs, window_time=2.0, step_time=2.0):
    """
    固定时间窗分段。

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples]

    fs : float
        采样率

    window_time : float
        每段长度，单位秒

    step_time : float
        步长，单位秒

    Returns
    -------
    epochs : np.ndarray
        shape = [n_epochs, n_channels, n_epoch_samples]

    start_samples : np.ndarray
        每段起始采样点
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("segment_by_fixed_window 要求输入为二维数据 [n_channels, n_samples]")

    window_samples = int(round(window_time * fs))
    step_samples = int(round(step_time * fs))

    n_channels, n_samples = data.shape

    epochs = []
    start_samples = []

    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        epochs.append(data[:, start:end])
        start_samples.append(start)

    if len(epochs) == 0:
        return np.empty((0, n_channels, window_samples)), np.array([])

    epochs = np.stack(epochs, axis=0)

    return epochs, np.array(start_samples)


def detect_marker_onsets(marker_data):
    """
    检测数值型 marker 的上升沿。

    marker 示例：
        0 0 0 1 1 1 0 0 2 2 2 0

    检测结果：
        事件位置为 1 和 2 第一次出现的位置。

    Parameters
    ----------
    marker_data : np.ndarray
        shape = [n_samples]

    Returns
    -------
    event_samples : np.ndarray
        事件采样点

    labels : np.ndarray
        事件标签
    """

    marker_data = np.asarray(marker_data)
    marker_int = np.round(marker_data).astype(int)

    marker_prev = np.concatenate([[0], marker_int[:-1]])

    event_mask = (marker_prev == 0) & (marker_int != 0)

    event_samples = np.where(event_mask)[0]
    labels = marker_int[event_samples]

    return event_samples, labels


def segment_by_marker(data, marker_data, fs, pre_time=0.0, post_time=2.0):
    """
    根据数值型 marker 分段。

    Parameters
    ----------
    data : np.ndarray
        shape = [n_channels, n_samples]

    marker_data : np.ndarray
        shape = [n_samples]

    fs : float
        采样率

    pre_time : float
        事件前时间，单位秒

    post_time : float
        事件后时间，单位秒

    Returns
    -------
    epochs : np.ndarray
        shape = [n_epochs, n_channels, n_epoch_samples]

    labels : np.ndarray
        shape = [n_epochs]

    event_samples : np.ndarray
        shape = [n_epochs]
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("segment_by_marker 要求输入为二维数据 [n_channels, n_samples]")

    event_samples, labels = detect_marker_onsets(marker_data)

    pre_samples = int(round(pre_time * fs))
    post_samples = int(round(post_time * fs))
    epoch_samples = pre_samples + post_samples

    n_channels, n_total_samples = data.shape

    epochs = []
    valid_labels = []
    valid_event_samples = []

    for event_sample, label in zip(event_samples, labels):
        start = event_sample - pre_samples
        end = event_sample + post_samples

        if start < 0:
            continue

        if end > n_total_samples:
            continue

        epoch = data[:, start:end]

        if epoch.shape[1] == epoch_samples:
            epochs.append(epoch)
            valid_labels.append(label)
            valid_event_samples.append(event_sample)

    if len(epochs) == 0:
        return (
            np.empty((0, n_channels, epoch_samples)),
            np.array([]),
            np.array([])
        )

    epochs = np.stack(epochs, axis=0)

    return (
        epochs,
        np.array(valid_labels),
        np.array(valid_event_samples)
    )


# ============================================================
# 5. 频谱分析
# ============================================================

def compute_amplitude_spectrum(signal, fs, remove_dc=True, use_window=True):
    """
    计算单通道幅值频谱。

    Parameters
    ----------
    signal : np.ndarray
        shape = [n_samples]

    fs : float
        采样率

    remove_dc : bool
        是否去均值

    use_window : bool
        是否加汉宁窗

    Returns
    -------
    freqs : np.ndarray
        频率轴

    amplitude : np.ndarray
        幅值谱
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("compute_amplitude_spectrum 仅支持单通道一维数据")

    if remove_dc:
        signal = signal - np.mean(signal)

    n_samples = len(signal)

    if use_window:
        window = np.hanning(n_samples)
        signal = signal * window

    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    amplitude = np.abs(fft_result) * 2.0 / n_samples

    return freqs, amplitude


def compute_power_spectrum(signal, fs, remove_dc=True, use_window=True):
    """
    计算单通道功率谱。

    Returns
    -------
    freqs : np.ndarray
    power : np.ndarray
    """

    freqs, amplitude = compute_amplitude_spectrum(
        signal,
        fs,
        remove_dc=remove_dc,
        use_window=use_window
    )

    power = amplitude ** 2

    return freqs, power


def find_top_frequencies(freqs, spectrum, fmin=0.5, fmax=40.0, top_n=10):
    """
    查找指定频段内谱峰最大的若干频率。

    Parameters
    ----------
    freqs : np.ndarray
        频率轴

    spectrum : np.ndarray
        幅值谱或功率谱

    fmin : float
        起始频率

    fmax : float
        终止频率

    top_n : int
        返回数量

    Returns
    -------
    top_freqs : np.ndarray

    top_values : np.ndarray
    """

    freqs = np.asarray(freqs)
    spectrum = np.asarray(spectrum)

    valid_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    valid_freqs = freqs[valid_idx]
    valid_values = spectrum[valid_idx]

    if len(valid_values) == 0:
        return np.array([]), np.array([])

    order = np.argsort(valid_values)[-top_n:][::-1]

    top_freqs = valid_freqs[order]
    top_values = valid_values[order]

    return top_freqs, top_values


def band_power(freqs, power, fmin, fmax):
    """
    计算指定频段能量。

    Parameters
    ----------
    freqs : np.ndarray
        频率轴

    power : np.ndarray
        功率谱

    fmin : float
        频段下限

    fmax : float
        频段上限

    Returns
    -------
    bp : float
        频段能量
    """

    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    if len(idx) == 0:
        return 0.0

    bp = np.trapz(power[idx], freqs[idx])

    return bp


def compute_common_band_powers(signal, fs):
    """
    计算常见脑电频段能量。

    频段：
        delta: 0.5~4 Hz
        theta: 4~8 Hz
        alpha: 8~13 Hz
        beta : 13~30 Hz
        gamma: 30~40 Hz

    Returns
    -------
    result : dict
    """

    freqs, power = compute_power_spectrum(signal, fs)

    result = {
        "delta_0.5_4": band_power(freqs, power, 0.5, 4),
        "theta_4_8": band_power(freqs, power, 4, 8),
        "alpha_8_13": band_power(freqs, power, 8, 13),
        "beta_13_30": band_power(freqs, power, 13, 30),
        "gamma_30_40": band_power(freqs, power, 30, 40),
    }

    return result