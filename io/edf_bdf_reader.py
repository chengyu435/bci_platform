# -*- coding: utf-8 -*-
"""
EDF/BDF 数据读取模块
说明：
1. 不依赖 MNE、pyEDFlib、BrainFlow 等现成脑机接口工具箱。
2. 只使用基础的数组、字典、列表等数据结构，便于后续集成到国产化脑机接口平台。
2. 直接解析 EDF/BDF 文件头和数据区。
3. 支持 EDF 16 bit 数据和 BDF 24 bit 数据。
4. 适合作为国产化脑机接口平台的数据读取基础模块。
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EEGData:
    """
    脑电数据统一数据结构
    """
    data: np.ndarray                  # shape: [n_channels, n_samples]
    channel_names: List[str]
    sampling_rates: Dict[str, float]
    main_sampling_rate: float
    physical_units: List[str]
    time: np.ndarray
    header: Dict
    raw_digital_data: Optional[np.ndarray] = None


class EDFBDFReader:
    """
    EDF/BDF 文件读取器
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_ext = os.path.splitext(file_path)[1].lower()

        if self.file_ext not in [".edf", ".bdf"]:
            raise ValueError("当前仅支持 .edf 和 .bdf 文件")

        self.header = {}
        self.signal_headers = []

    def read(self, preload: bool = True, physical: bool = True) -> EEGData:
        """
        读取 EDF/BDF 文件

        Parameters
        ----------
        preload : bool
            是否直接读取全部数据。
        physical : bool
            是否转换为物理量数据。

        Returns
        -------
        EEGData
            统一格式的脑电数据对象。
        """

        with open(self.file_path, "rb") as f:
            self._read_header(f)

            if not preload:
                raise NotImplementedError("当前版本暂只支持 preload=True")

            digital_data = self._read_data_records(f)

        if physical:
            data = self._digital_to_physical(digital_data)
        else:
            data = digital_data.astype(np.float64)

        channel_names = [ch["label"] for ch in self.signal_headers]
        physical_units = [ch["physical_dimension"] for ch in self.signal_headers]

        sampling_rates = {
            ch["label"]: ch["samples_per_record"] / self.header["duration_of_data_record"]
            for ch in self.signal_headers
        }

        # 主采样率：默认取第一个通道采样率
        main_fs = sampling_rates[channel_names[0]]

        # 当前版本假定主要脑电通道采样率一致
        n_samples = data.shape[1]
        time = np.arange(n_samples) / main_fs

        return EEGData(
            data=data,
            channel_names=channel_names,
            sampling_rates=sampling_rates,
            main_sampling_rate=main_fs,
            physical_units=physical_units,
            time=time,
            header=self.header,
            raw_digital_data=digital_data
        )

    def _read_ascii(self, f, n_bytes: int) -> str:
        """
        读取固定长度 ASCII 字段
        """
        return f.read(n_bytes).decode("latin-1", errors="ignore").strip()

    def _read_header(self, f):
        """
        读取 EDF/BDF 文件头
        """

        self.header["version"] = self._read_ascii(f, 8)
        self.header["patient_id"] = self._read_ascii(f, 80)
        self.header["recording_id"] = self._read_ascii(f, 80)
        self.header["start_date"] = self._read_ascii(f, 8)
        self.header["start_time"] = self._read_ascii(f, 8)

        header_bytes = int(self._read_ascii(f, 8))
        self.header["header_bytes"] = header_bytes

        self.header["reserved"] = self._read_ascii(f, 44)

        n_data_records_str = self._read_ascii(f, 8)
        self.header["n_data_records"] = int(n_data_records_str)

        duration_str = self._read_ascii(f, 8)
        self.header["duration_of_data_record"] = float(duration_str)

        n_signals = int(self._read_ascii(f, 4))
        self.header["n_signals"] = n_signals

        labels = [self._read_ascii(f, 16) for _ in range(n_signals)]
        transducer_types = [self._read_ascii(f, 80) for _ in range(n_signals)]
        physical_dimensions = [self._read_ascii(f, 8) for _ in range(n_signals)]

        physical_mins = [float(self._read_ascii(f, 8)) for _ in range(n_signals)]
        physical_maxs = [float(self._read_ascii(f, 8)) for _ in range(n_signals)]

        digital_mins = [int(self._read_ascii(f, 8)) for _ in range(n_signals)]
        digital_maxs = [int(self._read_ascii(f, 8)) for _ in range(n_signals)]

        prefilterings = [self._read_ascii(f, 80) for _ in range(n_signals)]
        samples_per_records = [int(self._read_ascii(f, 8)) for _ in range(n_signals)]

        reserved = [self._read_ascii(f, 32) for _ in range(n_signals)]

        self.signal_headers = []

        for i in range(n_signals):
            ch_info = {
                "label": labels[i],
                "transducer_type": transducer_types[i],
                "physical_dimension": physical_dimensions[i],
                "physical_min": physical_mins[i],
                "physical_max": physical_maxs[i],
                "digital_min": digital_mins[i],
                "digital_max": digital_maxs[i],
                "prefiltering": prefilterings[i],
                "samples_per_record": samples_per_records[i],
                "reserved": reserved[i],
            }
            self.signal_headers.append(ch_info)

        # 确保文件指针跳转到数据区起始位置
        f.seek(header_bytes)

    def _read_data_records(self, f) -> np.ndarray:
        """
        读取 EDF/BDF 数据记录

        EDF:
            每个采样点为 16 bit signed integer，小端格式。

        BDF:
            每个采样点为 24 bit signed integer，小端格式。
        """

        n_records = self.header["n_data_records"]
        n_signals = self.header["n_signals"]

        samples_per_record = [
            ch["samples_per_record"] for ch in self.signal_headers
        ]

        total_samples_per_channel = [
            n_records * spr for spr in samples_per_record
        ]

        max_samples = max(total_samples_per_channel)

        # 用 NaN 填充，兼容不同通道采样率不一致的情况
        data = np.full((n_signals, max_samples), np.nan, dtype=np.float64)

        for record_idx in range(n_records):
            for ch_idx in range(n_signals):
                n_samples = samples_per_record[ch_idx]

                if self.file_ext == ".edf":
                    raw = f.read(n_samples * 2)
                    values = np.frombuffer(raw, dtype="<i2").astype(np.float64)

                elif self.file_ext == ".bdf":
                    raw = f.read(n_samples * 3)
                    values = self._decode_bdf_24bit(raw).astype(np.float64)

                else:
                    raise ValueError("未知文件类型")

                start = record_idx * n_samples
                end = start + n_samples

                data[ch_idx, start:end] = values

        return data

    def _decode_bdf_24bit(self, raw: bytes) -> np.ndarray:
        """
        解码 BDF 24 bit 小端有符号整数

        BDF 每个采样点占 3 字节：
            byte0 + byte1 << 8 + byte2 << 16

        若最高位为 1，则表示负数，需要进行符号扩展。
        """

        byte_data = np.frombuffer(raw, dtype=np.uint8)

        if len(byte_data) % 3 != 0:
            raise ValueError("BDF 数据长度异常，无法按 24 bit 解码")

        byte_data = byte_data.reshape(-1, 3)

        values = (
            byte_data[:, 0].astype(np.int32)
            | (byte_data[:, 1].astype(np.int32) << 8)
            | (byte_data[:, 2].astype(np.int32) << 16)
        )

        # 24 bit 有符号整数符号扩展
        sign_bit = 1 << 23
        values = np.where(values & sign_bit, values - (1 << 24), values)

        return values.astype(np.int32)

    def _digital_to_physical(self, digital_data: np.ndarray) -> np.ndarray:
        """
        将数字量转换为物理量

        转换公式：
            physical = physical_min
                     + (digital - digital_min)
                     * (physical_max - physical_min)
                     / (digital_max - digital_min)
        """

        n_channels = digital_data.shape[0]
        physical_data = np.zeros_like(digital_data, dtype=np.float64)

        for ch_idx in range(n_channels):
            ch = self.signal_headers[ch_idx]

            dig_min = ch["digital_min"]
            dig_max = ch["digital_max"]
            phy_min = ch["physical_min"]
            phy_max = ch["physical_max"]

            scale = (phy_max - phy_min) / (dig_max - dig_min)
            offset = phy_min - dig_min * scale

            physical_data[ch_idx, :] = digital_data[ch_idx, :] * scale + offset

        return physical_data

    def print_summary(self):
        """
        打印文件摘要信息
        """
        print("========== EDF/BDF 文件信息 ==========")
        print(f"文件路径: {self.file_path}")
        print(f"文件类型: {self.file_ext}")
        print(f"患者信息: {self.header.get('patient_id')}")
        print(f"记录信息: {self.header.get('recording_id')}")
        print(f"开始日期: {self.header.get('start_date')}")
        print(f"开始时间: {self.header.get('start_time')}")
        print(f"数据记录数: {self.header.get('n_data_records')}")
        print(f"单条记录时长: {self.header.get('duration_of_data_record')} s")
        print(f"通道数量: {self.header.get('n_signals')}")
        print()

        print("========== 通道信息 ==========")
        for idx, ch in enumerate(self.signal_headers):
            fs = ch["samples_per_record"] / self.header["duration_of_data_record"]
            print(
                f"{idx + 1:02d}. {ch['label']} | "
                f"单位: {ch['physical_dimension']} | "
                f"采样率: {fs:.2f} Hz | "
                f"数字范围: [{ch['digital_min']}, {ch['digital_max']}] | "
                f"物理范围: [{ch['physical_min']}, {ch['physical_max']}]"
            )