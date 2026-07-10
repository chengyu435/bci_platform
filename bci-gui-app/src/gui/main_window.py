from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QLabel, QComboBox, QCheckBox, QSlider, QHBoxLayout
)
import sys
import numpy as np
from bci_platform.io.edf_bdf_reader import EDFBDFReader
from bci_platform.process.basic_data_process import basic_filter_pipeline
from bci_platform.process.brain_connectivity import (
    compute_correlation_matrix,
    compute_band_correlation,
    threshold_matrix,
    plot_brain_connectivity
)
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI GUI Application")
        self.setGeometry(100, 100, 800, 600)

        self.data_file = None
        self.channel_names = []
        self.filtered_data = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.file_button = QPushButton("选择数据文件")
        self.file_button.clicked.connect(self.load_data_file)
        layout.addWidget(self.file_button)

        self.channel_label = QLabel("选择数据通道:")
        layout.addWidget(self.channel_label)

        self.channel_combo = QComboBox()
        layout.addWidget(self.channel_combo)

        self.filter_checkbox = QCheckBox("应用滤波")
        layout.addWidget(self.filter_checkbox)

        self.low_freq_slider = QSlider()
        self.low_freq_slider.setRange(0, 40)
        self.low_freq_slider.setValue(1)
        layout.addWidget(QLabel("低频阈值:"))
        layout.addWidget(self.low_freq_slider)

        self.high_freq_slider = QSlider()
        self.high_freq_slider.setRange(0, 40)
        self.high_freq_slider.setValue(40)
        layout.addWidget(QLabel("高频阈值:"))
        layout.addWidget(self.high_freq_slider)

        self.threshold_slider = QSlider()
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(80)
        layout.addWidget(QLabel("连接阈值:"))
        layout.addWidget(self.threshold_slider)

        self.plot_button = QPushButton("生成连接图")
        self.plot_button.clicked.connect(self.plot_connectivity)
        layout.addWidget(self.plot_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_data_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "EDF Files (*.edf);;All Files (*)", options=options)
        if file_name:
            self.data_file = file_name
            self.read_data()

    def read_data(self):
        reader = EDFBDFReader(self.data_file)
        eeg = reader.read(preload=True, physical=True)
        self.channel_names = eeg.channel_names[:64]
        self.channel_combo.clear()
        self.channel_combo.addItems(self.channel_names)

    def plot_connectivity(self):
        if self.data_file is None:
            self.result_label.setText("请先选择数据文件")
            return

        selected_channel = self.channel_combo.currentText()
        ch_idx = self.channel_names.index(selected_channel)

        reader = EDFBDFReader(self.data_file)
        eeg = reader.read(preload=True, physical=True)
        eeg_data = eeg.data[:64, :]

        if self.filter_checkbox.isChecked():
            low_freq = self.low_freq_slider.value()
            high_freq = self.high_freq_slider.value()
            self.filtered_data = basic_filter_pipeline(eeg_data, eeg.main_sampling_rate, low_freq, high_freq)
        else:
            self.filtered_data = eeg_data

        alpha_matrix = compute_band_correlation(self.filtered_data, eeg.main_sampling_rate, band=(8, 13))
        threshold_value = self.threshold_slider.value() / 100
        alpha_net = threshold_matrix(alpha_matrix, threshold=threshold_value)

        plot_brain_connectivity(
            alpha_net,
            self.channel_names,
            min_threshold=0.8,
            max_threshold=1,
            show_weight=False
        )
        plt.show()
        self.result_label.setText("连接图已生成")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())