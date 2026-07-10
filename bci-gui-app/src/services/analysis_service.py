import os
import numpy as np
from bci_platform.io.edf_bdf_reader import EDFBDFReader
from bci_platform.process.basic_data_process import basic_filter_pipeline
from bci_platform.process.brain_connectivity import (
    compute_correlation_matrix,
    compute_band_correlation,
    threshold_matrix,
    plot_brain_connectivity
)

class AnalysisService:
    def __init__(self):
        self.eeg_data = None
        self.channel_names = []
        self.fs = None

    def load_data(self, file_path):
        reader = EDFBDFReader(file_path)
        eeg = reader.read(preload=True, physical=True)
        self.eeg_data = eeg.data[:64, :]  # Assuming first 64 channels are EEG
        self.channel_names = eeg.channel_names[:64]
        self.fs = eeg.main_sampling_rate
        return self.eeg_data, self.channel_names, self.fs

    def filter_data(self, low_freq, high_freq):
        if self.eeg_data is None:
            raise ValueError("EEG data not loaded.")
        return basic_filter_pipeline(self.eeg_data, self.fs, low_freq, high_freq)

    def compute_correlation(self, filtered_data):
        return compute_correlation_matrix(filtered_data)

    def compute_band_correlation(self, filtered_data, band):
        return compute_band_correlation(filtered_data, self.fs, band)

    def threshold_matrix(self, correlation_matrix, threshold):
        return threshold_matrix(correlation_matrix, threshold)

    def plot_connectivity(self, alpha_net, min_threshold, max_threshold):
        plot_brain_connectivity(
            alpha_net,
            self.channel_names,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            show_weight=False
        )