import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
for candidate in (PROJECT_ROOT, PARENT_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from bci_platform.io.edf_bdf_reader import EDFBDFReader
from bci_platform.process.basic_data_process import basic_filter_pipeline
from bci_platform.process.brain_connectivity import (
    compute_band_correlation,
    threshold_matrix,
    plot_brain_connectivity,
)


class BCIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("BCI EEG Analysis Workflow")
        self.master.geometry("1000x760")

        self.eeg = None
        self.eeg_data = None
        self.filtered_data = None
        self.channel_names = []
        self.file_path = None

        self.build_ui()
        self.show_placeholder_plot()

    def build_ui(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title = tk.Label(self.main_frame, text="EEG 数据处理流程")
        title.pack(anchor="w")

        intro = tk.Label(
            self.main_frame,
            text="按步骤进行：读取文件 -> 查看通道 -> 设置滤波参数 -> 查看滤波结果 -> 进行脑功能连接分析",
            justify="left",
            wraplength=950,
        )
        intro.pack(anchor="w", pady=(4, 10))

        controls = tk.Frame(self.main_frame)
        controls.pack(fill="x")

        self.load_button = tk.Button(controls, text="1. 读取 EDF 文件", command=self.load_data_file, width=24)
        self.load_button.pack(side="left", padx=(0, 8))

        self.channel_button = tk.Button(controls, text="2. 查看通道", command=self.show_channels, width=18, state="disabled")
        self.channel_button.pack(side="left", padx=(0, 8))

        self.filter_button = tk.Button(controls, text="3. 应用滤波", command=self.apply_filter, width=18, state="disabled")
        self.filter_button.pack(side="left", padx=(0, 8))

        self.connect_button = tk.Button(controls, text="4. 脑功能连接分析", command=self.run_connectivity, width=24, state="disabled")
        self.connect_button.pack(side="left")

        param_frame = tk.LabelFrame(self.main_frame, text="滤波参数", padx=8, pady=8)
        param_frame.pack(fill="x", pady=(10, 6))

        tk.Label(param_frame, text="低频截止(Hz)").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.low_freq_entry = tk.Entry(param_frame, width=10)
        self.low_freq_entry.grid(row=0, column=1, padx=4, pady=3)
        self.low_freq_entry.insert(0, "1")

        tk.Label(param_frame, text="高频截止(Hz)").grid(row=0, column=2, sticky="w", padx=4, pady=3)
        self.high_freq_entry = tk.Entry(param_frame, width=10)
        self.high_freq_entry.grid(row=0, column=3, padx=4, pady=3)
        self.high_freq_entry.insert(0, "40")

        content = tk.Frame(self.main_frame)
        content.pack(fill="both", expand=True)

        left = tk.LabelFrame(content, text="操作反馈", padx=8, pady=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.feedback_text = tk.Text(left, height=12, wrap="word")
        self.feedback_text.pack(fill="both", expand=True)
        self.feedback_text.configure(state="disabled")

        right = tk.LabelFrame(content, text="通道列表", padx=8, pady=8)
        right.pack(side="right", fill="y", padx=(6, 0))

        self.channel_listbox = tk.Listbox(right, width=26, height=16)
        self.channel_listbox.pack(fill="both", expand=True)

        self.plot_frame = tk.LabelFrame(self.main_frame, text="结果预览", padx=8, pady=8)
        self.plot_frame.pack(fill="both", expand=True, pady=(8, 0))

        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.figure.subplots_adjust(wspace=0.25, hspace=0.3)
        self.axes = self.figure.add_subplot(121)
        self.axes_topo = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def append_feedback(self, message):
        self.feedback_text.configure(state="normal")
        self.feedback_text.insert(tk.END, message + "\n")
        self.feedback_text.see(tk.END)
        self.feedback_text.configure(state="disabled")

    def show_placeholder_plot(self):
        self.axes.clear()
        self.axes.text(0.5, 0.5, "等待操作...", ha="center", va="center", fontsize=12)
        self.axes.set_axis_off()

        self.axes_topo.clear()
        self.axes_topo.text(0.5, 0.5, "等待操作...", ha="center", va="center", fontsize=12)
        self.axes_topo.set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw()

    def load_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("EDF Files", "*.edf")])
        if not file_path:
            return

        try:
            reader = EDFBDFReader(file_path)
            self.eeg = reader.read(preload=True, physical=True)
            self.eeg_data = self.eeg.data[:64, :]
            self.channel_names = list(self.eeg.channel_names[:64])
            self.file_path = file_path

            self.channel_listbox.delete(0, tk.END)
            for name in self.channel_names:
                self.channel_listbox.insert(tk.END, name)

            self.channel_button.config(state="normal")
            self.filter_button.config(state="disabled")
            self.connect_button.config(state="disabled")

            self.append_feedback(f"已读取文件：{file_path}")
            self.append_feedback(f"采样率：{self.eeg.main_sampling_rate} Hz")
            self.append_feedback(f"数据形状：{self.eeg_data.shape}")
            self.append_feedback("下一步：查看通道信息")

            self.show_placeholder_plot()
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            self.append_feedback(f"读取文件失败：{exc}")

    def show_channels(self):
        if self.eeg is None:
            messagebox.showwarning("提示", "请先读取 EDF 文件")
            return

        self.append_feedback("已显示通道列表")
        self.append_feedback(f"通道数量：{len(self.channel_names)}")
        self.append_feedback("前 10 个通道：" + ", ".join(self.channel_names[:10]))
        self.append_feedback("下一步：设置滤波参数并进行滤波")
        self.filter_button.config(state="normal")
        self.connect_button.config(state="disabled")

    def apply_filter(self):
        if self.eeg_data is None:
            messagebox.showwarning("提示", "请先读取 EDF 文件")
            return

        try:
            low_freq = float(self.low_freq_entry.get() or 1)
            high_freq = float(self.high_freq_entry.get() or 40)
            self.filtered_data = basic_filter_pipeline(
                self.eeg_data,
                fs=self.eeg.main_sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                notch_freq=50.0,
                notch_bandwidth=2.0,
                do_detrend=True,
            )

            self.append_feedback(f"已应用滤波：{low_freq} ~ {high_freq} Hz")
            self.append_feedback(f"滤波后数据形状：{self.filtered_data.shape}")
            self.append_feedback("下一步：查看滤波结果图")
            self.show_filtered_preview()
            self.connect_button.config(state="normal")
        except Exception as exc:
            messagebox.showerror("滤波失败", str(exc))
            self.append_feedback(f"滤波失败：{exc}")

    def show_filtered_preview(self):
        if self.filtered_data is None:
            return

        self.axes.clear()
        self.axes_topo.clear()
        x = range(min(500, self.filtered_data.shape[1]))
        y = self.filtered_data[0, : min(500, self.filtered_data.shape[1])]
        self.axes.plot(list(x), y, color="royalblue", linewidth=1.0)
        self.axes.set_title("滤波后信号预览（第 1 通道）")
        self.axes.set_xlabel("采样点")
        self.axes.set_ylabel("幅值")
        self.axes.grid(True, alpha=0.3)

        self.axes_topo.text(0.5, 0.5, "完成滤波后，可继续进行连接分析", ha="center", va="center")
        self.axes_topo.set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw()

    def run_connectivity(self):
        if self.filtered_data is None:
            messagebox.showwarning("提示", "请先完成滤波")
            return

        try:
            alpha_matrix = compute_band_correlation(self.filtered_data, self.eeg.main_sampling_rate, band=(8, 13))
            alpha_net = threshold_matrix(alpha_matrix, threshold=0.2)
            self.append_feedback("已完成脑功能连接分析")
            self.append_feedback("正在显示连接矩阵和拓扑图...")
            self.show_connectivity_plot(alpha_net)
            self.append_feedback("分析完成，可查看结果图")
        except Exception as exc:
            messagebox.showerror("连接分析失败", str(exc))
            self.append_feedback(f"连接分析失败：{exc}")

    def show_connectivity_plot(self, matrix):
        self.axes.clear()
        self.axes_topo.clear()

        try:
            self.axes.imshow(matrix, cmap="viridis")
            self.axes.set_title("脑功能连接矩阵")
            self.axes.set_xlabel("通道")
            self.axes.set_ylabel("通道")

            plot_brain_connectivity(
                matrix,
                self.channel_names,
                min_threshold=0.8,
                max_threshold=1.0,
                show_weight=False,
            )

            self.append_feedback("已显示脑功能连接矩阵和 EEG connectivity 拓扑图")
        except Exception as exc:
            self.append_feedback(f"拓扑图显示失败：{exc}")
            self.axes_topo.text(0.5, 0.5, "拓扑图绘制失败", ha="center", va="center")
            self.axes_topo.set_axis_off()

        self.axes_topo.set_title("EEG Connectivity 拓扑图")
        self.axes_topo.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = BCIApp(root)
    root.mainloop()