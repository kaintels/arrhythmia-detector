# -*- coding: utf-8 -*-
import datetime
import pickle
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.signal import find_peaks
from models.model import ATMNet
import json

warnings.filterwarnings("ignore")
QT_AUTO_SCREEN_SCALE_FACTOR = 2

if hasattr(Qt, "AA_EnableHighDpiScaling"):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, "AA_UseHighDpiPixmaps"):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

with open("config.json") as jsonfile:
    param = json.load(jsonfile)

RANGE = param["RANGE"]
cnt = param["cnt"]
switch = param["switch"]
detect = param["detect"]
tmp_index = param["tmp_index"]
finish = param["finish"]
check_detect = param["check_detect"]
clears = param["clears"]
run_time = datetime.datetime.now()

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=1, height=1, dpi=100):
        plt.style.use("dark_background")
        plt.rcParams["font.family"] = "Consolas"
        plt.rcParams["ytick.labelsize"] = 15
        plt.rcParams["xtick.labelsize"] = 15
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(
            111, xlim=(0, RANGE), ylim=(-1.5, 1.5), position=[0.15, 0.15, 0.75, 0.75]
        )
        self.axes.axvline(x=1200, color="orange", linestyle="--", linewidth=3)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.graphnow = datetime.datetime.now()
        tmptime = []
        tmp_tick = []
        tmp_loop = []
        plt.autoscale()

        for i in range(1):
            tmptime.append(int(str(self.graphnow)[17:19]) + i)
            tmp_tick.append(str(self.graphnow)[10:17] + str(tmptime[-1]))
        first_time = ["", "", "", "", "", "", "", tmp_tick[0]]
        self.first_time = first_time
        self.tmp_loop = tmp_loop
        self.axes.set_xlabel("Time (h:m:s)", color="white", fontsize=30)
        self.axes.set_xticklabels(self.first_time)
        self.axes.set_ylabel("Amplitude (mV)", color="white", fontsize=30)
        self.axes.set_title("Electrocardiogram", color="white", fontsize=30)

    def compute_initial_figure(self):
        pass


class AnimationWidget(QWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        self.canvas = MyMplCanvas(self)
        self.first_time = self.canvas.first_time
        self.tmp_loop = self.canvas.tmp_loop
        self.setStyleSheet("background-color:black;")
        self.FilePath = QTextBrowser()
        self.FilePath.setStyleSheet(
            "border: 2px solid rgb(255, 255, 255); color: white"
        )
        self.FilePath.setFont(QFont("Consolas", 10))
        self.FilePath.setFixedHeight(50)
        self.OpenButton = QPushButton("File load")
        self.OpenButton.setFixedHeight(50)
        self.OpenButton.setFont(QFont("Consolas", 15))
        self.OpenButton.setStyleSheet(
            "border: 2px solid rgb(0, 0, 0); border-radius : 15px; color: black ;background-color: #c9ced6"
        )
        self.DetectLabel = QLabel("Not Detecting")
        self.DetectLabel.setFixedWidth(400)
        self.DetectLabel.setAlignment(Qt.AlignCenter)
        self.DetectLabel.setFont(QFont("Consolas", 30))
        self.DetectLabel.setStyleSheet("color: white")
        self.Arrhythmias = QTextBrowser()
        self.Arrhythmias.setStyleSheet(
            "border: 2px solid rgb(255, 255, 255); color: white"
        )
        self.Arrhythmias.setFont(QFont("Consolas", 12))
        self.DetectionHistory = QLabel("Detection history")
        self.DetectionHistory.setFixedWidth(400)
        self.DetectionHistory.setAlignment(Qt.AlignCenter)
        self.DetectionHistory.setFont(QFont("Consolas", 15))
        self.DetectionHistory.setStyleSheet("color: white")
        self.RunButton = QPushButton("Run", self)
        self.StopButton = QPushButton("Pause", self)
        self.ClearButton = QPushButton("Clear", self)
        self.CloseButton = QPushButton("Close", self)
        self.Tempbutton = QPushButton("Temp", self)
        self.RunButton.setStyleSheet(
            "border: 2px solid rgb(0, 0, 0); border-radius : 15px; color: black ;background-color: #8a9ab5;"
        )
        self.RunButton.setFixedHeight(50)
        self.RunButton.setFont(QFont("Consolas", 15))
        self.StopButton.setFixedHeight(50)
        self.StopButton.setFont(QFont("Consolas", 15))
        self.StopButton.setStyleSheet(
            "border: 2px solid rgb(0, 0, 0); border-radius : 15px; color: black ;background-color: #8a9ab5"
        )
        self.ClearButton.setFixedHeight(50)
        self.ClearButton.setFont(QFont("Consolas", 15))
        self.ClearButton.setStyleSheet(
            "border: 2px solid rgb(0, 0, 0); border-radius : 15px; color: black ;background-color: #8a9ab5"
        )
        self.CloseButton.setStyleSheet(
            "border: 2px solid rgb(0, 0, 0); border-radius : 15px; color: black ;background-color: #8a9ab5"
        )
        self.CloseButton.setFixedHeight(50)
        self.CloseButton.setFont(QFont("Consolas", 15))

        self.MSG = QMessageBox()
        self.EndMSG = QMessageBox()

        self.Pbar = QProgressBar()
        self.Pbar.setFixedWidth(200)

        self.timer = QTimer()

        leftLayout = QVBoxLayout()
        self.leftLayout = leftLayout
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.FilePath)
        rightLayout.addWidget(self.OpenButton)
        rightLayout.addStretch(1)
        rightLayout.addWidget(self.DetectLabel)
        rightLayout.addWidget(self.DetectLabel)
        rightLayout.addStretch(1)
        rightLayout.addWidget(self.DetectionHistory)
        rightLayout.addWidget(self.Arrhythmias)
        rightLayout.addWidget(self.RunButton)
        rightLayout.addWidget(self.StopButton)
        rightLayout.addWidget(self.ClearButton)
        rightLayout.addWidget(self.CloseButton)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)
        self.OpenButton.clicked.connect(self.openpath)
        self.RunButton.clicked.connect(self.on_start)
        self.StopButton.clicked.connect(self.on_stop)
        self.ClearButton.clicked.connect(self.on_clear)
        self.CloseButton.clicked.connect(self.close_event)
        self.Tempbutton.clicked.connect(self.passes)
        self.showFullScreen()

        signals = []  # list of total ECG signal

        self.x = np.arange(RANGE)
        self.nan = np.ones(RANGE, dtype=np.float) * np.nan
        (self.line,) = self.canvas.axes.plot(
            self.x, self.nan, animated=True, color="deepskyblue", lw=2
        )
        (self.peak,) = self.canvas.axes.plot(
            self.x, self.nan, color="red", marker="*", lw=1, ms=10, ls="None",
        )
        (self.atrpeak,) = self.canvas.axes.plot(
            self.x, self.nan, color="red", marker="x", lw=3, ms=5, mew=100, ls="None",
        )

        x_values = []
        y_values = []
        self.x_values = x_values
        self.y_values = y_values
        ecg_peaks = []  # list of peak index of ECGs
        ecg_values = []  # list of peak value of ECGs
        self.y_values = y_values
        self.ecg_peaks = ecg_peaks
        self.ecg_values = ecg_values
        self.signals = signals
        self.ecg_atrpeak = []
        self.pre_peak = 0

    def openpath(self):
        filters = (
            "csv files (*.csv);; matlab variable files (*.mat);; pickle files (*.pkl),"
        )
        filename = QFileDialog.getOpenFileName(self, "Open file", ".", filters)
        self.FilePath.setText(str(filename[0]))

        data_load = self.FilePath.toPlainText()
        if str(data_load[-3]) == "c":
            self.data = pd.read_csv(str(data_load), index_col=0)
            self.data = self.data.values[:1800]
        if str(data_load[-3]) == "p":
            with open(str(data_load), "rb") as f:
                self.data = pickle.load(f)
                self.data = self.data[6000:6500]
        if str(data_load[-3]) == "m":
            self.data = loadmat(str(data_load), squeeze_me=True)["ECG"]
            self.data = self.data[6000:7000]

    def init(self):
        self.line.set_data([], [])
        return (self.line,)

    def update_line(self, i):
        global switch, cnt, detect, tmp_index, check_detect, clears, finish

        if clears == 1:
            (self.line,) = self.canvas.axes.plot(
                self.x, self.nan, animated=True, color="deepskyblue", lw=2
            )
            (self.peak,) = self.canvas.axes.plot(
                self.x, self.nan, color="red", marker="*", lw=2, ms=5, ls="None",
            )
            (self.atrpeak,) = self.canvas.axes.plot(
                self.x,
                self.nan,
                color="red",
                marker="x",
                lw=3,
                ms=5,
                mew=100,
                ls="None",
            )
            clears = 0

        i = tmp_index
        switch = 1
        if i == len(self.data) - 1:
            switch = 3
            if check_detect == 1:
                self.ani._stop()
                self.EndMSG.setText(
                    "The history result has been saved. "
                    + str(run_time)[:10]
                    + "_"
                    + str(run_time)[11:13]
                    + "_"
                    + str(run_time)[14:16]
                    + "_"
                    + str(run_time)[17:19]
                )
                self.EndMSG.setWindowTitle("Notice")
                self.EndMSG.setIcon(QMessageBox.Information)
                self.EndMSG.setStandardButtons(QMessageBox.Ok)
                self.EndMSG.exec_()
                result = open(
                    "./log/"
                    + str(run_time)[:10]
                    + "_"
                    + str(run_time)[11:13]
                    + "_"
                    + str(run_time)[14:16]
                    + "_"
                    + str(run_time)[17:19]
                    + "_detecton_log.txt",
                    "w",
                )
                result.write(self.Arrhythmias.toPlainText())
                result.close()
                self.EndMSG = QMessageBox()

            if check_detect == 0:
                self.ani._stop()
                self.EndMSG.setText("No arrhythmia detected.")
                self.EndMSG.exec_()

        y = self.data[i]
        old_y = self.line.get_ydata()
        new_y = np.r_[old_y[1:], y]
        self.line.set_ydata(new_y)
        ecg_peak, _ = find_peaks(
            np.abs(new_y), height=(0.3, 3), prominence=(0.4, 2), distance=80
        )
        ecg_peak = np.array(ecg_peak)
        if len(ecg_peak) > 0:
            self.ecg_peaks.append(ecg_peak[-1])
            self.ecg_values.append(new_y[ecg_peak[-1]])
            if ecg_peak[-1] == 1200 or ecg_peak[-1] < 1203 and ecg_peak[-1] > 1200:
                signal = new_y[ecg_peak[-1] - 70 : ecg_peak[-1] + 100]
                signal = np.array(signal)
                signal = signal[np.newaxis, np.newaxis, :]
                output = model(torch.FloatTensor(signal))
                _, predicted = torch.max(output.data, 1)
                if predicted == 0:
                    self.DetectLabel.setText("Normal")
                    self.DetectLabel.setStyleSheet("color: white")
                    self.DetectLabel.setAlignment(Qt.AlignCenter)
                    self.pre_peak = 0
                    detect = 1
                if predicted == 1:
                    temp_arr = []
                    now = datetime.datetime.now()
                    temp_arr.append("No." + str(cnt + 1) + "\t\t" + str(now)[:19])
                    self.DetectLabel.setText("Arrhythmia")
                    self.DetectLabel.setStyleSheet("color: red")
                    self.DetectLabel.setSizePolicy(
                        QSizePolicy.Expanding, QSizePolicy.Expanding
                    )
                    self.DetectLabel.setAlignment(Qt.AlignCenter)
                    if predicted - self.pre_peak == 1:
                        self.ecg_atrpeak.append(ecg_peak[-1])
                        self.Arrhythmias.append(temp_arr[-1])
                        cnt += 1
                        detect = 1
                        check_detect = 1
                    self.pre_peak = 1

        else:
            self.ecg_peaks.append(0)
            self.ecg_values.append(0)
        if len(self.ecg_atrpeak) and self.ecg_atrpeak[0] == 0:
            self.ecg_atrpeak.remove(0)
        self.ecg_atrpeak = [
            self.ecg_atrpeak[i] - 1 for i in range(len(self.ecg_atrpeak))
        ]
        self.peak.set_data(ecg_peak, new_y[ecg_peak])
        self.atrpeak.set_data(self.ecg_atrpeak, new_y[self.ecg_atrpeak])

        if i >= 240 and i % 240 == 0:
            self.noww = datetime.datetime.now()
            self.first_time.insert(len(self.first_time), str(self.noww)[10:19])
            self.first_time.pop(0)
            self.canvas.axes.set_xticklabels(self.first_time)
            switch = 1
            self.on_stop()
            self.on_start()

        tmp_index += 1

        return [self.line, self.peak, self.atrpeak]

    def on_start(self):
        global switch
        if switch == 0 or switch == 3:
            self.ani = animation.FuncAnimation(
                self.canvas.figure,
                self.update_line,
                interval=1,
                blit=True,
                frames=600000,
                repeat=True,
                save_count=50,
                cache_frame_data=False,
            )

    def on_stop(self):
        global switch
        if switch == 1:
            self.ani._stop()
            switch = 0
        else:
            pass

    def on_clear(self):
        global clears, tmp_index, cnt, detect, finish, check_detect
        tmp_index = 0
        if clears == 0:
            for ax in self.canvas.figure.axes:
                ax.clear()
            fig = Figure(figsize=(5, 4), dpi=100)
            self.axes = fig.add_subplot(111, xlim=(0, RANGE), ylim=(-1.5, 1.5))
            self.canvas.axes.set_xlim(0, RANGE)
            self.canvas.axes.set_ylim(-1.5, 1.5)
            self.canvas.axes.axvline(
                x=1200, color="orange", linestyle="--", linewidth=3
            )
            self.graphnow = datetime.datetime.now()
            tmptime = []
            tmp_tick = []
            tmp_loop = []
            for i in range(1):
                tmptime.append(int(str(self.graphnow)[17:19]) + i)
                tmp_tick.append(str(self.graphnow)[10:17] + str(tmptime[-1]))
            first_time = ["", "", "", "", "", "", "", tmp_tick[0]]
            self.first_time = first_time
            self.tmp_loop = tmp_loop
            self.canvas.axes.set_xlabel("Time (h:m:s)", color="white", fontsize=30)
            self.canvas.axes.set_xticklabels(self.first_time)
            self.canvas.axes.set_ylabel("Amplitude (mV)", color="white", fontsize=30)
            self.canvas.axes.set_title("Electrocardiogram", color="white", fontsize=30)
            self.canvas.draw()
            self.DetectLabel.setText("Not Detecting")
            self.DetectLabel.setStyleSheet("color: white")
            self.Arrhythmias.clear()
            clears = 1
            cnt = 0
            detect = 0
            finish = 0
            check_detect = 0

    def passes(self):
        pass

    def close_event(self):
        global switch
        if switch == 1:
            self.ani._stop()
            switch = 0
        self.MSG.setText("Do you want to quit?")
        self.MSG.setWindowTitle("Quit?")
        self.MSG.setIcon(QMessageBox.Information)
        self.MSG.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = self.MSG.exec_()
        if result == QMessageBox.Yes:
            sys.exit()
        elif result == QMessageBox.No:
            pass


if __name__ == "__main__":
    print(" 1 : 프로그램 실행 \t 2 : 모델 학습 \t 3 : 종료 \t (숫자 입력)")
    number = int(input())
    if number == 1:
        import sys

        print("실행중입니다..잠시 기다려주세요.")
        model = ATMNet()
        with torch.no_grad():
            model.eval()
        model.load_state_dict(
            torch.load("./checkpoint/predictor.pth", map_location="cpu")
        )
        app = QApplication(sys.argv)
        windows = AnimationWidget()
        windows.setWindowTitle("Arrhythmia Detection Program")
        windows.show()
        sys.exit(app.exec_())
    if number == 2:
        from trainer.trainer import train_model_slack_notify
        import os

        os_path = os.path.realpath("")
        with open(os_path + "\\dataset\\training\\feature.pkl", "rb") as f:
            feature = pickle.load(f)

        with open(os_path + "\\dataset\\training\\target.pkl", "rb") as f:
            target = pickle.load(f)

        train_model_slack_notify(
            target=feature, label=target, batch_size_num=32, epochs=5
        )

    if number == 3:
        print("종료")
        import sys

        sys.exit()
