import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random
import os
import librosa
import librosa.display
import numpy as np
import math
import json

MY_FILE = "scale.wav"

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.window_width, self.window_height = 1000, 800
        self.setMinimumSize(self.window_width, self.window_height)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        btn = QPushButton('Play', clicked=self.playAudioFile)
        self.layout.addWidget(btn)

        volumeControl = QHBoxLayout()
        self.layout.addLayout(volumeControl)

        btnVolumeUp = QPushButton('+', clicked=self.volumeUp)
        btnVolumeDown = QPushButton('-', clicked=self.volumeDown)
        butVolumeMute = QPushButton('Mute', clicked=self.volumeMute)
        volumeControl.addWidget(btnVolumeUp)
        volumeControl.addWidget(butVolumeMute)
        volumeControl.addWidget(btnVolumeDown)

        self.player = QMediaPlayer()

        self.plot_btn = QPushButton('Plot', clicked=self.spectrum)
        self.layout.addWidget(self.plot_btn)

        self.figure,self.ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)


    def spectrum(self):
        ax = self.ax
        data, sample_rate = librosa.load(MY_FILE, sr=22050)
        # STFT -> spectrogram
        hop_length = 512  # fourier transforma girecek verilerin bölüm bölüm taranırken sağa tarafa doğru ne kadarlık bir kayma olacağı
        n_fft = 2048  # bir fourier transforma girecek veri sayısı

        # calculate duration hop length and window in seconds
        hop_length_duration = float(hop_length) / sample_rate
        n_fft_duration = float(n_fft) / sample_rate

        print("STFT hop length duration is: {}s".format(hop_length_duration))
        print("STFT window duration is: {}s".format(n_fft_duration))

        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, hop_length=hop_length)

        mfcc_mesh = librosa.display.specshow(mfcc,x_axis="time",ax=ax[0])

        mesh = librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis="time",
                                        y_axis="linear", ax=ax[1])

        ax[0].set(title='Mel-frequency cepstral coefficients (MFCCs)')
        ax[0].label_outer()

        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()

        self.figure.colorbar(mfcc_mesh, ax=self.ax, format="%+2.f dB")
        plt.title("Spectrogram")
        self.canvas.draw()
        self.figure.clf()
        #plt.show()

    def plot(self):
        # random data
        data = [random.random() for i in range(10)]

        # clearing old figure
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

    def volumeUp(self):
        currentVolume = self.player.volume()  #
        print(currentVolume)
        self.player.setVolume(currentVolume + 5)

    def volumeDown(self):
        currentVolume = self.player.volume()  #
        print(currentVolume)
        self.player.setVolume(currentVolume - 5)

    def volumeMute(self):
        self.player.setMuted(not self.player.isMuted())

    def playAudioFile(self):
        full_file_path = os.path.join(os.getcwd(), 'scale.wav')
        url = QUrl.fromLocalFile(full_file_path)
        content = QMediaContent(url)

        self.player.setMedia(content)
        self.player.play()


if __name__ == '__main__':
    # don't auto scale when drag app to a different monitor.
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 20px;
        }
    ''')

    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')


