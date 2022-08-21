import sys, os
import matplotlib.pyplot as plt
import random
import os
import librosa
import librosa.display
import numpy as np

MY_FILE = "scale.wav"

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

mfcc = librosa.feature.mfcc(y=data, sr=sample_rate,hop_length=hop_length)


f,ax= plt.subplots(nrows=2, ncols=1, sharex=True)

mfcc_mesh = librosa.display.specshow(mfcc,x_axis="time",ax=ax[0])

mesh = librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length,x_axis="time",y_axis="linear",ax=ax[1])

ax[0].set(title='Mel-frequency cepstral coefficients (MFCCs)')
ax[0].label_outer()

ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()

f.colorbar(mfcc_mesh,ax=ax,format="%+2.f dB")
plt.title("Spectrogram")
plt.show()