import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
import numpy as np
import wave
import datetime

f = open('./music_list.txt')
songs = f.readlines()
plotn = 1
mfccs = []

for song in songs:
    song = song[:-1]
    now = datetime.datetime.now()
    fs, data = wavfile.read(song)  # load the data
    a = data.T[0]  # this is a two channel soundtrack, I get the first track
    b = [(ele/2**8.)*2-1 for ele in a]  # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b)  # calculate fourier transform (complex numbers list)
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)

    plt.figure(plotn)
    plt.title('Signal Wave...')
    plt.plot(data.T[0])
    plt.savefig("test_" + str(plotn) + ".png", dpi=150)
    plotn = plotn + 1

    plt.figure(plotn)
    plt.title('fft')
    plt.plot(abs(c[:int(d-1)]), 'r')
    plt.savefig("test_" + str(plotn) + ".png", dpi=150)
    plotn = plotn + 1

    print(datetime.datetime.now()-now)

    plt.show()
