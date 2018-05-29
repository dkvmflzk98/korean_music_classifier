from sklearn.cluster import KMeans
import numpy as np
import statistics
import librosa
import os

channel = 16
mfccs = []
mfcc_record = []
y_list = []
songs = []
cn = 0

for category in ['balad', 'dance', 'fork/bluse', 'korea_tradition', 'rap/hiphop', 'rock', 'trote']:
    for root, dirs, files in os.walk('./music/' + category):
        for fname in files:
            full_fname = os.path.join(root, fname)
            songs.append(full_fname)
            print(full_fname)
            y_list.append(cn)
    cn += 1

for song in songs:
    audio_path = librosa.util.example_audio_file()
    y, sr = librosa.load(song, offset=15.0, duration=30.0)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=channel)
    l = []
    for tmp in mfcc:
        l.append(np.mean(tmp))
    for tmp in mfcc:
        l.append(np.var(tmp))


    mfccs.append(l)
    print(song)
    print(l)

X = np.array(mfccs, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
y2 = kmeans.predict(X)

re = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

for i in range(len(y)):
    b = y2[i]
    a = y[i]
    re[b][a]+=1

for t in re:
    print(t)