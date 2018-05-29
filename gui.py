from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

import librosa
import numpy as np

clf = MLPClassifier()
clf = joblib.load('classifier/7/Neural_Net_4_0.638888888889.pkl')
print(clf)

audio_path = librosa.util.example_audio_file()
y, sr = librosa.load("music/balad/01 Zion T 자이언티   눈 Feat  이문세 12월 1주차 1위.wav", offset=15.0, duration=30.0)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)
log_S = librosa.logamplitude(S, ref_power=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=16)
l = []
for tmp in mfcc:
    l.append(np.mean(tmp))
print(l)

feature = np.array([l], dtype=np.float32)
print(feature)
predict = clf.predict(feature)
print('asdf')
print(predict)

