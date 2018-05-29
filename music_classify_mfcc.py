import librosa
from sklearn import linear_model
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

songs = ['Beach Boys - Surfin Usa HD.wav','Mark Ronson - Uptown Funk ft. Bruno Mars.wav']
songs = []
mfccs = []

f = open('./music_list.txt')
songs = f.readlines()

for song in songs:
    song = song[:-1]
    audio_path = librosa.util.example_audio_file()
    y, sr = librosa.load(song, offset=15.0, duration=30.0)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    l = []
    for tmp in mfcc:
        l.append(sum(tmp)/len(tmp))
    mfccs.append(l[:2])
    print(l)

iris = load_iris()
X = np.array(mfccs, dtype=np.float32)  # we only take the first two features.
Y = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9], dtype=np.int64)

print(X)
print(Y)

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
