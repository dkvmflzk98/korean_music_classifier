import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

import librosa
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import csv
import os

f=open('./ex_2_fin_good.csv', 'w')
csvWriter = csv.writer(f)
channel = 16

names = ["NearestNeighbors", "K-NN_k=3", "Linear_SVM", "RBF_SVM", "Gaussian_Process",
         "Decision_Tree_5", "Decision_Tree_10", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes"]

classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()
    ]


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)


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
    for tmp in mfcc:
        l.append(np.std(tmp))
    for tmp in mfcc:
        l.append(np.max(tmp) - np.min(tmp))
    for tmp in mfcc:
        l.append(statistics.median(tmp))

    mfccs.append(l)
    print(song)
    print(l)

X = np.array(mfccs, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)


# 1 : balad, 2 : dance, 3 : fork&bluse, 4 : korea_traition, 5 : rap, 6 : rock, 7 : trote

scores = []
for i in range(6):
    csvWriter.writerow([i])
    if i < 5:
        X = np.array(mfccs, dtype=np.float32)[:, i * channel: (i+1)*channel]
    else:
        X = np.array(mfccs, dtype=np.float32)[:, 0: 2*channel]
    linearly_separable = (X, y)

    datasets = [linearly_separable]

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2, random_state=42)

        print(y_test)

        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)

            #score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
            precision = precision_score(y_test, y_pred, average=None)
            recall = recall_score(y_test, y_pred, average=None)
            print(name, precision, recall, score)
            csvWriter.writerow([name, score])
            csvWriter.writerow(precision)
            csvWriter.writerow(recall)

            joblib.dump(clf, './classifier/15/'+name + '_' + str(i) + '_' + str(score) + '.pkl')

        print(max(scores))