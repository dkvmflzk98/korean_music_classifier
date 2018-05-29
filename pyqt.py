import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import librosa
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

clf = joblib.load('./Neural_Net_5_0.65.pkl')

links = []
jangre = ['balad', 'dance', 'fork&bluse', 'korean_tradition', 'rap&hiphop', 'rock', 'trote']


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.folderLayout = QWidget()

        self.pathRoot = QDir.rootPath()

        self.dirmodel = QFileSystemModel(self)
        self.dirmodel.setRootPath(QDir.currentPath())

        self.indexRoot = self.dirmodel.index(self.dirmodel.rootPath())

        self.folder_view = QTreeView()
        self.folder_view.setDragEnabled(True)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.setRootIndex(self.indexRoot)

        self.selectionModel = self.folder_view.selectionModel()

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.folder_view)

        self.folderLayout.setLayout(self.left_layout)

        splitter_filebrowser = QSplitter(Qt.Horizontal)
        splitter_filebrowser.addWidget(self.folderLayout)
        splitter_filebrowser.addWidget(Figure_Canvas(self))
        splitter_filebrowser.setStretchFactor(1, 1)

        hbox = QHBoxLayout(self)
        hbox.addWidget(splitter_filebrowser)

        self.centralWidget().setLayout(hbox)

        self.setWindowTitle('Music classifier')
        self.setGeometry(750, 100, 600, 500)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setGeometry(320, 20, 250, 460)
        self.tableWidget.setRowCount(100)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['title', 'result'])

    def addRow(self, data, row):
        for col in range(2):
            item = QTableWidgetItem(data[col])
            self.tableWidget.setItem(row, col, item)

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.update()

    def classify(self, l):
        feature = np.array([l], dtype=np.float32)
        pred = clf.predict(feature)

        print(jangre[pred[0]])
        self.addRow((links[-1].split('/')[-1], jangre[pred[0]]), len(links)-1)


class Figure_Canvas(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.setAcceptDrops(True)

        blabla = QLineEdit()

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(blabla)

        self.buttonLayout = QWidget()
        self.buttonLayout.setLayout(self.right_layout)

    def dragEnterEvent(self, e):

        if e.mimeData().hasFormat('text/uri-list'):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        url = e.mimeData().urls()[0]
        links.append(str(url.toLocalFile()))

        audio_path = librosa.util.example_audio_file()
        y, sr = librosa.load(links[-1], offset=15.0, duration=30.0)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)
        log_S = librosa.logamplitude(S, ref_power=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=16)
        l = []
        for tmp in mfcc:
            l.append(np.mean(tmp))
        for tmp in mfcc:
            l.append(np.var(tmp))
        print(l)
        ex.classify(l)


app = QApplication(sys.argv)
ex = Example()
ex.show()
app.exec_()
