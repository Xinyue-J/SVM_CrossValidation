from sklearn import svm
from plotSVMBoundaries import plotSVMBoundaries
import numpy as np
import csv


def csv_reader(x, y):
    data = []
    label = []
    with open(x) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(a) for a in row[0:2]])
    with open(y) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            label.append(int(row[0]))
    return np.array(data), np.array(label).reshape(20)


(data1, label1) = csv_reader('HW8_1_csv/train_x.csv', 'HW8_1_csv/train_y.csv')
(data2, label2) = csv_reader('HW8_2_csv/train_x.csv', 'HW8_2_csv/train_y.csv')


def svm_classfier(c, kernel, x, y, fig, circle=0, gamma='auto'):
    clf = svm.SVC(C=c, kernel=kernel, gamma=gamma)
    clf.fit(x, y)
    acc = clf.score(x, y)
    print(fig, 'accuracy:', acc)
    # clf.support_vectors_

    if circle == 0:
        plotSVMBoundaries(x, y, clf, fig)
    else:
        plotSVMBoundaries(x, y, clf, fig, clf.support_vectors_)
        w = clf.coef_
        w0 = clf.intercept_
        vectors = clf.support_vectors_
        for i in range(0, vectors.shape[0]):
            g = np.dot(w, vectors[i]) + w0
            print('point', i, 'g(x):', g[0])
        print('w:', w)
        print('w0:', w0)
        print('vectors:', vectors)


svm_classfier(1, 'linear', data1, label1, 'a1')
svm_classfier(100, 'linear', data1, label1, 'a2')
svm_classfier(100, 'linear', data1, label1, 'b', circle=1)
svm_classfier(1000, 'linear', data1, label1, 'c', circle=1)

svm_classfier(50, 'rbf', data2, label2, 'd1')
svm_classfier(5000, 'rbf', data2, label2, 'd2')
svm_classfier(1, 'rbf', data2, label2, 'e1', gamma=10)
svm_classfier(1, 'rbf', data2, label2, 'e2', gamma=50)
svm_classfier(1, 'rbf', data2, label2, 'e3', gamma=500)
