from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import matplotlib.pyplot as plt
from plotSVMBoundaries import plotSVMBoundaries


def wine_reader(feature, label):
    x = []
    y = []
    with open(feature) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x.append([float(a) for a in row[0:2]])
    with open(label) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            y.append(int(row[0]))
    return np.array(x), np.array(y).reshape(89)


# f_tr l_tr: train / f_v l_v: validation / f_te l_te: test
(feature_train, label_train) = wine_reader('wine_csv/feature_train.csv', 'wine_csv/label_train.csv')
(feature_test, label_test) = wine_reader('wine_csv/feature_test.csv', 'wine_csv/label_test.csv')


def crossvalidation(C, gamma, f_tr, l_tr, skf):
    acc_ = []
    for train_index, test_index in skf.split(f_tr, l_tr):
        nf_tr, f_v = f_tr[train_index], f_tr[test_index]
        nl_tr, l_v = l_tr[train_index], l_tr[test_index]
        clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(nf_tr, nl_tr)
        pred_l_v = clf.predict(f_v)
        acc_.append(accuracy_score(l_v, pred_l_v))
    return acc_


kf = StratifiedKFold(n_splits=5, shuffle=True)  # initialize
acc1 = crossvalidation(1, 1, feature_train, label_train, kf)
print('average cross-validation accuracy:', np.mean(acc1))


def find_bestpair(ACC, DEV, Cs=np.logspace(-3, 3, 50), gammas=np.logspace(-3, 3, 50)):
    index_maxmean = np.argwhere(ACC == np.max(ACC))
    # print(index_maxmean)
    i1 = index_maxmean[:, 0]
    j1 = index_maxmean[:, 1]
    # print(np.max(ACC[t,:,:]))
    index_mindev = np.argwhere(DEV == np.min(DEV))
    # print(index_mindev)
    i2 = index_mindev[:, 0]
    j2 = index_mindev[:, 1]
    # print(np.min(DEV[t,:,:]))

    if len(i1) == 1 and DEV[i1[0], j1[0]] == np.min(DEV):  # clear choice
        print('clear choice:C=', Cs[i1[0], j1[0]], 'gamma=', gammas[i1[0], j1[0]])
        print("mean=", ACC[i1[0], j1[0]], 'deviation=', DEV[i1[0], j1[0]])
    else:  # max mean -> min dev
        pair = []
        for index in range(len(i1)):
            pair.append(DEV[i1[index], j1[index]])
            # pair contains the dev with max mean, then we find the min dev among this
        best_index = np.argwhere(pair == np.min(pair))  # position in pair
        best_c = Cs[i1[best_index[0][0]]]
        best_gamma = gammas[j1[best_index[0][0]]]
        print('best choice:C=', best_c, 'gamma=', best_gamma)
        mean = ACC[i1[best_index[0][0]], j1[best_index[0][0]]]
        deviation = DEV[i1[best_index[0][0]], j1[best_index[0][0]]]
        print("mean=", mean, 'deviation=', deviation)
        return best_c, best_gamma, mean, deviation


def validation():
    Cs = np.logspace(-3, 3, 50)
    gammas = np.logspace(-3, 3, 50)
    ACC_ = np.zeros([len(Cs), len(gammas)])
    DEV_ = np.zeros([len(Cs), len(gammas)])
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            acc2 = crossvalidation(Cs[i], gammas[j], feature_train, label_train, kf)
            ACC_[i, j] = np.mean(acc2)
            DEV_[i, j] = np.std(acc2)

    # plt.figure()
    # plt.imshow(ACC_, origin='lower')
    # plt.colorbar()
    # plt.show()
    # plt.savefig('fig')
    find_bestpair(ACC_, DEV_)
    return ACC_, DEV_


# 2b
validation()

# 2c
print('-----------------------')
acc = np.zeros([20, 50, 50])
dev = np.zeros([20, 50, 50])
for T in range(20):
    (a, b) = validation()
    acc[T, :] = a
    dev[T, :] = b

print('-----------------------')
acc_aver = np.zeros([50, 50])
dev_aver = np.zeros([50, 50])
for i in range(50):
    for j in range(50):
        acc_aver[i, j] = (np.mean(acc[:, i, j]))
        dev_aver[i, j] = (np.mean(dev[:, i, j]))

final_c, final_gamma, final_mean, final_deviation = find_bestpair(acc_aver, dev_aver)

# 2d
print('-------------')
clf = svm.SVC(C=final_c, kernel='rbf', gamma=final_gamma)
clf.fit(feature_train, label_train)
predict_label_test = clf.predict(feature_test)
predict_label_train = clf.predict(feature_train)
acc_test = accuracy_score(label_test, predict_label_test)
acc_train = accuracy_score(label_train, predict_label_train)
print('accuracy on train:', acc_train)
print('accuracy on test:', acc_test)
print('final_mean:', final_mean)
print('final_deviation:', final_deviation)
