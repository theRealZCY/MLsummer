# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve as plc

start_time = interval = time.time()

train = pd.read_csv('train.csv')

print "File reading time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

# print "calculating timestamps\n"
# train["day_in_week"] = train["time_min"].astype(int) // 1440 % 7
# train["hour_in_day"] = train["time_min"].astype(int) // 60 % 24
# test["day_in_week"] = test["time_min"].astype(int) // 1440 % 7
# test["hour_in_day"] = test["time_min"].astype(int) // 60 % 24

print "processing data"
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(np.mean(train["Age"]))
data = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
targets = train["Survived"].values
offset = int(0.7*len(train))
x_train, y_train = data[:offset], targets[:offset]
x_test, y_test = data[offset:], targets[offset:]
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

print "Data processing time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

print "trainning"
variance = ['linear', 'poly', 'rbf']
clfs = []
for var in variance:
    clfs.append(svm.SVC(kernel = var, degree = 2))

for ind, clf in enumerate(clfs):
    clf.fit(x_train, y_train)
    print "clf %s kernel training time: %.4f seconds" % (variance[ind], time.time() - interval)
    interval = time.time()

print "\nTruncating and Testing data..."
increment = 5000
i = 0
scores = np.empty((len(clfs), 0)).tolist()
while i < x_test.shape[0]:
    end = i + 5000 if i + 5000 < x_test.shape[0] else x_test.shape[0]

    for ind, clf in enumerate(clfs):
        score = clf.score(x_test[i:end, :], y_test[i:end,])
        scores[ind].append(score)
        msg1 = "clf %d increment testing time: %.3f seconds. " % (ind, time.time() - interval)
        interval = time.time()
        msg2 = "kernel: %s, score: %.3f" % (variance[ind], score)
        print msg1 + msg2

    print "Data Segment Test Finished\n"
    i = end

for ind, var in enumerate(variance):
    print "%s kernel mean score: %.3f" % (variance[ind], np.mean(scores[ind]))

print "\npreparing learning curves"
for ind, var in enumerate(variance):
    plc(clfs[ind], "learning_curve_%s" % var, x_train, y_train, None, None,
                        1, np.linspace(.3, 1.0, 5))
    print "curve_%d finished, time elapsed:  %.4f seconds" % (ind, time.time() - interval)
    interval = time.time()

plt.show()

print("\nTime elapsed time overall: %.4f seconds" % (time.time() - start_time))