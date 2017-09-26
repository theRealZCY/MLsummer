# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve as plc

start_time = interval = time.time()

train = pd.read_csv('datatraining_copy.csv')
#test = pd.read_csv('datatest2copy.csv')

print "File reading time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

# print "calculating timestamps\n"
# train["day_in_week"] = train["time_min"].astype(int) // 1440 % 7
# train["hour_in_day"] = train["time_min"].astype(int) // 60 % 24
# test["day_in_week"] = test["time_min"].astype(int) // 1440 % 7
# test["hour_in_day"] = test["time_min"].astype(int) // 60 % 24

print "processing data"
data = train[["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values
targets = train["Occupancy"].values

offset = int(0.7*len(train))
x_train, y_train = data[:offset], targets[:offset]
x_test, y_test = data[offset:], targets[offset:]

print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

print "Data processing time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

print "training"
variance = ['linear', 'poly', 'rbf']
clfs = []
for var in variance:
    #change the c here
    clfs.append(svm.SVC(C=1, kernel = var, degree=2))

for ind, clf in enumerate(clfs):
    clf.fit(x_train, y_train)
    print "clf %s kernel training time: %.4f seconds" % (variance[ind], time.time() - interval)
    interval = time.time()

print "\nTruncating and Testing data..."
increment = 4000
i = 0
scores = np.empty((len(clfs), 0)).tolist()
while i < x_test.shape[0]:
    end = i + 4000 if i + 4000 < x_test.shape[0] else x_test.shape[0]

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
    if var == 'rbf':
        plc(clfs[ind], "learning_curve_%s" % var, x_train, y_train, None, None,
                            1, np.linspace(.3, 1.0, 5))
        print "curve_%d finished, time elapsed:  %.4f seconds" % (ind, time.time() - interval)
        interval = time.time()

plt.show()

print("\nTime elapsed time overall: %.4f seconds" % (time.time() - start_time))