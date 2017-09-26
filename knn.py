# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve as plc

start_time = interval = time.time()

train = pd.read_csv('datatraining.txt.csv')
test = pd.read_csv('datatest2copy.csv')

print "File reading time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

print "processing data"

x_train = train[["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values
y_train = train["Occupancy"].values

x_test = test[["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values
y_test = test["Occupancy"].values

print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

print "Data processing time: %.4f seconds\n" % (time.time() - interval)
interval = time.time()

print "trainning"
variance = np.arange(2, 31, 3)
clfs = []
for var in variance:
    clfs.append(KNeighborsClassifier(n_neighbors=var))

for ind, clf in enumerate(clfs):
    clf.fit(x_train, y_train)
    print "clf %s neighbors training time: %.4f seconds" % (variance[ind], time.time() - interval)
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
        msg2 = "neighbors: %s, score: %.3f" % (variance[ind], score)
        print msg1 + msg2

    print "Data Segment Test Finished\n"
    i = end

for ind, var in enumerate(variance):
    print "%s neighbors mean score: %.3f" % (variance[ind], np.mean(scores[ind]))

#print "\npreparing learning curves"
#for ind, var in enumerate(variance):
#    plc(clfs[ind], "learning_curve_%snn" % var, x_train, y_train, None, None,
#                        1, np.linspace(.3, 1.0, 5))
#   print "curve_%d finished, time elapsed:  %.4f seconds" % (ind, time.time() - interval)
#    interval = time.time()

#plt.show()

print("\nTime elapsed time overall: %.4f seconds" % (time.time() - start_time))