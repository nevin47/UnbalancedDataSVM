__author__ = 'nevin47'

from numpy import *
import random
import matplotlib.pyplot as plt
import HyperInterval as HN
from sklearn import svm

################## test svm #####################
## step 1: load data
print "step 1: load data..."
dataSet = []
labels = []
fileIn = open('/Users/nevin47/Desktop/Python Test/iris.data')
for line in fileIn.readlines():
	lineArr = line.strip().split(',')
	newX = [float(lineArr[0]) - random.uniform(0,1),float(lineArr[0]) + random.uniform(0,1)]
	newY = [float(lineArr[1]) - random.uniform(0,1),float(lineArr[1]) + random.uniform(0,1)]
	newZ = [float(lineArr[2]) - random.uniform(0,1),float(lineArr[2]) + random.uniform(0,1)]
	newW = [float(lineArr[3]) - random.uniform(0,1),float(lineArr[3]) + random.uniform(0,1)]
	dataSet.append([newX,newY,newZ,newW])
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))
	labels.append(float(lineArr[4]))

NewdataSet = []
for i in dataSet:
	newI = HN.HyperIntervalNumber(i)
	out = newI.GetAllPosPoint()
	for j in out:
		NewdataSet.append(j)
print "The test data is:"
print NewdataSet

dataSet = mat(NewdataSet)
print len(dataSet)
labels = mat(labels).T
train_x = dataSet[0:800, :]
train_y = ravel(labels[0:800, :])
test_x = dataSet[801:1599, :]
test_y = ravel(labels[801:1599, :])

## step 2: training...
print "step 2: training..."
C = 0.6
toler = 0.001
maxIter = 50

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(train_x, train_y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]

M = clf.predict(test_x)

for j,i in enumerate(M):
	print i