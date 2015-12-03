# coding:utf-8
__author__ = 'nevin47'
from sklearn import svm
import numpy as np


def trainSVM(dataSet, trainLabel, **svmParameter):
    kernel = svmParameter['kernel']
    C = svmParameter['C']
    gamma = svmParameter['gamma']
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(dataSet, trainLabel)
    return clf


def testSVM(dataSet, model):
    pre = np.float64(model.predict(dataSet))
    return pre