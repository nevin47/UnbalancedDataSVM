# coding:utf-8
__author__ = 'nevin47'

import math
import collections as co
import numpy as np
import csv
from sklearn import preprocessing

def readData(filename, scaler = 1):
    '''
    按照不同类别读出数据
    :param filename: 文件名
    :param scaler: 是否归一化
    :return: 不同类别的数据
    '''
    dataSet1 = []
    labels1 = []
    dataSet2 = []
    labels2 = []
    reader = csv.reader(file(filename, 'rb'))
    for line in reader:
        tempdata = np.array(line[1:-1], dtype='float64')
        tempLabel = np.array(line[-1], dtype='float64')
        # tempLabel = line[-1]
        if(tempLabel == 1):
            # 插入数据
            dataSet1.append(tempdata)
            # 插入类别标签
            labels1.append(tempLabel)
        elif(tempLabel == -1):
            # 插入数据
            dataSet2.append(tempdata)
            # 插入类别标签
            labels2.append(tempLabel)
    if(scaler == 1):
        min_max_scaler = preprocessing.MinMaxScaler()  # 设置归一化
        dataSet1 = min_max_scaler.fit_transform(dataSet1)  # 归一化
        dataSet2 = min_max_scaler.fit_transform(dataSet2)
    return dataSet1, labels1, dataSet2, labels2

def testSample(pre, test):
    '''
    测试分类器准确度
    :param pre: Array,预测结果
    :param test: Array,实际分类
    :return:Gmeans，Fmeasure
    '''
    testCount = co.Counter(test)
    allN = testCount[1]
    allP = testCount[-1]
    testResult = test - pre
    FN = 0
    FP = 0
    for index, num in enumerate(testResult):
        if (num == 2):
            FN += 1
        elif (num == -2):
            FP += 1
    TP = allP - FP
    TN = allN - FN
    TPR = float(TP) / (TP + FN)
    TNR = float(TN) / (TN + FP)
    precision = float(TP) / (TP + FP)
    Gmeans = math.sqrt(TPR * TNR)
    Fmeasure = 2 * TPR * precision / (TPR + precision)
    # return Gmeans, Fmeasure
    return Gmeans