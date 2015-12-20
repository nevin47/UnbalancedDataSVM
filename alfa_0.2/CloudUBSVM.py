# coding:utf-8
__author__ = 'nevin47'

import numpy as np
import basefunc.basefunc as basef
import entropy.entropy as en
import HVS.HVS as Hyper
import svm.SVM_Core as HSVM
import csv
import collections as co
import math

from scipy.optimize import minimize
import matplotlib.pyplot as pl

def main(filename, scaler, MAXFEATURENUM, **svmParameter):
    # 获取基本参数
    kernel = svmParameter['kernel']
    C = svmParameter['C']
    gamma = svmParameter['gamma']

    print "step 1: 读取数据..."


    cloudReader = csv.reader(file("/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/cloudSVM/Dataset/cloud.csv",'rb'))
    Cloud = []
    Origin = []

    for i in cloudReader:
        Cloud.append(i)

    originReader =  csv.reader(file("/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/cloudSVM/Dataset/Origin.csv",'rb'))

    for j in originReader:
        Origin.append(j)
    Cloud = np.array(Cloud)
    Origin = np.array(Origin,dtype='float64')

    for i in range(10):
        testSample = Origin[:,i]

        # print testSample
        lowIndex = []
        highIndex = []
        x = 0
        for index,score in enumerate(testSample):
            if score > 3:
                highIndex.append(index)
                x += 1
            elif score < 3 and score != 0:
                lowIndex.append(index)

        highCloud = []
        lowCloud = []
        # 导入云数据
        for m in lowIndex:
            lowCloud.append(Cloud[m])
        for n in highIndex:
            highCloud.append(Cloud[n])

        highCloud = np.array(highCloud)
        lowCloud = np.array(lowCloud)
        if len(highCloud) > len(lowCloud):
            highlabels = [1 for i in range(len(highCloud))]
            lowlabels = [-1 for i in range(len(lowCloud))]
            dataSet1, labels1, dataSet2, labels2 = highCloud,highlabels,lowCloud,lowlabels
        else:
            highlabels = [-1 for i in range(len(highCloud))]
            lowlabels = [1 for i in range(len(lowCloud))]
            dataSet1, labels1, dataSet2, labels2 = lowCloud,lowlabels,highCloud,highlabels

        initArray = en.featureSample(dataSet1, labels1, dataSet2, labels2, MAXFEATURENUM)


        preTestX = np.vstack([dataSet1,dataSet2])
        preLabels = np.hstack([labels1,labels2])
        # GMM = []
        # for i in range(500):
        #     ii = i + 1
        #     tempi = ii/500.0
        #     arrayValue = [0.1,0.1,1,0.1]
        #     arrayValue[1] = tempi
        #     print "step2.2: 平衡数据"
        #     data = (dataSet1, labels1, dataSet2, labels2, initArray)
        #
        #     train_X, train_label = Hyper.balanceDataforGA(arrayValue,*data)
        #
        #     print "step 3: 训练..."
        #     clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)
        #
        #     print "step 4: 测试..."
        #     pre = HSVM.testSVM(preTestX , clf)
        #     tt = np.array(preLabels, dtype="float64")
        #     Gm = basef.testSample(pre, tt)
        #     GMM.append(Gm)
        # pl.plot(range(500),GMM)
        # pl.show()


        print "\tstep2.2: 平衡数据"
        # Todo:此处可以直接调用遗传算法模块了 -- By nevin47
        data = (dataSet1, labels1, dataSet2, labels2, initArray)
        arrayValue = [1,1,1,1,1,1,1]
        train_X, train_label = Hyper.balanceDataforGA(arrayValue,*data)
        # train_X, train_label = Hyper.balanceData(dataSet1, labels1, dataSet2, labels2, initArray)

        print "step 3: 训练..."
        clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)

        print "step 4: 测试..."
        pre = HSVM.testSVM(preTestX , clf)
        trainLabel = np.array(preLabels, dtype="float64")

        #######
        testCount = co.Counter(trainLabel)
        allN = testCount[1]
        allP = testCount[-1]
        testResult = trainLabel - pre
        FN = 0
        FP = 0
        for index, num in enumerate(testResult):
            if (num == 2):
                FN += 1
            elif (num == -2):
                FP += 1
        TP = allP - FP
        TN = allN - FN
        if (TP+FN) == 0:
            TPR = 0
        else:
            TPR = float(TP) / (TP + FN)
        TNR = float(TN) / (TN + FP)
        precision = float(TP) / (TP + FP)
        Gmeans = math.sqrt(TPR * TNR)
        if (precision+TPR) == 0:
            Fmeasure = 0
        else:
            Fmeasure = 2 * TPR * precision / (TPR + precision)
        print Gmeans,Fmeasure

if __name__ == "__main__":
    # demo
    filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/test/wpbc2.csv' # 设置读取文件
    # filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/Heart2.csv'
    scaler = 1 # 决定是否归一化数据
    MAXFEATURENUM = 5 # 设置指标离散最大值
    main(filename, scaler, MAXFEATURENUM, kernel='rbf', C=5.0, gamma= 1)

# test

# print dataSet1