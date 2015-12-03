# coding:utf-8
__author__ = 'nevin47'

import numpy as np
import basefunc.basefunc as basef
import entropy.entropy as en
import HVS.HVS as Hyper
import svm.SVM_Core as HSVM

from scipy.optimize import minimize
import matplotlib.pyplot as pl

def main(filename, scaler, MAXFEATURENUM, **svmParameter):
    # 获取基本参数
    kernel = svmParameter['kernel']
    C = svmParameter['C']
    gamma = svmParameter['gamma']

    print "step 1: 读取数据..."

    # 读取数据
    dataSet1, labels1, dataSet2, labels2 = basef.readData(filename, scaler)

    print "step 2: 预处理数据..."
    print "step2.1: 计算指标信息增益"
    initArray = en.featureSample(dataSet1, labels1, dataSet2, labels2, MAXFEATURENUM)


    preTestX = np.vstack([dataSet1,dataSet2])
    preLabels = np.hstack([labels1,labels2])
    GMM = []
    for i in range(1000):
        ii = i + 1
        tempi = ii/1000.0
        initArray[5] = tempi
        print "step2.2: 平衡数据"
        train_X, train_label = Hyper.balanceData(dataSet1, labels1, dataSet2, labels2, initArray)

        print "step 3: 训练..."
        clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)

        print "step 4: 测试..."
        pre = HSVM.testSVM(preTestX , clf)
        tt = np.array(preLabels, dtype="float64")
        Gm = basef.testSample(pre, tt)
        GMM.append(Gm)
    pl.plot(range(1000),GMM)
    pl.show()


    # print "step2.2: 平衡数据"
    # train_X, train_label = Hyper.balanceData(dataSet1, labels1, dataSet2, labels2, initArray)
    #
    # print "step 3: 训练..."
    # clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)
    #
    # print "step 4: 测试..."
    # pre = HSVM.testSVM(train_X , clf)
    # tt = np.array(train_label, dtype="float64")
    #
    #
    # print basef.testSample(pre, tt)


if __name__ == "__main__":
    # demo
    filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/CreditOriginData2.csv' # 设置读取文件
    scaler = 1 # 决定是否归一化数据
    MAXFEATURENUM = 5 # 设置指标离散最大值
    main(filename, scaler, MAXFEATURENUM, kernel='rbf', C=5.0, gamma= 1)

# test

# print dataSet1