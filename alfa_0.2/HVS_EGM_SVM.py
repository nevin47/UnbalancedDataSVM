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
    print "\tstep2.1: 计算指标信息增益"
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
    arrayValue = [0.6,0.5,0.5,0.1,1,1,1]
    train_X, train_label = Hyper.balanceDataforGA(arrayValue,*data)
    # train_X, train_label = Hyper.balanceData(dataSet1, labels1, dataSet2, labels2, initArray)

    print "step 3: 训练..."
    clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)

    print "step 4: 测试..."
    pre = HSVM.testSVM(preTestX , clf)
    tt = np.array(preLabels, dtype="float64")


    print "Final G-means:",basef.testSample(pre, tt)
    print basef.testSampleShow(pre, tt)
    # return basef.testSample(pre, tt)
    return basef.testSampleShow(pre, tt)

if __name__ == "__main__":
    # demo
    filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/test/movietest1.csv' # 设置读取文件
    # filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/Heart2.csv'
    scaler = 1 # 决定是否归一化数据
    MAXFEATURENUM = 5 # 设置指标离散最大值
    main(filename, scaler, MAXFEATURENUM, kernel='rbf', C=20.0, gamma= 1)

# test

# print dataSet1