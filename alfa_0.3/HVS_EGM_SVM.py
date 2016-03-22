# coding:utf-8
__author__ = 'nevin47'

import numpy as np
import basefunc.basefunc as basef
import entropy.entropy as en
import HVS.HVS as Hyper
import svm.SVM_Core as HSVM
import random
from scipy.optimize import minimize
import matplotlib.pyplot as pl

def main(filename, scaler, MAXFEATURENUM, **svmParameter):
    # 获取基本参数
    kernel = svmParameter['kernel']
    C = svmParameter['C']
    gamma = svmParameter['gamma']


    # 读取数据,Data1一定是多数类
    dataSet1, labels1, dataSet2, labels2 = basef.readData(filename, scaler)

    # 鲁棒测试开关
    flag = 0
    if(flag == 1):
        pickoutArray = range(len(labels1))
        pickoutNum = round(len(labels1) * 0.1)
        if(len(labels2) < pickoutNum):
            dataSet2 = dataSet2[:]
            labels2 = labels2[:]
        else:
            PICKdata = []
            PICKlabels = []
            x = 0
            for i in pickoutArray:
                if(random.randrange(0,2) == 1 and x <= pickoutNum):
                    PICKdata.append(dataSet2[i])
                    PICKlabels.append(labels2[i])
                x += 1
            dataSet2 = np.array(PICKdata[:])
            labels2 = np.array(PICKlabels[:])



    # print "step 2: 预处理数据..."
    # print "\tstep2.1: 计算指标信息增益"
    initArray = en.featureSample(dataSet1, labels1, dataSet2, labels2, MAXFEATURENUM)
    print "初始化指标:",initArray

    preTestX = np.vstack([dataSet1,dataSet2])
    preLabels = np.hstack([labels1,labels2])


    # print "\tstep2.2: 平衡数据"
    # Todo:此处可以直接调用遗传算法模块了 -- By nevin47
    data = (dataSet1, labels1, dataSet2, labels2, initArray)
    arrayValue = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1]
    train_X, train_label = Hyper.balanceDataforGA(arrayValue,*data)

    # print "step 3: 训练..."
    clf = HSVM.trainSVM(train_X, train_label, kernel=kernel, C=C, gamma= gamma)

    # print "step 4: 测试..."
    pre = HSVM.testSVM(preTestX , clf)
    #proba = HSVM.testSVMwithProb(preTestX, clf)
    tt = np.array(preLabels, dtype="float64")


    # print "Final G-means:",basef.testSample(pre, tt)
    #print "\nPRO:",proba,"\n"
    print "\nPRE:",pre,"\n"
    print basef.testSampleShow(pre, tt)
    return basef.testSampleShow(pre, tt)

if __name__ == "__main__":
    # demo
    filename1 = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/test/wpbc.csv' # 设置读取文件
    filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/CreditOriginData2.csv'
    # filename = '/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/Heart2.csv'
    scaler = 1 # 决定是否归一化数据
    MAXFEATURENUM = 5 # 设置指标离散最大值
    SUMG = []
    SUMF = []
    for i in range(30):
        tempG,tempF = main(filename, scaler, MAXFEATURENUM, kernel='rbf', C=15.0, gamma= 1)
        SUMG.append(tempG)
        SUMF.append(tempF)
    print "AVG-G: %f,AVG-F %f",sum(SUMG)/30.0,sum(SUMF)/30.0

# test

# print dataSet1