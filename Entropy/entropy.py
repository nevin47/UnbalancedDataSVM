# coding:utf-8
__author__ = 'nevin47'

from math import log
import collections as co
import numpy as np
import random
import csv
import math
from sklearn import svm
from sklearn import preprocessing


## Base Function

base = [str(x) for x in range(10)] + [ chr(x) for x in range(ord('A'),ord('A')+6)]
def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num,rem = divmod(num, 2)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])

def toInterval(originalInput, randLength, method = 'solid'):
    '''
    区间化函数
    :param originalInput:精确数据
    :param randLength:区间化水平，数组
    :param method:区间化方法 solid:固定区间化 random:动态区间化
    :return:区间化数据
    '''
    output = []
    for index, tempDemention in enumerate(originalInput):
        if(method == 'solid'):
            tempLow = float(tempDemention) - randLength[index]
            tempHigh = float(tempDemention) + randLength[index]
            if(tempLow == tempHigh):
                tempNewDemention = tempLow
            else:
                tempNewDemention = [tempLow, tempHigh]
            output.append(tempNewDemention)
        elif(method == 'random'):
            tempLow = float(tempDemention) - random.uniform(0,randLength[index])
            tempHigh = float(tempDemention) + random.uniform(0,randLength[index])
            if(tempLow == tempHigh):
                tempNewDemention = tempLow
            else:
                tempNewDemention = [tempLow, tempHigh]
            output.append(tempNewDemention)
    return output

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
        #tempLabel = line[-1]
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

def hyperSampler(intervalSample, randLength):
    '''
    区间数重采样函数
    :param intervalSample:区间数样本
    :param randLength:区间化水平
    :return:
    '''
    intervalNO = [] #样本是否使用区间数
    for index,j in enumerate(randLength):
        if(j == 0):
            intervalNO.append(0)
        else:
            intervalNO.append(1)
    shape = co.Counter(randLength) #获取区间化水平的计数
    SampleShape = len(randLength) - shape[0] #获取有效区间个数
    ResultTemp = []
    ResultFinal = []
    for i in range(2**SampleShape):
        temp = dec2bin(i)
        alllen = SampleShape
        resultlen = len(temp)
        Origin = []
        for j in range(alllen - resultlen):
            Origin.append(0)
        FR = Origin + list(temp)
        ResultTemp.append(FR)
    for i in ResultTemp:
        TempMiddle =[]
        lastPoint = 0
        for j in i:
            while(intervalNO[lastPoint] == 0):
                TempMiddle.append(intervalSample[lastPoint])
                lastPoint += 1
            TempMiddle.append(intervalSample[lastPoint][int(j)])
            lastPoint += 1
        if(lastPoint < len(intervalSample)):
            TempMiddle += intervalSample[lastPoint:]
        ResultFinal.append(TempMiddle)
    return ResultFinal

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
        if(num == 2):
            FN += 1
        elif(num == -2):
            FP += 1
    TP = allP - FP
    TN = allN - FN
    TPR = float(TP) / (TP + FN)
    TNR = float(TN) / (TN + FP)
    precision = float(TP) / (TP + FP)
    Gmeans = math.sqrt( TPR * TNR)
    Fmeasure = 2 * TPR * precision / (TPR + precision)
    return Gmeans, Fmeasure

def balanceData(dataSet1,labels1,dataSet2,labels2,randArray):
    num1 = len(labels1)
    num2 = len(labels2)
    newNum1 = int(num1 * 1)
    newNum2 = int(num2 * 1)

    train_X = dataSet1[:newNum1]
    train_label = labels1[:newNum1]

    # 区间化补足
    temptrain_Xx = []
    for index , tempdata in enumerate(dataSet2[:newNum2]):
        t = toInterval(tempdata, randArray, method='random')
        temptrain_Xx.append(t)

    for tempXx in temptrain_Xx:
        inputa = hyperSampler(tempXx, randArray)
        for i in inputa:
            train_X = np.concatenate((train_X, np.mat(i)))
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
        train_label += [-1]
    return train_X,train_label

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)  # log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        # iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

#def dataDisperse(AllData):




# preWork
#设置区间化水平
randArray = [0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0]

## TEST AREA

## TEST AREA END

# step 1: load data
print "step 1: load data..."

dataSet1, labels1, dataSet2, labels2 = readData('/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/DataSet/CreditOriginData2.csv',1)

AllData = np.vstack([dataSet1,dataSet2]) # 组合数据
AllLabel = np.hstack([labels1,labels2]) # 组合标签

FinalData = np.column_stack((AllData,AllLabel))
print AllData
print FinalData
print chooseBestFeature(FinalData.tolist())
