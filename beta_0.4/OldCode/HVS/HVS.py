# coding:utf-8
__author__ = 'nevin47'

import collections as co
import numpy as np
import random


## Base Function

base = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A') + 6)]

def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 2)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])

def toInterval(originalInput, randLength, method='solid'):
    '''
    区间化函数
    :param originalInput:精确数据
    :param randLength:区间化水平，数组
    :param method:区间化方法 solid:固定区间化 random:动态区间化
    :return:区间化数据
    '''
    output = []
    for index, tempDemention in enumerate(originalInput):
        if (method == 'solid'):
            tempLow = float(tempDemention) - randLength[index]
            tempHigh = float(tempDemention) + randLength[index]
            if (tempLow == tempHigh):
                tempNewDemention = tempLow
            else:
                tempNewDemention = [tempLow, tempHigh]
            output.append(tempNewDemention)
        elif (method == 'random'):
            tempLow = float(tempDemention) - random.uniform(0, randLength[index])
            tempHigh = float(tempDemention) + random.uniform(0, randLength[index])
            if (tempLow == tempHigh):
                tempNewDemention = tempLow
            else:
                tempNewDemention = [tempLow, tempHigh]
            output.append(tempNewDemention)
    return output

def hyperSampler(intervalSample, randLength):
    '''
    区间数重采样函数
    :param intervalSample:区间数样本
    :param randLength:区间化水平
    :return:
    '''
    intervalNO = []  # 样本是否使用区间数
    for index, j in enumerate(randLength):
        if (j == 0):
            intervalNO.append(0)
        else:
            intervalNO.append(1)
    shape = co.Counter(randLength)  # 获取区间化水平的计数
    SampleShape = len(randLength) - shape[0]  # 获取有效区间个数
    ResultTemp = []
    ResultFinal = []
    for i in range(2 ** SampleShape):
        temp = dec2bin(i)
        alllen = SampleShape
        resultlen = len(temp)
        Origin = []
        for j in range(alllen - resultlen):
            Origin.append(0)
        FR = Origin + list(temp)
        ResultTemp.append(FR)
    for i in ResultTemp:
        TempMiddle = []
        lastPoint = 0
        for j in i:
            while (intervalNO[lastPoint] == 0):
                TempMiddle.append(intervalSample[lastPoint])
                lastPoint += 1
            TempMiddle.append(intervalSample[lastPoint][int(j)])
            lastPoint += 1
        if (lastPoint < len(intervalSample)):
            TempMiddle += intervalSample[lastPoint:]
        ResultFinal.append(TempMiddle)
    return ResultFinal

def balanceData(dataSet1, labels1, dataSet2, labels2, randArray):
    train_X = dataSet1[:]
    train_label = labels1[:]
    featureNumT = co.Counter(randArray)
    featureNum = len(randArray) - featureNumT[0]
    # 区间化补足
    temptrain_Xx = []
    for index, tempdata in enumerate(dataSet2):
        t = toInterval(tempdata, randArray, method='random')
        temptrain_Xx.append(t)

    for tempXx in temptrain_Xx:
        inputa = hyperSampler(tempXx, randArray)
        for i in inputa:
            train_X = np.concatenate((train_X, np.mat(i)))

        for j in range(2 ** featureNum):
            train_label += [-1]
    return train_X, train_label

def balanceDataforGA(arrayValue, *data):
    '''
    The Balance Function for GA
    :param arrayValue:the interval Array only with value
    :param data:the set of data & initArray
    :return: train_X, train_label
    '''
    dataSet1, labels1, dataSet2, labels2, initArray = data # unpack the data&labels
    j = 0
    # Set the arrayValue into the initArray
    for index,value in enumerate(initArray):
        if(value != 0):
            # initArray[index] = arrayValue[j]
            initArray[index] = arrayValue[random.randrange(0,6)]  # Happy!
            j += 1
    print "GAINIT",initArray
    return balanceData(dataSet1, labels1, dataSet2, labels2, initArray)
