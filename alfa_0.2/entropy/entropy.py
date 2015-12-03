# coding:utf-8
__author__ = 'nevin47'

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from math import log

# basic function for feature choose
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

def chooseFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    gainArray = []
    for i in range(numFeatures):        # iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        gainArray.append(infoGain)
    return gainArray                      #returns an integer
# end of basic function

def getChosenNum(unB):
    '''
    简化算法求最适合的区间化指标数
    :param unB:非均衡程度
    :return:最优区间化指标数
    '''
    testArray = [2,4,8,16,32,64]
    nearestDis = 99999.0
    nearestIndex = 0.0
    for index, i in enumerate(testArray):
        testDis = abs(unB - i)
        if(testDis < nearestDis):
            nearestDis = testDis
            nearestIndex = index + 1
    return nearestIndex

def featureDiscrete(feature, MAXFEATURENUM):
    '''
    指标离散函数
    :param feature:指标下数据
    :return:
    '''
    # first step: To determine whether the feature need to be discrete
    difArray = np.unique(feature)
    if len(difArray) > MAXFEATURENUM:
        cutArray = pd.cut(feature, MAXFEATURENUM, labels=[i for i in range(MAXFEATURENUM)])
        return cutArray._codes
    else:
        return feature

def featureSample(dataSet1, labels1, dataSet2, labels2, MAXFEATURENUM):
    '''
    基于信息增益的指标筛选函数
    :param dataSet1:数据集1
    :param labels1:标签1
    :param dataSet2:数据集2
    :param labels2:标签2
    :param MAXFEATURENUM:最大指标离散程度
    :return:筛出的关键指标序列
    '''
    # 先计算非均衡程度
    numN = np.shape(dataSet1)[0]
    numP = np.shape(dataSet2)[0]
    unB = float(numN) / float(numP)
    if(unB < 1):
        unB = 1 / unB
    chosenNum = getChosenNum(unB) # 获取区间化指标数
    # 将两部分矩阵进行组合
    allData = np.vstack([dataSet1,dataSet2]) # 组合数据
    allLabel = np.hstack([labels1,labels2]) # 组合标签
    featureNum = np.shape(allData)[1]
    # 将连续指标离散化
    resMatrix = []
    for i in range(featureNum):
        tempFeature = featureDiscrete(allData[:,i], MAXFEATURENUM)
        resMatrix.append(tempFeature)
    resMatrix = np.array(resMatrix).T # 不带label的数据矩阵
    resMatrixL = np.column_stack((resMatrix,allLabel)) # 带label的数据矩阵
    # 计算信息增益
    featureGainList = chooseFeature(resMatrixL.tolist())
    # 获取关键指标排序
    gainSort = np.argsort(featureGainList)
    # 生成初始化区间化力度数组
    ## old version
    initArray = np.zeros(featureNum)
    for i in range(chosenNum):
        initArray[gainSort[i]] = 1
    return initArray
    ## new version
    # initArray = []
    # for i in range(chosenNum):
    #     initArray.append(gainSort[i])
    # return initArray

