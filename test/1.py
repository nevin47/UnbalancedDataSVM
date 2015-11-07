# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import collections as co
import numpy as np
import random
import csv
import math
from sklearn import preprocessing

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

def readData(filename):
    '''
    按照不同类别读出数据
    :param filename: 文件名
    :return: 不同类别的数据
    '''
    dataSet1 = []
    labels1 = []
    dataSet2 = []
    labels2 = []
    reader = csv.reader(file(filename, 'rb'))
    for line in reader:
        tempdata = np.array(line[1:-1], dtype='float64')
        tempLabel = line[-1]
        if(tempLabel == '1'):
            #插入数据
            dataSet1.append(tempdata)
            #插入类别标签
            labels1.append(tempLabel)
        elif(tempLabel == '-1'):
            #插入数据
            dataSet2.append(tempdata)
            #插入类别标签
            labels2.append(tempLabel)
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


#from sklearn.linear_model import SGDClassifier

# we create 40 separable points
# rng = np.random.RandomState(0)
# n_samples_1 = 500
# n_samples_2 = 50
# X = np.r_[1.5 * rng.randn(n_samples_1, 2),
#           0.5 * rng.randn(n_samples_2, 2) + [3, 3]]
# y = [0] * (n_samples_1) + [1] * (n_samples_2)

X = np.load("x.npy")
y = np.load("y.npy")


Xnew = []
ynew = []
for i in range(len(y)):
    if(y[i] == 0):
        tx = X[i,0]
        tx2 = X[i,1]
        Xnew.append([tx,tx2])
        ynew.append(y[i])
    else:
        rd = float(random.uniform(0,0.1))
        rd2 = float(random.uniform(0,0.1))
        tx1 = X[i,0] - rd
        tx2 = X[i,0] + rd
        tx3 = X[i,1] - rd2
        tx4 = X[i,1] + rd2
        Xnew.append([tx1,tx3])
        Xnew.append([tx1,tx4])
        Xnew.append([tx2,tx3])
        Xnew.append([tx2,tx4])
        ynew.append(1)
        ynew.append(1)
        ynew.append(1)
        ynew.append(1)

Xnew = np.array(Xnew)
# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(Xnew, ynew)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]


# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

# plot separating hyperplanes and samples
j =0
h0 = plt.plot(xx, yy, 'k-', label='no weights')
# h1 = plt.plot(xx, wyy, 'k--', label='with weights')
for i in range(len(ynew)):
    if(ynew[i] == 0):
        plt.scatter(Xnew[i, 0], Xnew[i, 1], c=ynew[i])
        j += 1
    else:
        plt.scatter(Xnew[i, 0], Xnew[i, 1], c=ynew[i], marker='x')
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
print '111',j
plt.legend()

plt.axis('tight')
plt.show()
print '111',j