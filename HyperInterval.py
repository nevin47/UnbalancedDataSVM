__author__ = 'nevin47'

from numpy import *
import matplotlib.pyplot as plt



#DEC to BIN
base = [str(x) for x in range(10)] + [ chr(x) for x in range(ord('A'),ord('A')+6)]
def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num,rem = divmod(num, 2)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])

class HyperIntervalNumber:
    ##The hyper interval number would use 2 pemeter to describe
    def __init__(self,OriginIntervalNum):
        self.IntervalNum = mat(OriginIntervalNum)
        self.IntervalNum = self.IntervalNum.astype(float64)
        self.SampleShape = self.IntervalNum.shape

    def GetAllPosPoint(self):
        ##get all the posibility of the choice
        ResultTemp = []
        ResultFinal = []
        for i in range(2**self.SampleShape[0]):
            temp = dec2bin(i)
            alllen = self.SampleShape[0]
            resultlen = len(temp)
            Origin = []
            for j in range(alllen - resultlen):
                Origin.append(0)
            FR = Origin + list(temp)
            ResultTemp.append(FR)
        for i in ResultTemp:
            TempMiddle =[]
            for j in range(len(i)):
                TempMiddle.append(self.IntervalNum[j,i[j]])
            ResultFinal.append(TempMiddle)
        return ResultFinal

    def CalMiddlePoint(self):
        TargetPoint = []
        for i in self.IntervalNum:
            Lower = i[0,0]
            Higher = i[0,1]
            Middle = (Lower + Higher)/2
            TargetPoint.append(Middle)
        return TargetPoint

