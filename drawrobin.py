# coding:utf-8

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

reader = csv.reader(file('/Users/nevin47/Desktop/Project/Academic/Code/Python/SVM/UnbalancedDataSVM/RobinTest.csv','rb'))
robintest = []
for line in reader:
    print robintest.append(line)
robintest = np.array(robintest,dtype='float64')
ro = robintest.T
min_max_scaler = preprocessing.MinMaxScaler()  # 设置归一化
ro = min_max_scaler.fit_transform(ro)  # 归一化

print ro
N = 9
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
haberman = ro[:,0]
german = ro[:,1]
pima = ro[:,2]
wpbc = ro[:,3]
print haberman

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, haberman, width, color='red')
p2 = plt.bar(ind, german, width, color='black',bottom=haberman)
p3 = plt.bar(ind, pima, width, color='blue',bottom=german)
p4 = plt.bar(ind, wpbc, width, color='g',bottom=pima)


plt.ylabel('G-Means')
plt.title('Scores by G-Means')
plt.xticks(ind + width/2., ('SVM','SPU','RU','SMOTE','BSMOTE','WEIGHT','RUS','ADSYN','HVSEGASVM'))
plt.yticks(np.arange(0, 10, 10))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()