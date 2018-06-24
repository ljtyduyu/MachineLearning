#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import csv
from numpy import tile
from sklearn import preprocessing
from collections import Counter
from collections import OrderedDict
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from time import time

'''KNN分类器'''

def Data_Input():
    data = pd.read_csv("D:/Python/File/iris.csv")
    data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
    print(data.shape,'\n',data.head())
    data.iloc[:,0:-1] = preprocessing.scale(data.iloc[:,0:-1])
    data['class_c'] =  pd.factorize(data['class'])[0]
    x,y = data.iloc[:,0:-2],data.iloc[:,-1]  
    train_data,test_data,train_target,test_target = train_test_split(x,y,test_size = 0.25)
    return train_data,test_data,train_target,test_target

# KNN分类算法函数定义
def kNN_Classify(test_data,train_data,train_target,k):
    #diff = train_data -  np.tile(test_data,(len(train_data),1)) 
    #test_data  =  test_data.values[0].reshape(1,4)
    # 数据标准化
    # step 1: 计算距离
    diff = train_data -  np.repeat(test_data,repeats = len(train_data) ,axis = 0)
    squaredDist = np.sum(diff ** 2, axis = 1) 
    distance = squaredDist ** 0.5

    # step 2: 对距离排序
    sortedDistIndices = np.argsort(distance).values
    classCount = []
    for i in range(k):
        # step 3: 选择k个最近邻
        target_sort = train_target.values[sortedDistIndices[i]]
        classCount.append(target_sort)
    # step 4: 计算k个最近邻中各类别出现的次数
    counter = Counter(classCount)
    # step 5: 返回出现次数最多的类别标签
    Max_count = counter.most_common(1)[0][0] 
    return Max_count

kNN_Classify(test_data.values[0].reshape(1,4),train_data,train_target,k = 5)

    
def datingClassTest(test_data,train_data,train_target,test_target,k = 5):
    m = test_data.shape[0]
    w = test_data.shape[1]
    errorCount = 0.0
    test_predict = []
    for i in range(m):
        classifierResult =  kNN_Classify(
            test_data    = test_data.values[i].reshape(1,w),
            train_data   = train_data,
            train_target = train_target,
            k = k
            )
        if (classifierResult != test_target.values[i]): 
            errorCount += 1.0
        test_predict.append(classifierResult)
    test_data['test_predict'] = test_predict
    test_data['class'] = test_target
    confusion_matrix = pd.crosstab(train_target,test_predict)
    print ("in datingClassTest,the total error rate is: %f" % (errorCount/float(m)))
    print ('in datingClassTest,errorCount:',errorCount)
    target_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(test_target,test_predict, target_names=target_names))
    return test_data,confusion_matrix

if __name__ == "__main__":
    #计时开始：
    t0 = time.time()
    train_data,test_data,train_target,test_target = Data_Input()
    test_reslut,confusion_matrix = datingClassTest(test_data,train_data,train_target,test_target,k = 5)
    name = "KNN" + str(int(time.time())) + ".csv"
    print ("Generating results file:", name)
    with open("D:/Python/File/" + name, "w",newline='') as csvfile:
        open_file_object = csv.writer(csvfile)
        open_file_object.writerow(['sepal_length','sepal_width','petal_length','petal_width','test_predict','class'])
        open_file_object.writerows(test_reslut.values)
    t1 = time.time()
    total = t1 - t0
    print("消耗时间：{}".format(total))

    
