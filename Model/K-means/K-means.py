#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt 

def DataInput():
    data = pd.read_csv("D:/Python/File/iris.csv")
    data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
    print(data.shape,'\n',data.head())
    data.iloc[:,0:-1] = preprocessing.scale(data.iloc[:,0:-1])
    data['class_c'] =  pd.factorize(data['class'])[0]
    return  data.iloc[:,0:-2],data.iloc[:,-1]  


def DistEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) 

def RandCentre(dataSet, k):     
    n = dataSet.shape[1]
    Centres = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = np.min(dataSet.iloc[:,j])
        rangeJ = np.float(np.max(dataSet.iloc[:,j]) - minJ)
        Centres[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1)) 
    return Centres


def kMeans(dataSet,k): 
    m = dataSet.shape[0]
    ClusterAssment = np.mat(np.zeros((m,2)))
    Centres = RandCentre(dataSet,k)
    ClusterChanged = True
    while ClusterChanged:
        ClusterChanged = False
        for i in range(m): 
            minDist = np.inf;minIndex = -1
            for j in range(k):
                distJI = DistEclud(Centres[j,:],dataSet.iloc[i,:].values)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if ClusterAssment[i,0] != minIndex:
                ClusterChanged = True
                ClusterAssment[i,:] = minIndex,minDist**2
        print(Centres)
        for cent in range(k):
            index = np.nonzero(ClusterAssment[:,0].A== cent)[0].tolist()
            ptsInClust = dataSet.iloc[index,:]
            Centres[cent,:] = np.mean(ptsInClust,axis=0).values
    return Centres,ClusterAssment


def show(k): 
    dataSet,y = DataInput()
    centroids, clusterAssment = kMeans(dataSet,k)
    m,n = dataSet.shape  
    mark  = ['or', 'ob', 'og']
    for i in range(m):  
        markIndex = np.int(clusterAssment[i,0])  
        plt.plot(dataSet.iloc[i,0], dataSet.iloc[i,1],mark[markIndex])  
    for i in range(k):  
        plt.plot(centroids[i,0], centroids[i,1], mark[i], markersize = 12)  
    plt.show()
      
show(k=3)