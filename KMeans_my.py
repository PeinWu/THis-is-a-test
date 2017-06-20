# -*-coding:utf-8-*-
"""
Created on 2017/06/19
@author WuYe
"""
import pandas as pd
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fitLine = map(float, curLine)
        dataMat.append(fitLine)
    return dataMat

def disEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # 计算两个向量的欧式距离

def randCent(dataSet, k): #给出随机的的质心点
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(np.max(dataSet[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ*np.random.rand(k,1))
    return centroids

def KMeans(dataSet, k, distMeas=disEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data point
                                            # to a centroid,also hold SE of each point
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in  range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,:] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust,axis=0)
    return centroids, clusterAssment























