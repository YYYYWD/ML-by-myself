import numpy as np
import random
import pandas as pd
import numpy as np



def read_file():
    train_file="train.csv"
    data=pd.read_csv(train_file)
    data.insert(0, 'c', np.ones(1824))
    x_train=data.values[:,0:54]
    y_train=data.values[:,54:55]
    y_train=y_train.transpose()[0]

    return x_train,y_train



def loadDataSet():
    dataMat=[];labelMat=[];
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)    #100X3
    labelMat=np.mat(classLabels).transpose()  #100X1
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))      #3X1
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)#  h:100x1
        error=(labelMat-h)   # 100X1
        weights=weights+alpha*dataMatrix.transpose()*error   #weights:3X1
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix=np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=0):
    dataMatrix = np.array(dataMatrix)
    m,n=np.shape(dataMatrix)
    weighs=np.ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.001
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weighs))
            error=classLabels[randIndex]-h
            weighs=weighs+alpha*error*dataMatrix[randIndex]
            #del(dataIndex[randIndex])
    return weighs



def plotBestFit(wei):
    import matplotlib.pyplot as plt

    weights=wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataMat)[0]
    xcord1=[];ycord1=[];
    xcord2=[];ycord2=[];
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def read_test():
    train_file = "test.csv"
    data = pd.read_csv(train_file)

    data.insert(0,'c',np.ones(783))
    x_test = data.values[:, 0:54]

    return x_test

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__=='__main__':
    dataArr, labelMat = loadDataSet()
    dataArr1, labelMat1 =read_file()#loadDataSet()
    dataArr1=dataArr1.tolist()
    labelMat1=labelMat1.tolist()
    wei=gradAscent(dataArr1,labelMat1)

    test_data=read_test()
    test_data=test_data.tolist()

    t1=[]
    for i in test_data:
        t1.append(np.dot(i,wei))
    print(t1)
