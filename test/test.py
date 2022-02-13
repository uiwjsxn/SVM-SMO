import svm
import numpy as np
import os
def img2vector(filename):
    with open(filename,'r') as fp:
        data = fp.readlines()
    if not data:
        raise Exception("failed to load file in test.py=>img2vector")
    m = len(data)
    vector = np.zeros((1,1024))
    for i in range(m):
        for j in range(32):
            vector[0,i*32+j] = int(data[i][j])
    return vector

def readfile(folderName): #依次读取文件夹中每个txt文件，最终汇总成dataSet和labels
    fileList = os.listdir(folderName) #listdir获取文件夹下文件名列表
    m = len(fileList)
    labels = []
    dataSet = np.zeros((m,1024))
    for i in range(m):
        filename = fileList[i]
        dataSet[i] = img2vector(folderName+'/'+filename)
        labels.append(int(filename.split('_')[0]))
    labelArr = np.array(labels)
    labelArr[np.nonzero(labelArr==9)[0]] = -1
    return np.mat(dataSet),labelArr

def test():
    smo = svm.SVM(C=1000,kernal=('rbf',30)) #def __init__(self,C=10,toler=0.00001,deta=0.0001,kernal=None,maxIter=1000)
    dataMat,labelArr = readfile('data/digits/trainingDigits')
    smo.train(dataMat,labelArr)
    print('train error:')
    smo.test(dataMat,labelArr)
    
    dataMat,labelArr = readfile('data/digits/testDigits')
    print('test error')
    smo.test(dataMat,labelArr)
    
test()