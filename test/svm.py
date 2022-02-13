import numpy as np
from matplotlib import pyplot as plt
import copy
import pickle

class SVM:
    def __init__(self,C=10,toler=0.0001,deta=0.001,kernal=None,maxIter=1000):
        self.labelArr = None
        self.alphas = None
        self.supportMat = None
        self.b = 0
        self.setParam(C,toler,deta,kernal,maxIter)
    
    def setParam(self,C,toler,deta,kernal,maxIter): #deta: KKT条件中的toler
        self.C = C
        self.kernal = kernal
        self.maxIter = maxIter
        self.toler = toler
        self.deta = deta
    
    def __readFile(self,filename): #返回dataMat labelArr
        with open(filename,'r') as fr:
            data = fr.readlines()
        if not data:
            raise Exception('failed to load data')
        m = len(data)
        n = len(data[0].strip().split('\t'))-1
        dataSet = np.zeros((m,n))
        labels = np.zeros(m)
        for i in range(m):
            line = data[i].strip().split('\t')
            dataSet[i] = line[0:n]
            labels[i] = line[-1]
        return np.mat(dataSet),labels
    
    def __kernal(self,vecMat1,vecMat2):
        if self.kernal==None:
            return vecMat1*vecMat2.T
        elif self.kernal[0]=='rbf': #kernal为二元组('rbf',theta)
            numerator = np.power((vecMat1-vecMat2),2).sum()
            denominator = (-2*self.kernal[1]*self.kernal[1])
            return np.exp(numerator/denominator)
        else: raise Exception('unknown kernal')
            

    def __setK(self,dataMat):
        m,n = dataMat.shape
        kMat = np.mat(np.zeros((m,m)))
        for i in range(m):
            for j in range(m):
                kMat[i,j] = self.__kernal(dataMat[i],dataMat[j])
        return kMat
    
    def __evaluateY(self,i,alphas,labelArr,kMat):
        return (alphas*labelArr*kMat[:,i]+self.b)[0][0]
    
    def __KKTViolate(self,i,errorCache,labelArr,alphas):
        if (alphas[i] < self.C and labelArr[i]*errorCache[i] < -self.deta) or (alphas[i] > 0 and labelArr[i]*errorCache[i] > self.deta):
            return True
        else:
            return False
    
    def __randStart(self,lf,rt):
        return int(np.random.rand()*(rt-lf)+lf)
    
    def __clip(self,value,lf,rt):
        if value > rt: value = rt
        elif value < lf: value = lf
        return value
    
    def __calcNewError(self,errorCache,alphas,labelArr,K):
        for i in range(alphas.shape[0]):
            errorCache[i] = self.__evaluateY(i,alphas,labelArr,K)-labelArr[i]
    
    def __update(self,i,j,errorCache,labelArr,alphas,K): #先更新alphaj
        if i==j: return False
        div = K[i,i]+K[j,j]-2.0*K[i,j]
        if div <= 0:
            print('bad div')
            return False
        newAlphaj = alphas[j]+labelArr[j]*(errorCache[i]-errorCache[j])/div
        L = 0;H = self.C
        if labelArr[i]!=labelArr[j]:
            L = max(0,alphas[j]-alphas[i])
            H = min(self.C,self.C+alphas[j]-alphas[i])
        else:
            L = max(0,alphas[j]+alphas[i]-self.C)
            H = min(self.C,alphas[j]+alphas[i])
        if L >= H:
            #print('L>=H L: %f    H: %f'%(L,H))
            return False
        newAlphaj = self.__clip(newAlphaj,L,H)
        #print(newAlphaj)
        if abs(newAlphaj-alphas[j]) < self.toler: 
            #print('already stable')
            return False
        newAlphai = alphas[i]+labelArr[i]*labelArr[j]*(alphas[j]-newAlphaj)
        newAlphai = self.__clip(newAlphai,0,self.C)
        if newAlphai < self.toler: newAlphai = 0
        b1 = -errorCache[i]-labelArr[i]*(newAlphai-alphas[i])*K[i,i]-labelArr[j]*(newAlphaj-alphas[j])*K[i,j]+self.b
        b2 = -errorCache[j]-labelArr[i]*(newAlphai-alphas[i])*K[i,j]-labelArr[j]*(newAlphaj-alphas[j])*K[j,j]+self.b
        if newAlphai > 0 and newAlphai < self.C: self.b = b1
        elif newAlphaj > 0 and newAlphaj < self.C: self.b = b2
        else: self.b = (b1+b2)/2.0
        alphas[i] = newAlphai
        alphas[j] = newAlphaj
        self.__calcNewError(errorCache,alphas,labelArr,K) #只要有一个alpha变化，所有的Ei都改变,但太慢了
        return True
    
    def __updateAlphas(self,i,errorCache,dataMat,labelArr,alphas,K): #由alpha1：i 选出alpha2
        m,n = dataMat.shape
        indexes = np.nonzero((alphas>0) == (alphas < self.C))[0]
        indexNum = len(indexes) #i不一定在indexes中
        maxError = 0;maxj = -1
        for index in indexes: #选择|E1-E2|最大的j
            if index == i: continue
            error = abs(errorCache[i]-errorCache[index])
            if error > maxError:
                maxError = error
                maxj = index
        if maxj != -1:
            if self.__update(i,maxj,errorCache,labelArr,alphas,K): return True
            j = self.__randStart(0,indexNum) #随机选择起点,#前面更新失败，从满足 0<alpha<C 的j 中选择alphaj
            maxIter = indexNum;x = 0
            while x<maxIter:
                x += 1
                if self.__update(i,indexes[j],errorCache,labelArr,alphas,K): return True
                j = (j+1)%indexNum
        j = self.__randStart(0,m) #极端情况，前面更新都失败，从所有alpha中选取alphaj  
        validJ = [1]*m
        for index in indexes:
            validJ[index] = 0
        maxIter = m-indexNum;x = 0
        while x < maxIter:
            x += 1
            while validJ[j]==0: j = (j+1)%m
            if self.__update(i,j,errorCache,labelArr,alphas,K): return True
            j = (j+1)%m
        return False

    def train(self,dataMat,labelArr):
        m,n = dataMat.shape;
        alphas = np.zeros(m)
        errorCache = np.zeros(m)
        kMat = self.__setK(dataMat)
        errorCache = -labelArr #初始时yi的预测值为0，error = (0+b) - labelArr[i],b初值为0
        changed = True;fullSearch = True;x = 0
        while changed and x < self.maxIter:
            x += 1
            if fullSearch:
                fullSearch = False
                changed = False
                for i in range(m):
                    if self.__KKTViolate(i,errorCache,labelArr,alphas):
                        if self.__updateAlphas(i,errorCache,dataMat,labelArr,alphas,kMat):
                            changed = True
            else:
                fullSearch = True
                indexes = np.nonzero((alphas>0) == (alphas < self.C))[0]
                for i in indexes:
                    if self.__KKTViolate(i,errorCache,labelArr,alphas):
                        if self.__updateAlphas(i,errorCache,dataMat,labelArr,alphas,kMat):
                            fullSearch = False
        print('iteration times %d'%(x))
        print(alphas)
        supportIndexes = np.nonzero(alphas>0)[0]
        self.alphas = alphas[supportIndexes]
        self.supportMat = dataMat[supportIndexes,:]
        self.labelArr = labelArr[supportIndexes]
        print('number of support vectors: %d'%self.alphas.shape[0])
        return self.alphas,self.supportMat
    
    def classify(self,dataSet): #dataSet: list[array1,array2,...]
        m,n = dataSet.shape
        yHat = np.zeros(m)
        for i in range(m):
            yHat[i] = self.b
            for j in range(self.alphas.shape[0]):
                yHat[i] += (self.__kernal(dataSet[i],self.supportMat[j])*self.alphas[j]*self.labelArr[j])
        yRes = np.sign(yHat)
        #yRes[np.nonzero(yRes==-1)[0]]=9 #数字9表示为-1，现在还原
        return yRes
        
    def trainFromFile(self,filename):
        dataMat,labelArr = self.__readFile(filename)
        self.train(dataMat,labelArr)
        self.plotRes(dataMat.A,labelArr)
    
    def test(self,dataMat,labelArr):
        m,n = dataMat.shape
        error = 0
        yRes = self.classify(dataMat.A)
        for i in range(m):
            if yRes[i] != labelArr[i]:
                error += 1
            #print("predicted:%f    result:%f"%(yRes[i],labelArr[i]))
        correct = 1.0-error/float(m)
        print("error times: %d,    total times: %d\ncorrect rate: %f%%"%(error,m,correct*100))
        
    def testFromFile(self,filename):
        dataMat,labelArr = self.__readFile(filename)
        self.test(dataMat,labelArr)
    
    def saveTrainedData(self,filename='data/trained_data.txt'):
        data = {}
        data['alphas'] = copy.deepcopy(self.alphas)
        data['supportMat'] = copy.deepcopy(self.supportMat)
        data['labelArr'] = self.labelArr
        data['b'] = self.b
        data['kernal'] = self.kernal
        with open(filename,'wb') as fw:
            pickle.dump(data,fw)
        
    def loadTrainedData(self,filename='data/trained_data.txt'):
        with open(filename,'rb') as fr:
            data = pickle.load(fr)
        self.alphas = data['alphas']
        self.supportMat = data['supportMat']
        self.labelArr = data['labelArr']
        self.b = data['b']
        self.kernal = data['kernal']
    
    def plotRes(self,dataSet,labelArr):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        indexes1 = np.nonzero(labelArr==1);indexes2 = np.nonzero(labelArr==-1)
        ax.scatter(dataSet[indexes1,0].tolist(),dataSet[indexes1,1].tolist(),color='blue',s=60)
        ax.scatter(dataSet[indexes2,0].tolist(),dataSet[indexes2,1].tolist(),color='green',s=60)
        ax.scatter(self.supportMat[:,0].T.A[0].tolist(),self.supportMat[:,1].T.A[0].tolist(),color='red',s=10)
        if self.kernal == None: #仅对不使用核函数有效,画出分界直线
            ws = ((self.alphas*self.labelArr)*self.supportMat).A[0]
            print(ws)
            xmin = dataSet[:,0].min()
            xmax = dataSet[:,0].max()
            X = np.arange(xmin,xmax,0.01)
            Y = (X*ws[0]+self.b)/(-ws[1])
            plt.plot(X.tolist(),Y.tolist()) 