import svm

def test1():
    smo = svm.SVM()
    smo.trainFromFile('data/testSet.txt')
    
def test2():
    smo = svm.SVM(200,0.00001,0.0001,kernal=('rbf',0.5)) #def __init__(self,C=10,toler=0.00001,deta=0.0001,kernal=None,maxIter=1000)
    smo.trainFromFile('data/testSetRBF.txt')
    smo.saveTrainedData()
    smo_ = svm.SVM()
    smo_.loadTrainedData()
    print('train error:')
    smo_.testFromFile('data/testSetRBF.txt')
    print('test error:')
    smo_.testFromFile('data/testSetRBF2.txt')
    
test2()