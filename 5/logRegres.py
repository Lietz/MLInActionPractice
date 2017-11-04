from numpy import *


def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt','r',errors='ignore')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #截距，x1,x2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))
# mat，二维
# array，多维
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)# 样本
    # numpy.ndarray.transpose()
    # For a 1-D array, this has no effect.
    # To change between column and row vectors, first cast the 1-D array into a matrix object.
    labelMat=mat(classLabels).transpose()# 标签
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)  # h hypothesis,  a column vector
        # 计算cost，调整权重
        error=(labelMat-h)              #cost=h-y
        weights=weights+alpha*dataMatrix.transpose()*error  #更新：theta=theta-alpha* (h-y)*x
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    #weights=mat(wei).getA() #numpy.matrix.getA :Return self as an ndarray object.
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]  #n行：n个样本
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i]==1):
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') #s:大小
    ax.scatter(xcord2,ycord2,s=30,c='green')

    #画决策边界线
    x=arange(-3.0,3.0,0.1)  #arange(起点，终点，步长）
      #  w0x0+w1x1+w2x2>0
    y=(-weights[0]-weights[1]*x)/weights[2] # y即为x2
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


#随机梯度：一次用一个样本更新参数
def stocGradAscent0(dataMatrix,classLabels):
    #不用矩阵转换
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights)) # h是一个数
        error=classLabels[i]-h #也是一个数
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):  #i：第i个选出的样本
            #alpha每次减少1/（j+i），当j<<max(i)，就不是严格下降的
            alpha=4/(1.0+j+i)+0.0001 #apha decreases with iteration #j<<max(i)时可避免参数严格下降
            randIndex=int(random.uniform(0,len(dataIndex)))# uniform,产生下一个(x,y)之内的随机数
            h=sigmoid(sum(dataMatrix[randIndex]*weights)) # h是一个数
            error=classLabels[randIndex]-h #也是一个数
            weights=weights+alpha*error*dataMatrix[randIndex]
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt','r')
    frTest=open('horseColicTest.txt','r')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)

    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
      #  print("sizearray"+str(size(array(lineArr))))
     #   print("sizeweights"+str(size(trainWeights)))

        for i in range(21):
            lineArr.append(float(currLine[i]))

        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is:%f"%errorRate)
    return errorRate


def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest() #调用colicTest10次求平均值
    print("after %d iterations the average error rate is: %f"%(numTests,errorSum/float(numTests)))



