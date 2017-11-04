from numpy import *
from math import inf


def loadSimpData():
    datMat=matrix([[1.,2.1],
                  [2.,1.1],
                  [1.3,1.],
                  [1.,1.],
                  [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels


#根据阈值比较对数据分类---设置阈值和不等号方向，把预测值分两类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): #（输入矩阵，第几列，阈值，lt或者gt:lessthan,graterthan----*
    retArray=ones((shape(dataMatrix)[0],1))  #5x1向量，全1
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D): #测试stumpclassify找到最佳单层决策树，D是权重向量
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T  #加T得到转置
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClasEst=mat(zeros((m,1)))
    minError=inf  #无穷大
    for i in range(n): #n，特征数，遍历所有特征
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps #计算步长
        for j in range (-1,int(numSteps)+1): 
            for inequal in ['lt','gt']: #切换大于小于
                threshVal=(rangeMin+float(j)*stepSize)#阈值步增
                predictedVals=stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr=mat(ones((m,1)))  #错误向量
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr #错误*权重得error值
               # print ("split:dim %d,thresh %.2f,thresh ineqal: %s, the weighted error is %.3f"%(i,threshVal,inequal,weightedError))
                
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst     

def adaBoostTrainDS(dataArr,classLabels,numIt=40):           #dicision stump
    weakClassArr=[]   #一个单层决策树的数组
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m) #初始化为相等值
    aggClassEst=mat(zeros((m,1)))#每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print ("D",D.T)  ###
        alpha=float(0.5*log((1.0-error)/max(error,1e-16))) ###此分类器的alpha，1e-16防止除零溢出
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)  #将此弱分类器加入
        print ("classEst:",classEst.T)
        expon=multiply(-1*alpha*mat(classLabels).T,classEst) #计算新D权重向量
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst  #0+-alpha
        print ("aggClassEst:",aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print ("total error:",errorRate,"\n")
        if errorRate==0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat (datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):  #遍历所有弱分类器
        classEst=stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t')) #对一行按制表符切割，得到一个数组，每个元素代表一列
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndecies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndecies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is:",ySum*xStep)
