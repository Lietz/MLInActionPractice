from numpy import *

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #fltLine=map(float,curLine) #把每行映射成浮点数
        fltLine = list((map(float, curLine))) # python3，需要转型
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):#生成叶节点/回归树中为目标变量的均值
    return mean(dataSet[:,1])

def regErr(dataSet):#计算目标变量的平方误差（和）
    return var(dataSet[:,-1])*shape(dataSet)[0]#均方差函数var（）*样本个数

#                      树的类型，建立叶子的函数，误差计算函数，其他参数元组
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree



def chooseBestSplit(dataSet,liafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0] # 容许的误差下降值
    tolN=ops[1] # 切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#统计不同剩余特征值的数目，为1则退出
        return None,regLeaf(dataSet)
    m,n=shape(dataSet) # 大小
    S=errType(dataSet) # 误差
    bestS=inf
    bestIndex=0
    bestValue=0
    # 在所有可能的特征及其取值上遍历，找最佳切分方式
    for featIndex in range(n-1):
        #for splitVal in set(dataSet[:,featIndex]):   #python3的问题
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
        if(S-bestS)<tolS:
            return None,regLeaf(dataSet)     #误差减少不大则退出
        mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
        if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN): #切分出的数据集很小则退出
            return None,regLeaf(dataSet)
        return bestIndex,bestValue

    def isTree(obj):
        return (type(obj).__name__=='dict')

    def getMean(tree):
        if isTree(tree['right']):
            tree['right']=getMean(tree['right'])
        if isTree(tree['left']):
            tree['left']=getMean(tree['left'])
        return (tree['left']+tree['right'])/2.0

    def prune(tree,testData):
        if shape(testData)[0]==0:
            return getMean(tree)
        if(isTree(tree['right']) or isTree(tree['left'])):
            lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        if isTree(tree['left']):
            tree['left']=prune(tree['left'],lSet)
        if isTree(tree['right']):
            tree['right']=prune(tree['right'],rSet)
        if not isTree(tree['left']) and not isTree(tree['right']):
            lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
            errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))
            treeMean=(tree['left']+tree['right'])/2.0
            errorMerge=sum(power(testData[:,-1]-treeMean,2))
            if errorMerge<errorNoMerge:
                print('merging')
                return treeMean
            else:
                return tree
        else:
            return tree

    def linearSolve(dataSet):
        m,n=shape(dataSet)
        X=mat(ones((m,n)))
        Y=mat(ones((m,1)))
        X[:,1:n]=dataSet[:,0:n-1]
        Y=dataSet[:,-1]
        xTx=X.T*X
        if linalg.det(xTx)==0.0:
            raise NameError('This matrix is singular,cannot do inverse,try increasing the second value of ops')
        ws=xTx.I*(X.T*Y)
        return ws,X,Y

    def modelLeaf(dataSet):
        ws,X,Y=linearSolve(dataSet)
        return ws

    def modelErr(dataSet):
        ws,X,Y=linearSolve(dataSet)
        yHat=X*ws
        return sum(power(Y-yHat,2))



