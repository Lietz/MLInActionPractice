from numpy import *

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1 #特征数量,'\t制表'
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)  #.I,求逆矩阵
    #ws=linalg.solve(xTx,xMat.t*yMatT)
    return ws   #回归系数

# 绘图
# import matplotlib.pyplot as plt
# Backend TkAgg is interactive backend. Turning interactive mode on.
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# <matplotlib.collections.PathCollection object at 0x000001627A8060F0>
# xCopy=xMat.copy()
# xCopy.sort(0)
# yHat=xCopy*ws
# ax.plot(xCopy[:,1],yHat)
# [<matplotlib.lines.Line2D object at 0x000001627A948908>]
# plt.show()

#计算相关系数
# yHat=xMat*ws
# corrcoef(yHat.T,yMat)
# array([[ 1.        ,  0.98647356],
#        [ 0.98647356,  1.        ]])


#locally weighted linear regression,LWLR
#给“待预测点”附近每个点赋予权重，在此子集上回归
#每次预测均需要先选出对应数据子集
#给定x空间任意一点，计算yHat
def lwlr(testPoint,xArr,yArr,k=1.0):  # k越大，训练用的点越多
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))  #创建对角权重矩阵weights
    #权重大小以指数级衰减
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T*(weights*yMat))
    return testPoint*ws

#对每个点调用lwlr()
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat


# 对（0,0）进行估计
# yArr[0]
# 3.176513
# regression.lwlr(xArr[0],xArr,yArr,1.0)
# matrix([[ 3.12204471]])
# regression.lwlr(xArr[0],xArr,yArr,0.001)
# matrix([[ 3.20175729]])

# 对所有点进行估计
# yHat=regression.lwlrTest(xArr,xArr,yArr,0.003)

# 绘图：
# 对xArr排序
# xMat=mat(xArr)
# srtInd=xMat[:,1].argsort(0)
# xSort=xMat[srtInd][:,0,:]
# 绘图：
# import matplotlib.pyplot as plt
# Backend TkAgg is interactive backend. Turning interactive mode on.
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# [<matplotlib.lines.Line2D object at 0x0000020E1D5E50F0>]
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# <matplotlib.collections.PathCollection object at 0x0000020E16097710>
# plt.show()

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()  #  **2，乘方

# 预测鲍鱼年龄~~~~
# abX,abY=regression.loadDataSet('abalone.txt')
# yHat01=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
# yHat1=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
# yHat10=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
# 计算误差，不同大小核进行对比
# regression.rssError(abY[0:99],yHat01.T)
# 56.784209118372083
# regression.rssError(abY[0:99],yHat1.T)
# 429.89056187030394
# regression.rssError(abY[0:99],yHat10.T)
# 549.11817088260648
# 在新数据上的表现
# yHat01=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
# regression.rssError(abY[100:199],yHat01.T)
# 25119.459111157415
# yHat1=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
# regression.rssError(abY[100:199],yHat1.T)
# yHat10=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
# regression.rssError(abY[100:199],yHat10.T)

# 和简单线性回归比较
# ws=regression.standRegres(abX[0:99],abY[0:99])
# yHat=mat(abX[100:199])*ws
# regression.rssError(abY[100:199],yHat.T.A)  #.A  从matrix得到nparray
# 518.63631532464512


#如果特征点比样本点多-->引入岭回归(ridge regression)
#可以解决奇异矩阵，并缩减系数（shrinkage）

def ridgeRegres(xMat,yMat,lam=0.2):  #给定lambda计算岭回归系数
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:  # 防止lambda为0时报错
        print("This matrix is singular,cannot do inverse")
        return
    ws=denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr): #在一组lambda上测试结果
    xMat=mat(xArr)
    yMat=mat(yArr).T
    #标准化
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar

    numTestPts=30  #测试30个lambda
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

# 画图
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(ridgeWeights)
# [<matplotlib.lines.Line2D object at 0x0000020E172A3D68>, <matplotlib.lines.Line2D object at 0x0000020E172AB438>, <matplotlib.lines.Line2D object at 0x0000020E172AB668>, <matplotlib.lines.Line2D object at 0x0000020E172AB898>, <matplotlib.lines.Line2D object at 0x0000020E172ABAC8>, <matplotlib.lines.Line2D object at 0x0000020E172ABD30>, <matplotlib.lines.Line2D object at 0x0000020E172ABF60>, <matplotlib.lines.Line2D object at 0x0000020E17540550>]
# plt.show()

#岭回归，第二范式？
#lasso，第一范式？

#前向逐步回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):# eps每次调整的步长
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat) #？？？？
    # xMeans = mean(xMat, 0)
    # xVar = var(xMat, 0)
    # xMat = (xMat - xMeans) / xVar
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()

    for i in range(numIt):
        print(ws.T)
        lowestError=inf; # 初始误差正无穷
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def regularize(xMat):
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    return xMat

from time import sleep
import json
import urllib.request #python3
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr='get from code.google.com'
    searchURL='https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json'%(myAPIstr,setNum)
    pg=urllib.request.urlopen(searchURL)  #python3
    retDict=json.loads(pg.read())  #得到字典
    for i in range(len(retDict['items'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condition']=='new':
                newFlay=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice=item['price']
                # 如果价格比原始价格低一半，则认为不完整
                if sellingPrice>origPrc*0.5:
                    print("%d\t%d\t%d\t%f\t%f"%(yr,numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d'%i)

def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX, retY, 10030,2002,3096,369.99)
    searchForSet(retX, retY, 10179,2007,5195,499.99)
    searchForSet(retX, retY, 10181,2007,3428,199.99)
    searchForSet(retX, retY, 10189,2008,5922,299.99)
    searchForSet(retX, retY, 10196,2009,3263,249.99)





