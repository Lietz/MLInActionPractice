from numpy import *
import operator
from os import listdir

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group ,labels
    
    
    #(输入向量，样本集，标签向量，选择几个最近邻居）
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #行数
    #欧氏距离计算
    diffMat=tile(inX,(dataSetSize,1))-dataSet #向量化计算出X与所有dataset点之差
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)#axis=1,横求和,axis=0，竖求和
    distances=sqDistances**0.5
    
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range (k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    
    
    #版本问题 iteritems->items
    sortedClassCount=sorted(classCount.items(),
     key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   #返回一个int表示分类
    
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines) #readlines()一次读取所有内容并按行返回list
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[] #[]，list
    index=0
    for line in arrayOLines:
        line=line.strip()  #strip,去除字符串头尾指定字符，截取回车
        listFromLine=line.split('\t')#split,通过指定分隔符对字符串切片为列表，'\t'制表符，'\n'换行,''空格，
        returnMat[index,:]=listFromLine[0:3] #returnMat[0,:]=listFromLine[0:3]，每个样本前三个元素作为一个list赋给样本矩阵对应行，！！读代码往后看看！！读完循环！！
        classLabelVector.append(int(listFromLine[-1])) #最后一列赋给标签向量
        index += 1 
    return returnMat,classLabelVector
   
#归一化：newValue=(oldValue-min)/(max-min)
    
def autoNorm(dataSet):
    minVals=dataSet.min(0)   #从列中选最值，而不是行中
    maxVals=dataSet.max(0)   
    ranges=maxVals-minVals  #一行
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1)) #行方向重复m次，列方向1次（不变）
    normDataSet=normDataSet/tile(ranges,(m,1)) #值相除,!=矩阵除法
    return normDataSet,ranges,minVals
    
# reload(sys)
# sys.setdefaultencoding("utf-8")
# 在Python 3.x中不好使了 提示 name ‘reload’ is not defined
# 在3.x中已经被毙掉了被替换为
# import importlib
# importlib.reload(sys)

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio) #划分训练集和测试集容量
    errorCount=0.0
    for i in range(numTestVecs):
        #    此例结果=分类器（测试样本i，测试样本集[倒数i个实例,标签向量，最近邻居数）
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i]))  #%d,整数占位符，f浮点，s字符串
        if(classifierResult!=datingLabels[i]): errorCount+=1.0
        
    print("the total error rate is %f" %(errorCount/float(numTestVecs)))
    
    
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?")) #raw_input:用户输入文本行命令并返回此命令
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probabaly like this person:"+resultList[classifierResult-1])
    

def img2vector(filename):   #32x32图像
    returnVect=zeros((1,1024))
    fr=open(filename)
    #循环独处前32行，将头32个字符存入数组
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect  

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))  #m行1024列的训练矩阵
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) #储存类标签向量
        trainingMat[i,:]=img2vector('trainingDigits/%s' %fileNameStr)
        
        #获得了特征矩阵和标签向量，下面进行测试，training即“dataset”

    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])   #注意括号,split('_')[0])中间没标点
        vectorUnderTest=img2vector('testDigits/%s' %fileNameStr)   #一条测试向量
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)  #kNN，计算与哪类距离最近
        
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr))  #百分号。。。改用ide了
      #  print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i])) 
        if (classifierResult != classNumStr) : errorCount+=1.0
        
    print("\nthe total number of errors is: %d" %errorCount )
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))
    
    
    
    
    
    
