from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries=len(dataSet)  #数据集中实例的总数
    labelCounts={} #大括号，字典
    for featVec in dataSet:  #feature，一条特征向量
        currentLabel=featVec[-1] #取出最后一列（分类标签）
        if currentLabel not in labelCounts.keys(): #键值不存在，则加入
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1     #记录当前类别出现的次数
            
    shannonEnt=0.0
    for key in labelCounts:  #循环，计算香农熵
        prob=float(labelCounts[key])/numEntries # 每个类别出现的概率
        shannonEnt-=prob*log(prob,2)  
    
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value): #按特征axis的value划分数据集
    retDataSet=[] #不修改原始数据集，其元素是列表，是axis=value且抽出了value的数据集
    for featVec in dataSet: #每条实例
        if featVec[axis]==value:  #特征符合value
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  #extend，加一个列表，合并元素，append会保持列表
            #reducedFeatVec不包含axis了
            retDataSet.append(reducedFeatVec)
    return retDataSet
   
def chooseBestFeatureToSplit(dataSet):     #dataset由列表组成的列表，元素长度也相等，最后一列是类别标签
    numFeatures=len(dataSet[0])-1 #判定有多少属性（减去了类别标签）
    baseEntropy=calcShannonEnt(dataSet) #原始香农熵，以做比较
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures): #遍历所有特征
        #写入所有第i个特征值到featList
        featList=[example[i] for example in dataSet]  #列表生成式，[元素 for 循环 if]
        #用set来去重
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals: #遍历所有唯一属性值
            subDataSet=splitDataSet(dataSet, i, value) #以当前特征划分数据集
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)#计算熵
        infoGain=baseEntropy-newEntropy #计算信息增益
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i           #计算最佳信息增益
    return bestFeature           #第i个特征为最佳特征

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0] #soretedClassCount是tuple组成的List
 
def createTree(dataSet,labels):            
    classList=[example[-1] for example in dataSet] #每个实例最后一列，标签
    if classList.count(classList[0])==len(classList): #count,元素在list出现的次数---list长度等于第一个元素出现次数，即为所有元素相同
        return classList[0]
    if len(dataSet[0])==1:  #只有一个特征了：便利完所有特征
        return majorityCnt(classList)  #返回出现次数最多的特征
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #用字典储存树信息
    del(labels[bestFeat])  #del:列表删除,从标签向量删除已被选用的标签
    #get list of unique values
    featValues=[example[bestFeat]for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:] #复制了类标签，存储在sublabels中（不因按参传递影响
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
    #{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}}
  
def classify(inputTree,featLabels,testVec): #再看
    #firstStr=inputTree.keys()[0]
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)  #list.index 返回索引值
    for key in secondDict.keys():
        if testVec[featIndex]==key:  #比较testvec中的值与树节点的值
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel

# 我们把变量从内存中变成可存储或传输的过程称之为序列化，
# 在Python中叫pickling，
# 在其他语言中也被称之为serialization，marshalling，flattening等等
# ，都是一个意思。
# 序列化之后，就可以把序列化后的内容写入磁盘，
# 或者通过网络传输到别的机器上。
# 反过来，把变量内容从序列化的对象重新读到内存里称之为反序列化，
# 即unpickling。

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename,):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)


# >>> fr=open('lenses.txt')
# >>> lenses=[inst.strip().split('\t') for inst in fr.readlines()]
# >>> lensesLabels=['age','prescript','astigmatic','tearRate']
# >>> lensesTree=trees.createTree(lenses,lensesLabels)
# >>> lensesTree
# {'tearRate': {'normal': {'astigmatic': {'yes': {'prescript': {'hyper': {'age': {'presbyopic': 'no lenses', 'pre': 'no lenses', 'young': 'hard'}}, 'myope': 'hard'}}, 'no': {'age': {'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}, 'pre': 'soft', 'young': 'soft'}}}}, 'reduced': 'no lenses'}}
# >>> treePlotter.createPlot(lensesTree)
# >>> lenses
# [['young', 'myope', 'no', 'reduced', 'no lenses'], ['young', 'myope', 'no', 'normal', 'soft'], ['young', 'myope', 'yes', 'reduced', 'no lenses'], ['young', 'myope', 'yes', 'normal', 'hard'], ['young', 'hyper', 'no', 'reduced', 'no lenses'], ['young', 'hyper', 'no', 'normal', 'soft'], ['young', 'hyper', 'yes', 'reduced', 'no lenses'], ['young', 'hyper', 'yes', 'normal', 'hard'], ['pre', 'myope', 'no', 'reduced', 'no lenses'], ['pre', 'myope', 'no', 'normal', 'soft'], ['pre', 'myope', 'yes', 'reduced', 'no lenses'], ['pre', 'myope', 'yes', 'normal', 'hard'], ['pre', 'hyper', 'no', 'reduced', 'no lenses'], ['pre', 'hyper', 'no', 'normal', 'soft'], ['pre', 'hyper', 'yes', 'reduced', 'no lenses'], ['pre', 'hyper', 'yes', 'normal', 'no lenses'], ['presbyopic', 'myope', 'no', 'reduced', 'no lenses'], ['presbyopic', 'myope', 'no', 'normal', 'no lenses'], ['presbyopic', 'myope', 'yes', 'reduced', 'no lenses'], ['presbyopic', 'myope', 'yes', 'normal', 'hard'], ['presbyopic', 'hyper', 'no', 'reduced', 'no lenses'], ['presbyopic', 'hyper', 'no', 'normal', 'soft'], ['presbyopic', 'hyper', 'yes', 'reduced', 'no lenses'], ['presbyopic', 'hyper', 'yes', 'normal', 'no lenses']]
# >>> inst