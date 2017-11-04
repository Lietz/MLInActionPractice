from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):  #创建一个包含在所有文档中出现的不重复的词的列表
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)  # |  并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet): #输入词汇表和某个文档
    # 快速写法1：
    # listVal = [[0] * 100];
    # 快速写法2：
    # listZero = [0]
    # listVal = listZero * 100;
    returnVec=[0]*len(vocabList) # 创建和词汇表等长的向量，元素都为0
    for word in inputSet:  #遍历文档
        if word in vocabList:
            returnVec[vocabList.index(word)]=1    #出现的单词对应置1
        else: print('the word: %s is not in my Vocabulary!'%word)
    return returnVec  #词汇表中的单词在输入文档中是否出现

def bagOfWords2VecMN(vocabList,inputSet): #词袋模型
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        #else: print('the word: %s is not in my Vocabulary!'%word)
    return returnVec
#学习：得到先验和条件概率
def trainNBO(trainMatrix,trainCategory):  #文档矩阵，类别标签向量
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    #p(ci|w)=  p(w|ci)p(ci)    （判定为敏感词文档的概率）=（敏感词文档中敏感词出现的概率/似然/条件概率）（是敏感词文档的比率：先验）
    #           -----------                               -----------------------------------
    #             p(w)                                  （给定词：证据）

    # p(1|w)=  p(w|1)p(1)
    #           -----------
    #             p(w)
    # p(0|w)=  p(w|0)p(0)
    #           -----------
    #             p(w)
    pAbusive=sum(trainCategory)/float(numTrainDocs) #p(1)=有敏感词汇的文章数/文章总数,p(0)=1-p(1)
    #p0Num=zeros(numWords);p1Num=zeros(numWords) #p0 (class=0), p1(class=1) 分子
    p0Num = ones(numWords);p1Num = ones(numWords) #考虑0对概率乘积影响
    p0Denom=2.0;p1Denom=2.0  #分母  #考虑概率乘积,0.0  ->   2.0
    #遍历所有文档
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:  #敏感
            p1Num+=trainMatrix[i] #词计数加1
            p1Denom+=sum(trainMatrix[i]) #敏感文章的总词数
        else: #不敏感
            p0Num+=trainMatrix[i] #词计数加1
            p0Denom+=sum(trainMatrix[i])  # 不敏感文章总词数
    #对每个元素除以该类别的总词数

    # p1Vect=p1Num/p1Denom  # change to log()    p(w|1)  给定类别下词汇表中单词出现概率
    # p0Vect=p0Num/p0Denom  # change to log()    p(w|0)
    # # 考虑下溢出，使用对数

    #
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):   #要分类的向量p(w)，trainNBO返回值

    # p(1|w)=  p(w|1)p(1)
    #           -----------
    #             p(w)

    #-------------------最大似然估计-------------------------------

    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)

    #返回大概率对应的标签
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString) #去掉少于两个字符的字符串，转为小写
    return [tok.lower() for tok in listOfTokens if len(tok)>2]  #字符串列表

def spamTest():
    docList=[];classList=[];fullText=[]
    #导入并解析文本文件
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i,'r').read())  #read()可读取全部内容
        docList.append(wordList)  #append 接受一个参数，这个参数可以是任何数据类型，并且简单地追加到 list 的尾部
        fullText.extend(wordList)  #extend 接受一个参数，这个参数总是一个 list，并且把这个 list 中的每个元素添加到原 list 中
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i,'r',errors='ignore').read())  #忽略不规范编码
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList); testSet=[]
    trainingSet=list(range(50) )  #[0,1,2....49]   # #python3中range（）返回range对象，改为list(range())
    # 随机构建训练集
    for i in range(10):  #随机选择十封邮件做测试集
        randIndex=int(random.uniform(0,len(trainingSet))) #uniform() 方法将随机生成下一个实数，它在[x,y]范围内
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainNBO(array(trainMat),array(trainClasses))
    errorCount=0
    #对测试集分类
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount +=1   #错误率
            print("classification error", docList[docIndex])
    print('the error rate is',float(errorCount)/len(testSet))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  # python2,3区别
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):  # 每次访问一条rss源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 去掉出现次数最高的词
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(50));
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is', float(errorCount) / len(testSet))
    return vocabList,p0V,p1V

# import bayes
# import feedparser
# ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList,pSF,pNY=nayes.localWords(ny,sf)

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    #返回大于某个阈值的所有词，按条件概率排序
    for i in range(len(p0V)):
        if p0V[i]>-6.0: topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0: topNY.append((vocabList[i],p1V[i]))
    #lambda，提供一个匿名函数
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)  #key=lambda pair: pair[1] 按元素的第二个参数排序
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


