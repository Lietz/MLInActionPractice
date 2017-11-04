import matplotlib.pyplot as plt
#定义文本框和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8") #decisionNode  #sawtooth:波浪线，fc：注解框内颜色深度
leafNode=dict(boxstyle="round4",fc="0.8") #leafNode
arrow_args = dict(arrowstyle="<-") #arrow
#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # createPlot.ax1 绘图区
    # xy是箭头的位置（终点），xytext是起点（注解框位置），
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
    
# def createPlot():
#     fig=plt.figure(1,facecolor='white')#创建新figure
#     fig.clf()#清空绘图区
#     createPlot.ax1=plt.subplot(111,frameon=False)
#     plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)#decisionNode
#     plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)#leafNode
#     plt.show()
#结合树字典理解
def getNumLeafs(myTree):
    numLeafs=0
    #firstStr=myTree.keys()[0] #第一个关键字，也是第一次划分数据集的类别标签，数值是子节点的取值
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': #该节点也是一个判断节点
            numLeafs+=getNumLeafs(secondDict[key])  #递归
        else : numLeafs+=1  #叶子结点
    return numLeafs
# 深度即为计算判断节点的个数
def getTreeDepth(myTree):
    maxDepth=0

    # ‘dict_keys‘ object
    # does
    # not support
    # indexing，这是因为python3改变了dict.keys,
    # 返回的是dict_keys对象, 支持iterable
    # 但不支持indexable，我们可以将其明确的转化成list
    # firstStr=myTree.keys()[0]
    firstSides=list(myTree.keys())
    firstStr=firstSides[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys(): #深度遍历
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):  #retrieve:检索   返回列表,i=0,第一棵树，i=1，第二棵树
    listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                 {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]
#父子节点间填充文本
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotation=30)

def plotTree(myTree,parentPt,nodeTxt):
    #计算宽与高
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]

    # plotTree.totalW存储树宽度，D存储深度；使用这两个变量计算树节点的摆放位置
    # 将树绘制在水平和垂直方向的中心位置
    # 树的宽度用于计算放置判断节点的位置，原则是放在所有叶子结点的中间
    # xOff，yOff追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    # 绘制图形的x轴，y轴有效范围是0.0到1.0
    # 按照叶子结点的数目将x轴划分为若干部分
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)

    # 绘出子节点具有的特征值，或沿此分支向下的数据实例必须具有的特征值
    plotMidText(cntrPt,parentPt,nodeTxt) # 计算父节点和子节点的中间位置
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]

    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD #按比例减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops) # **args  关键字参数（组装成含参数名的dict） *args 可变参数（组装成tuple）
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW; plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


