from sklearn import datasets
from sklearn import svm
digits=datasets.load_digits() #数字识别数据集

#digits.data  #数据
#digits.target #类标签
#digits.images[0] #第一列,"0",8x8

clf=svm.SVC(gamma=0.001,C=100.)

#训练，不使用最后一条实例；-1，倒数第一个元素的索引
clf.fit(digits.data[:-1],digits.target[:-1]) 

#预测，用最后一条
clf.predict(digits.data[-1:] #-->array([8])


#存储model
import pickle
s=pickle.dumps(clf)  #对象化序列为bytes
clf2=pickle.loads(s) #反序列化
#clf2.predict(...)

#大数据序列化
#from sklearn.externals import joblib
#joblib.dump(clf, 'filename.pkl') 
#反序列化
#clf = joblib.load('filename.pkl')

#多类匹配多标签
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X) #-->array([0, 0, 1, 1, 2])
#标签向量是1d所以结果和其对应
#也可以匹配2d标签
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X) 
#--> array([[1, 0, 0],
  #     [1, 0, 0],
   #    [0, 1, 0],
    #   [0, 0, 0],
     #  [0, 0, 0]])
     
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
 y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
 #标签二值化
 y = preprocessing.MultiLabelBinarizer().fit_transform(y)
 #  -->y=array([[1, 1, 0, 0, 0],
  #             [1, 0, 1, 0, 0],
   #            [0, 1, 0, 1, 0],
    #           [1, 0, 1, 1, 0],
     #          [0, 0, 1, 0, 1]])

     
   

