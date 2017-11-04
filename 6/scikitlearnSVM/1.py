from sklearn import svm


# X=[[0,0],[1,1]]
# y=[0,1]
# clf=svm.SVC()  #SVC,NuSVC,LinearSVC是可执行多类分类的类
#
# #console中输入才打出来
# clf.fit(X,y)
# clf.predict([[2.,2.]])
# clf.support_vectors_
# clf.support_   #get indeces of support vectors
# clf.n_support_  # # get number of support vectors for each class


# Multi-class classification
# SVC and NuSVC implement the “one-against-one” approach
# (Knerr et al., 1990) for multi- class classification.
# If n_class is the number of classes, then n_class * (n_class - 1) / 2
# classifiers are constructed and each one trains data from two classes.
# To provide a consistent interface with other classifiers,
# the decision_function_shape option allows to aggregate the results of the
# “one-against-one” classifiers to a decision function of shape (n_samples, n_classes):

X=[[0],[1],[2],[3]]
Y=[0,1,2,3]

clf=svm.SVC(decision_function_shape='ovo')  #one vs one
clf.fit(X,Y)
dec=clf.decision_function([[1]])
dec.shape[1]  #4个类 4*3/2=6

clf.decision_function_shape="ovr" #one vs the rest
dec=clf.decision_function([[1]])
dec.shape[1] #4个类

# On the other hand, LinearSVC implements “one-vs-the-rest”
# multi-class strategy, thus training n_class models.
# If there are only two classes, only one model is trained:
lin_clf=svm.LinearSVC()
lin_clf.fit(X,Y)
dec=lin_clf.decision_function([[1]])
dec.shape[1]  #4
