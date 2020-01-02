import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import scipy.io as sio
mat_path = os.path.join('mldata', 'mnist-original.mat')
mnist = sio.loadmat(mat_path)#手动下载的mnist的数据集，保存为mat形式
images = mnist["data"].T
targets = mnist["label"].T
X = images/255
Y = targets
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
print(type(X_train))
Y_train=Y_train.ravel()
Y_test=Y_test.ravel()
from sklearn.preprocessing import StandardScaler

# 逻辑回归法
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Amodel = LogisticRegression(max_iter=1000,fit_intercept=True,dual=False,C=0.1,random_state=50,penalty='l2')
# Amodel.fit(X_train, Y_train)
# train_accuracy = Amodel.score(X_train, Y_train)
# Y_pred=(Amodel.predict(X_test))
# test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
# print('Training accuracy: %0.2f%%' % (train_accuracy*100))
# print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

#贝叶斯
# from sklearn.naive_bayes import BernoulliNB
# Bmodel=BernoulliNB(alpha=0.1,binarize=0.5)
# Bmodel.fit(X_train,Y_train)
# train_accuracy = Bmodel.score(X_train,Y_train)
# Y_pred=(Bmodel.predict(X_test))
# test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
# print('Training accuracy: %0.2f%%' % (train_accuracy*100))
# print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

#SVC
# def std(X):
#     standardScaler = StandardScaler()
#     standardScaler.fit(X)
#     X_standard = standardScaler.transform(X)
#     return(X_standard)
# print(X_train.shape)
# X_train=std(X_test)
# print(X_train.shape)
# X_test=std(X_test)
print(type(X_train))
print(type(Y_train))
print(X_train.shape)
print(Y_train)
from sklearn.svm import LinearSVC
Cmodel=LinearSVC()
Cmodel.fit(X_train,Y_train)
train_accuracy=Cmodel.score(X_train,Y_train)
Y_pred=Cmodel.predict(X_test)
test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

#改SVC参数
# Dmodel=LinearSVC(penalty='l2',dual=True,C=0.02,loss='hinge')
# Dmodel.fit(X_train,Y_train)
# train_accuracy=Dmodel.score(X_train,Y_train)
# Y_pred=Dmodel.predict(X_test)
# test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
# print('Training accuracy: %0.2f%%' % (train_accuracy*100))
# print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
