import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from readData import *
from sklearn.externals import joblib 
import datetime

cur_time=datetime.datetime.now()
y_train=np.array(train_tag)
y_train=y_train.reshape(-1,1)
y_train=y_train.ravel()

data=pd.read_csv('now_feature/feature_train.csv')
x_train=data.values

data=pd.read_csv('now_feature/feature_test.csv')#保存的最新的特征，其余特征都可以在old_feature里找到
x_test=data.values

def std(X):
    standardScaler = StandardScaler()
    standardScaler.fit(X)
    X_standard = standardScaler.transform(X)
    return(X_standard)
x_train=std(x_train)
x_test=std(x_test)

maxx=0
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Cmodel=LinearSVC(penalty='l2',loss='squared_hinge',dual=True,tol=1e-6,C=40.0,multi_class='ovr',fit_intercept=True,max_iter=1000)
# Cmodel=SVC(probability=True,tol=1e-5,kernel='rbf',C=25.5,gamma=0.05)
Cmodel=SVC(probability=True,tol=1e-5,kernel='rbf',C=0.63,gamma=2.17)
# Cmodel=SVC(probability=True,tol=1e-6,kernel='rbf',C=0.54,gamma=2.8,shrinking=True)
Cmodel.fit(x_train,y_train)
train_accuracy=Cmodel.score(x_train,y_train)

# test_accuracy=metrics.accuracy_score(Y_test,Y_pred)
# print('Training accuracy: %0.2f%%' % (train_accuracy*100))
# print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
y_pred=Cmodel.predict(x_test)
yy=Cmodel.predict_proba(x_test)
y_out=[]
for i in range(len(test_name)):
    y_out.append([test_name[i],yy[i][1]])
df = pd.DataFrame(y_out, columns=['Id','Predicted'])
print(train_accuracy)
df.to_csv('out_prob.csv',index=0)
joblib.dump(Cmodel,'model/'+str(cur_time)+'.pkl')
print('model saved')
# Amodel = LogisticRegression(max_iter=1000,fit_intercept=True,dual=False,C=0.04,random_state=None,penalty='l1')
# Amodel.fit(x_train, y_train)
# train_accuracy = Amodel.score(x_train, y_train)
# y_pred=(Amodel.predict(x_test))
# print('Training accuracy: %0.2f%%' % (train_accuracy*100))

model_path='model/70.46.pkl'#保存的最高分模型
Zsvm=joblib.load(model_path)
print('model loaded')
ans=Zsvm.predict_proba(x_test)#输出的结果
