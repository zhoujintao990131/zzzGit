from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from readData import *
from sklearn.externals import joblib 
from sklearn.preprocessing import StandardScaler
model_path='model/70.46.pkl'#保存的最高分模型
Zsvm=joblib.load(model_path)
data=pd.read_csv('now_feature/feature_test.csv')#保存的最新的特征，其余特征都可以在old_feature里找到
x_test=data.values
def std(X):
    standardScaler = StandardScaler()
    standardScaler.fit(X)
    X_standard = standardScaler.transform(X)
    return(X_standard)
x_test=std(x_test)
print('model loaded')
y_out=[]
ans=Zsvm.predict_proba(x_test)#输出的结果
for i in range(len(test_name)):
    y_out.append([test_name[i],ans[i][1]])
df = pd.DataFrame(y_out, columns=['Id','Predicted'])
df.to_csv('submission.csv',index=0)
