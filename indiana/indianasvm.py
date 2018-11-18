# xunlian模型并存储
import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
RATIO = 0.9


#split the data

data = pd.read_csv('H:\data\Y.csv',header=None)
data = data.values
data_D = data[:,:-1] # 训练样本
data_L = data[:,-1]  # 标签
data_train, data_test, label_train,label_test = train_test_split(data_D,data_L,test_size=RATIO)

clf =SVC(kernel='linear') # kernel='rbf',gamma=0.125,C=4)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
kappa = metrics.cohen_kappa_score(label_test,pred)
classify_report =  metrics.classification_report(label_test,pred)
print(accuracy)
print(kappa)
print(classify_report)
joblib.dump(clf,"indianasvm.m")
