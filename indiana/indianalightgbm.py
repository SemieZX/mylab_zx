import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
import pandas as pd
# from xgboost import plot_importance
# from matplotlib import  pyplot
RATIO = 0.9


#split the data

data = pd.read_csv('H:\data\Y.csv',header=None)
data = data.as_matrix()
data_D = data[:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train,label_test = train_test_split(data_D,data_L,test_size=RATIO)

clf = lgb.LGBMClassifier()
# eval_set = [(data_test, label_test)]
# clf.fit(data_train, label_train, early_stopping_rounds=10, eval_metric="mlogloss", eval_set=eval_set, verbose=True)
clf.fit(data_train, label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
print(accuracy)
joblib.dump(clf,"indianalightgbm.m")