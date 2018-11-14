import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import pandas as pd
# from xgboost import plot_importance
# from matplotlib import  pyplot
RATIO = 0.9


#split the data

data = pd.read_csv('E:\data\Y.csv',header=None)
data = data.values
data_D = data[:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train,label_test = train_test_split(data_D,data_L,test_size=RATIO)

clf = XGBClassifier(max_depth=20,learning_rate=0.05,n_estimators=300,silent=False,
                    objective='multi:softmax',
                    min_child_weight=1,
                    gamma=0.,
                    scale_pos_weight=1) #max_depth=200,learning_rate=0.1,n_estimators=2,silent=True,objective='binary:logistic'
# eval_set = [(data_test, label_test)]
# clf.fit(data_train, label_train, early_stopping_rounds=10, eval_metric="mlogloss", eval_set=eval_set, verbose=True)
clf.fit(data_train, label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
kappa = metrics.cohen_kappa_score(label_test,pred)
classify_report =  metrics.classification_report(label_test,pred)
print(accuracy)
print(kappa)
print(classify_report)
joblib.dump(clf,"indianaxgboost.m")

# plot_importance(clf)
# pyplot.show()