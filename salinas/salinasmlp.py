import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd



#split the data

data = pd.read_csv('H:\data\salinas.csv',header=None)
data = data.as_matrix()
data_D = data[:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train,label_test = train_test_split(data_D,data_L,test_size=0.8)
# train the model
clf =MLPClassifier(solver = 'lbfgs',alpha=1e-5,hidden_layer_sizes=(4,6,16 ),random_state=1)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
print(accuracy)
joblib.dump(clf,"salinasmlp.m")
