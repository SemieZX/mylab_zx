if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.svm import SVC
    from sklearn import metrics
    from sklearn import preprocessing
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    RATIO = 0.9


    #split the data

    data = pd.read_csv('H:\data\Y.csv',header=None)
    data = data.as_matrix()
    data_D = data[:,:-1] # 训练样本
    data_L = data[:,-1]  # 标签
    data_train, data_test, label_train,label_test = train_test_split(data_D,data_L,test_size=RATIO)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4, 8, 16], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}

    svr = SVC()
    clf =GridSearchCV(svr, parameters, n_jobs=-1)
    clf.fit(data_train,label_train)
    pred = clf.predict(data_test)
    accuracy = metrics.accuracy_score(label_test, pred)*100
    print(accuracy)
    joblib.dump(clf,"indianasvm_grid.m")
    print(clf.best_params_)
