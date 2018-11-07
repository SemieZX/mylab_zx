import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
import joblib
from sklearn import metrics
from sklearn.decomposition import PCA

input_image = loadmat('H:\data\hyp_data.mat')['hyp_data']
output_image = loadmat('H:\data\X.mat')['X']

testdata = np.genfromtxt('H:\data\X.csv',delimiter=',')

data_test = testdata[:,:-1]
label_test = testdata[:,-1]

pca = PCA(n_components=100)
pca.fit(data_test)
data_test_pca = pca.transform(data_test)



clf = joblib.load('indianasvm_pca.m')
predict_label = clf.predict(data_test_pca)

accuracy = metrics.accuracy_score(label_test,predict_label)*100
kappa = metrics.cohen_kappa_score(label_test,predict_label)
each_class_accuracy = metrics.precision_score(label_test, predict_label, average=None)*100

aver_accuracy = np.mean(each_class_accuracy)
classify_report =  metrics.classification_report(label_test,predict_label)

print(classify_report )
print(accuracy)
print(kappa)
print(each_class_accuracy)
print(aver_accuracy)

new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0

for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0 :
            new_show[i][j] = predict_label[k]
            k += 1

ground_truth = spectral.imshow(classes=output_image.astype(int), figsize=(3, 3))
ground_predict = spectral.imshow(classes=new_show.astype(int), figsize=(3, 3))
plt.show(ground_truth)
plt.show(ground_predict)
