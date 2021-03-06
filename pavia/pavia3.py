from scipy.io import loadmat
import spectral
import joblib
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np

input_image = loadmat('H:\data\Pavia.mat')['pavia']
output_image = loadmat('H:\data\Pavia_gt.mat')['pavia_gt']

testdata = np.genfromtxt('H:\data\pavia.csv',delimiter=',')
data_test = testdata[:,: -1]
label_test = testdata[:,-1]


clf = joblib.load('pavia.m')
predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,predict_label)*100
kappa = metrics.cohen_kappa_score(label_test,predict_label)
print(accuracy)
print(kappa)

new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0 :
            new_show[i][j] = predict_label[k]
            k +=1

ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(5,5))
ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(5,5))
plt.show(ground_truth)
plt.show(ground_predict)