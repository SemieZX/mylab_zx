import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
from functools import reduce
from sklearn import preprocessing
import pandas as pd

input_image = loadmat('H:\data\Pavia.mat')['pavia']
output_image = loadmat('H:\data\Pavia_gt.mat')['pavia_gt']
# print(input_image)
# print(output_image.shape)
# print(np.unique(output_image))


dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]]=0
            dict_k[output_image[i][j]] +=1

# print (dict_k)
# print (reduce(lambda x,y:x+y,dict_k.values()))
#
# ground_truth = spectral.imshow(classes = output_image.astype(int),figsize=(5,5))
# plt.show(ground_truth )

need_label = np.zeros([output_image.shape[0],output_image.shape[1]])
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0:
            need_label[i][j] = output_image[i][j]
print(need_label)

new_datawithlabel_list = []
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if need_label[i][j] != 0:
            c2l = list(input_image[i][j])
            c2l.append(need_label[i][j])
            new_datawithlabel_list.append(c2l)
new_datawithlabel_array = np.array(new_datawithlabel_list)

print(new_datawithlabel_list)
print(new_datawithlabel_array)


data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]

new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
new_.to_csv('H:\data\pavia.CSV',header=False,index=False)