import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
from functools import reduce
from sklearn import preprocessing
import pandas as pd

#load进去字典形式，进行定位
input_image = loadmat('H:\data\hyp_data.mat')['hyp_data']
output_image = loadmat('H:\data\X.mat')['X']

#  测试出入输出形式
print(input_image.shape)
# print(output_image.shape)
# print(np.unique(output_image))

# 统计地物各类别个数
dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]]=0
            dict_k[output_image[i][j]] +=1
print (dict_k)
# print (reduce(lambda x,y:x+y,dict_k.values()))

ground_truth = spectral.imshow(classes = output_image.astype(int),figsize=(5,5))
plt.show(ground_truth )

# 去掉0这个不需要分类的类别，把需要分类的类取出来
need_label = np.zeros([output_image.shape[0],output_image.shape[1]])
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0:
            need_label[i][j] = output_image[i][j]
print(need_label.shape)# 注释掉


# 其中c21保存的是一个像元的光谱信息
new_datawithlabel_list = [] # 新的标签列表
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if need_label[i][j] != 0:
            c2l = list(input_image[i][j])# (i,j)位置保存的是该处像元的光谱信息
            c2l.append(need_label[i][j]) # 将(i,j)处像元的标签加在特征之后
            new_datawithlabel_list.append(c2l) # 把特征连和标签组合后 放入数据集
new_datawithlabel_array = np.array(new_datawithlabel_list)# 把生成的list转换为便于处理的numpy矩阵




#print(len(new_datawithlabel_list))
#print(len(new_datawithlabel_array[-1])) 验证合成向量长度为201





# 标准化存储数据

data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]

new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
new_.to_csv('H:\data\Y.CSV',header=False,index=False)
# print(new.shape)  # 测试标准化后存储的数据(10366, 201)
# print(new_)


# 换颜色显示方法
# ksc_color =np.array([[255,255,255],
#      [184,40,99],
#      [74,77,145],
#      [35,102,193],
#      [238,110,105],
#      [117,249,76],
#      [114,251,253],
#      [126,196,59],
#      [234,65,247],
#      [141,79,77],
#      [183,40,99],
#      [0,39,245],
#      [90,196,111],
#      [10,124,167],
#      [86,120,220],
#      [100,32,162],
#      [198,172,40]])
#
# ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9),colors=ksc_color)
# plt.show(ground_truth )



