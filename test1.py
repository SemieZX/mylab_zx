import matplotlib.pyplot as plt

img = plt.imread('indianantest.jpeg')
img_s = img[:,:,0]
sc = plt.imshow(img_s)
sc.set_cmap('Greys')
sc.set_clim(0,100)
# plt.colorbar()
plt.show()
