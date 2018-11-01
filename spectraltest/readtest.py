from spectral import *
img = open_image('92AV3C.lan')

# print(img)

# print(img.shape)
#
# pixel = img[50,100]
# print(pixel.shape)
#
#
# band6 = img[:,:,5]
# print(band6.shape)


# print(pixel)

img.bands = aviris.read_aviris_bands('92AV3C.spc')