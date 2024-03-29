import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import matplotlib
import rasterio
from random import random
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Get image date
# Landstat8 atami
# with rasterio.open('atami\LC08_L1TP_108036_20210610_20210615_02_T1_B10.TIF') as src2:
# sentinel 1 atami
with rasterio.open('atami\T54SUD_20210601T012659_B02.jp2') as src2:
    
#with rasterio.open('kyoto\T53SNU_20210510T013651_B8A.jp2') as src2:
#with rasterio.open('kyoto\LC08_L1TP_110036_20201027_20201106_01_T1_B2.tiff') as src2:
#with rasterio.open('singapore\LC08_L1TP_125059_20180524_20180605_01_T1_B2.tiff') as src2:
    data2 = src2.read()# numpy形式で読み込み

# データのサイズの確認
print('======band2')
print(data2.shape)
print(data2.ndim)  

plt.imshow(data2[0])
plt.show()


'''
#############################
# img[top : bottom, left : right]
# サンプル1の切り出し、保存

print(data2.shape)

# Reshape to work with clustering
X = img2.reshape((-1,1))

# Clustering
k_means = cluster.KMeans(n_clusters=20)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(img2.shape)


plt.figure(dpi=300)
plt.axis('off')

plt.imshow(data2, cmap="hsv")


'''