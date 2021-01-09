import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import rasterio
import cv2
from random import random
import datetime

now = datetime.datetime.now()
print (now.strftime("%Y%m%d-%H%M"))
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Define the data/city to analyze
city = 'kyoto'
dataset = 'LC08_L1TP_110036_20201027_20201106_01_T1_B'

# Get image date to band x
r = 2

# Set number of cluster
cluster_num = 10

# Set image file name
imagefile = city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")

#############################
# img[top : bottom, left : right]
# サンプル1の切り出し、保存
x=5570
y=2083
h=90
w=50
mimage = np.zeros((h, w, r), dtype=int)

for i in range(r):
    b = i + 1
    print('====== Extract band' + str(b))

    with rasterio.open(city + '\\' + dataset + str(b) + '.tiff') as src:
        data = src.read()# numpy形式で読み込み

    # データのサイズの確認
    print('Shape: ' + str(data.shape))
    print('Dimension: ' + str(data.ndim))    

    img = data[0][y:y+h, x:x+w]

    print('Extracted Image Shape: ' + str(img.shape))
    mimage[:, :, i] = img
    print('Merged Flatted shape:'+ str(mimage.shape))    
    print('Merged Flatted data:'+ str(mimage[:, :, i]))    


new_shape = (mimage.shape[0] * mimage.shape[1], mimage.shape[2])
print('New shape: ' + str(new_shape))

X = mimage[:, :, :r].reshape(new_shape)

print('X shape:' + str(X.shape))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print(X2D.shape)
print(X2D.shape[0])

km = cluster.KMeans(10)
data2_clst = km.fit(X2D)
for i in range(X2D.shape[0]):
    plt.scatter(X2D[i,0], X2D[i,1], cmap="hsv")
plt.show()

'''
# Clustering
k_means = cluster.KMeans(n_clusters=cluster_num)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(img.shape)



plt.figure(dpi=600)
plt.axis('off')

plt.imshow(X_cluster, cmap="hsv")
plt.savefig(imagefile + '.png')
'''