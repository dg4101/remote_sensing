import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
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
r = 7

# Set number of cluster
cluster_num = 10

# Set image file name
imagefile = city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")

#############################
# img[top : bottom, left : right]
# サンプル1の切り出し、保存
x=5570
y=2083
h=100
w=100
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

# Clustering
#k_means = cluster.KMeans(n_clusters=cluster_num)

k_means = cluster.AgglomerativeClustering(n_clusters=cluster_num, affinity='euclidean', linkage='ward')
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(img.shape)


'''
#set randum color
colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
#print(X_cluster)
#plt.figure(figsize=(20,20))
plt.imshow(X_cluster, cmap=new_map)
plt.savefig('kyoto-b1-7.png')
#plt.show()
'''
plt.figure(dpi=600)
plt.axis('off')
plt.imshow(X_cluster, cmap="hsv")
plt.savefig(imagefile + '.png')
plt.show()
