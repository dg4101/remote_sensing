import numpy as np
from osgeo import gdal, gdal_array
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import rasterio
import pandas as pd
import cv2
from random import random
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.cluster.hierarchy import dendrogram
import datetime
from sklearn import cluster

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
h=900
w=500
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
#X_cluster = linkage(X, metric='euclidean', method='ward')
agg  = cluster.AgglomerativeClustering(affinity='euclidean',
linkage='ward', 
n_clusters=cluster_num)

X_cluster = agg.fit_predict(X)
print('Clustered data shape: ' + str(X_cluster.shape))
print('Clustered data : ' + str(X_cluster[:10]))

X_cluster_new = X_cluster.reshape(img.shape)
print('Clustered data shape: ' + str(X_cluster_new.shape))
print('Clustered data : ' + str(X_cluster_new[:2]))

# クラスタリング結果カウント用のリスト初期化
results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# グラフ表示ラベル用のリスト
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# グルーピング結果をカウントする
for v in X_cluster:
    results[int(v)] += 1

# グラフ表示のための設定（クラスタリング結果用）
plt.subplot(121)
plt.xlabel('results')
plt.pie(x=results, labels=labels, autopct='%.2f%%')

plt.subplot(122)
plt.axis('off')
plt.imshow(X_cluster_new, cmap="hsv")
plt.savefig(imagefile + '.png')
plt.show()

'''
X_cluster = linkage(X, metric='euclidean', method='ward')
### Print
group1 = fcluster(X_cluster, 10, criterion='distance') 
group2 = fcluster(X_cluster, cluster_num, criterion='maxclust')


print('by distance:' + str(group1.shape))
print('by distance data:' + str(group1[:10]))
print('by cluster' + str(group2.shape))
print('by cluster' + str(group2[:10]))

X_cluster = group2.reshape(img.shape)

def rewrite_id(id, link, group, step, n):
    i = int(link[step,0])
    j = int(link[step,1])
    if i<n:
        group[i] = id
    else:
        rewrite_id(id, link, group, i-n, n)

    if j<n:
        group[j] = id
    else:
        rewrite_id(id, link, group, j-n, n)


n = X.shape[0]
threshold=40
group=np.empty(n,dtype='int32')
step=0
while True:
    if step>= n-2:
        break
    dist = linked[step,2]   
    if dist>threshold:
        break
    rewrite_id(step+n, linked, group, step, n)
    step=step+1

# 結果のプロット

cmap = plt.get_cmap("tab10")
cids = list(set(group))

print('cluster ids:',cids)

for i in range(X.shape[0]):
    ell = cids.index(group[i]) % 10
    plt.scatter(X[i,0], i, color=cmap(ell))
plt.grid(True)

plt.show()





# ユークリッド距離とウォード法を使用してクラスタリング
z = linkage(X, metric='euclidean', method='ward')

# 結果を可視化
fig = plt.figure(figsize=(8, 15), facecolor="w")
ax = fig.add_subplot(3, 1, 1, title="樹形図: 全体")
dendrogram(z)
ax = fig.add_subplot(3, 1, 2, title="樹形図: lastp 16")
dendrogram(z, truncate_mode="lastp", p=16)
ax = fig.add_subplot(3, 1, 3, title="樹形図: level 3")
dendrogram(z, truncate_mode="level", p=3)
plt.show()
'''
'''
# Clustering

plt.figure(dpi=600)
plt.axis('off')

plt.imshow(X_cluster, cmap="hsv")
plt.savefig(imagefile + '.png')
'''