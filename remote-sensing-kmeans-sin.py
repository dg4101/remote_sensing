import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import matplotlib
import rasterio
import cv2
from random import random
import datetime
import collections

now = datetime.datetime.now()
print (now.strftime("%Y%m%d-%H%M"))
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Define the data/city to analyze
city = 'singapore'
dataset = 'LC08_L1TP_125059_20180524_20180605_01_T1_B'

# Get image date to band x
r = 7

# Set number of cluster
cluster_num = 10

# Set image file name
imagefile = city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")

#############################
# img[top : bottom, left : right]
# サンプル1の切り出し、保存
x=2500
y=4100
h=500
w=800
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
    #print('Merged Flatted data:'+ str(mimage[:, :, i]))    


new_shape = (mimage.shape[0] * mimage.shape[1], mimage.shape[2])
print('New shape: ' + str(new_shape))

X = mimage[:, :, :r].reshape(new_shape)

print('X shape:' + str(X.shape))

# Clustering
k_means = cluster.KMeans(n_clusters=cluster_num)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster_new = X_cluster.reshape(img.shape)

print('X_cluster data:' + str(X_cluster[0]))
print('X_cluster_new shape:' + str(X_cluster_new.shape))
#print('X_cluster_new data:' + str(X_cluster_new[0]))


cluster_info = collections.Counter(k_means.labels_)

print('Size of cluster: ' + str(cluster_info))
print('Total size: ' + str(sum(cluster_info.values())))

# Get cluster info in np array
cluster_size = np.array(list(cluster_info.items()))

###################################
# IMSHOW
plt.figure(dpi=400)
plt.get_current_fig_manager().full_screen_toggle()
plt.rcParams["font.size"] = 6
#ax = plt.gca()
#ax.title.set_fontsize(8)

plt.subplot(122)
import matplotlib.patches as mpatches
print('unique color:' + str(cluster_size[:,0]))
print('unique color:' + str(cluster_size[:,1]))

im = plt.imshow(X_cluster_new,cmap='jet')
# get the colors of the values, according to the colormap used by imshow
colors = [ im.cmap(im.norm(value)) for value in cluster_size[:,0]]
# create a patch (proxy artist) for every color 
patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=cluster_size[:,0][i]) ) for i in range(len(cluster_size[:,0])) ]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

###################################
# PIE
cluster_size = np.array(list(cluster_info.items()))
print('Cluster Size: ' + str(cluster_size))
plt.subplot(121)
plt.pie(x=cluster_size[:,1], labels=cluster_size[:,0], autopct='%.2f%%',colors=colors)
plt.tight_layout()

plt.savefig(imagefile + '.png')
plt.show()




'''

##############################################

#############################
# Count items in cluster
for i in range(cluster_num):
    b = i + 1
    cluster_item = np.array([data.tolist() for label, data in zip(k_means.labels_, X) if label==i])
    
    print('cluster_item data:' + str(cluster_item[0]))
    print('cluster_item shape:' + str(cluster_item.shape))

c_cycle=("#3498db","#51a62d","#1abc9c","#9b59b6","#f1c40f",
         "#7f8c8d","#34495e","#446cb3","#d24d57","#27ae60",
         "#663399","#f7ca18","#bdc3c7","#2c3e50","#d35400",
         "#9b59b6","#ecf0f1","#ecef57","#9a9a00","#8a6b0e")

print('c_cycle data:' + str(c_cycle[0]))

new_map = matplotlib.colors.ListedColormap(c_cycle)

print('New map:' + str(new_map))


dst = X_cluster_new
for i in range(cluster_num):
    #X_cluster_new[np.where(X_cluster_new == i)] = c_cycle[i]
    dst = np.where((X_cluster_new == i) , i * 50, dst)

#dst = np.where((X_cluster_new == 1) | (X_cluster_new == 5), 255, 0)
fig, ax = plt.subplots()
ax.imshow(dst)
plt.show()


# Count items in cluster
for i in range(cluster_num):
    b = i + 1
    cluster_item = np.array([data.tolist() for label, data in zip(k_means.labels_, X) if label==i])
    
    print('cluster_item data:' + str(cluster_item[0]))
    print('cluster_item shape:' + str(cluster_item.shape))


#set randum color
colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', c_cycle, N=256)
#print(X_cluster)
#plt.figure(figsize=(20,20))
plt.imshow(X_cluster, cmap=new_map)
plt.savefig('kyoto-b1-7.png')
#plt.show()
'''