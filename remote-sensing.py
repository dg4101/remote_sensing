import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

with rasterio.open('kyoto\LC08_L1TP_110036_20201027_20201106_01_T1_B1.tiff') as src1:
    data1 = src1.read()# numpy形式で読み込み
with rasterio.open('kyoto\LC08_L1TP_110036_20201027_20201106_01_T1_B2.tiff') as src2:
    data2 = src2.read()# numpy形式で読み込み
with rasterio.open('kyoto\LC08_L1TP_110036_20201027_20201106_01_T1_B3.tiff') as src3:
    data3 = src3.read()# numpy形式で読み込み
    
# データのサイズの確認
print('======band1')
print(data1.shape)
print(data1.ndim)
print('======band2')
print(data2.shape)
print(data2.ndim)

# データの可視化
'''
fig, ax = plt.subplots()
plt.imshow(data[0])
plt.colorbar()
#plt.show(block=False)
print("trimming")
'''
#############################
# img[top : bottom, left : right]
# サンプル1の切り出し、保存
x=5570
y=2083
h=900
w=500
img1 = data1[0][y:y+h, x:x+w]
img2 = data2[0][y:y+h, x:x+w]
img3 = data3[0][y:y+h, x:x+w]

#cv2.imshow("original", data[0])
cv2.imshow("trim1", img1)
cv2.imshow("trim2", img2)
cv2.imshow("trim3", img3)

print('======band1')
print(img1[:10])
print(img1.shape)
print(img1.ndim)

print('======band2')
print(img2[:10])
print(img2.shape)
print(img2.ndim)


# Generate array
#xdata = np.stack([img1,img2,img3], axis=0)
#xdata = np.dstack([img1,img2,img3])
# Stack the individual bands to make a 3-D array.
xdata = np.concatenate((img1, img2, img3))
print(xdata.shape)

print('======band combined')
print(xdata[:10])
print(xdata.shape)
print(xdata.ndim)


model = KMeans(20) # x種類のグループに分ける
model.fit(xdata)
#print(model.labels_)
for i in range(20):
    p = xdata[model.labels_ == i, :]
    plt.scatter(p[:, 0], p[:, 1], color = 'r')

plt.show()

