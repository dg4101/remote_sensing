import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import plot
import cv2
from PIL import Image
from osgeo import gdal

def get_color_pic(city,dataset,r,x,y,h,w,now):

    b = r[0]
    print('image:' + str(b))

    with rasterio.open(city + '\\' + dataset + b + '.jp2',driver='JP2OpenJPEG') as src0:
        data0 = src0.read()
        img0 = data0[0][y:y+h, x:x+w]
    b = r[1]
    print('image:' + str(b))
    with rasterio.open(city + '\\' + dataset + b + '.jp2',driver='JP2OpenJPEG') as src1:
        data1 = src1.read()
        img1 = data1[0][y:y+h, x:x+w]
    b = r[2]
    print('image:' + str(b))
    with rasterio.open(city + '\\' + dataset + b + '.jp2',driver='JP2OpenJPEG') as src2:
        data2 = src2.read()
        img2 = data2[0][y:y+h, x:x+w]

    # Set image file name
    listToStr = ' '.join(map(str, r))
    imagefile = 'images/x-' + city + '_bandnum_' + listToStr + '_' + now + '_color'


    trueColor = rasterio.open(imagefile + '.tiff','w',
                    driver='Gtiff', 
                    width=w, height=h, 
                    count=3, crs=src2.crs,
                    transform=src2.transform, 
                    dtype=src2.dtypes[0]) 
    trueColor .write(img0,3) # blue
    trueColor .write(img1,2) # green
    trueColor .write(img2,1) # red
    trueColor .close()

    scale = '-scale 0 255 0 25'
    options_list = [
        '-ot Byte',
        '-of JPEG',
        scale
    ] 
    options_string = " ".join(options_list)

    gdal.Translate(imagefile +'.jpg',
                imagefile +'.tiff',
                options=options_string)

    src = rasterio.open(imagefile + '.jpg')
    plot.show(src)

    '''
    Xrgb = np.dstack((np.dstack((X[:,0].reshape([h,w]), X[:,1].reshape([h,w]))), X[:,2].reshape([h,w])))

    # RGB合成した画像の可視化
    #plt.imshow(Xrgb)
    plt.imshow(X[:,0].reshape([h,w]))
    plt.title('RGB Image')
    plt.show()
    '''