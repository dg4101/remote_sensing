import rm_k_means
import rm_x_means_thermo
import rm_x_means
import datetime
from osgeo import gdal, gdal_array

now = datetime.datetime.now()
print (now.strftime("%Y%m%d-%H%M"))
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Define the data/city to analyze
city = 'kyoto'
yeardate = 'Oct/27 2020'
#yeardate = 'Apr/21 1992'
#dataset = 'LC08_L1TP_110036_20201027_20201106_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20180718_20180730_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20171222_20180102_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20201027_20201106_01_T1_B'
#dataset = 'LT05_L1TP_110036_19920421_20170123_01_T1_B'
#Sentinel2
dataset = 'T53SNU_20210510T013651_B0'

# Get image date to band x
#r = 2
r = 9

# Set number of cluster
cluster_num = 15

# Set image file name
imagefile = city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")

#############################
# img[top : bottom, left : right]
# same as theses

# LS8
#x=5677
#y=2318
# LS5
#x=6047
#y=2018
#h=500
#w=300

#Sentinel2
x=2900
y=800
h=1200
w=850

#大文字
#x=5900
#y=2250
#h=200
#w=200

'''
# new
#x=5570
#y=2083
#h=900
#w=500
'''

###################################
X = rm_k_means.k_get_raw_data(city,dataset,r,x,y,h,w)

rm_k_means.k_cal_show_image(X,city,dataset,r,cluster_num,x,y,h,w,now)

#rm_k_means.k_get_elbow(X,cluster_num)
#rm_k_means.k_get_hist(X,cluster_num,r)

#rm_x_means.x_cal_plot(X,city,r,cluster_num,h,w,now)
#rm_x_means_thermo.x_cal_thermo(X,city,r,cluster_num,h,w,now,yeardate)

print ('Start: ' + now.strftime("%Y%m%d-%H%M"))
print ('End: ' + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
