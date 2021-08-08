import rm_k_means
import rm_x_means_thermo
import rm_x_means
import datetime
from osgeo import gdal, gdal_array

now1 = datetime.datetime.now()

now = now1.strftime("%Y%m%d-%H%M") + '_20200815_'

#print (now.strftime("%Y%m%d-%H%M"))
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Define the data/city to analyze
city = 'atami'
yeardate = 'Jun/15 2021'
#yeardate = 'Apr/21 1992'
#dataset = 'LC08_L1TP_110036_20201027_20201106_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20180718_20180730_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20171222_20180102_01_T1_BR'
#dataset = 'LC08_L1TP_110036_20201027_20201106_01_T1_B'
#dataset = 'LT05_L1TP_110036_19920421_20170123_01_T1_B'
#Sentinel2
#dataset = 'LC08_L1TP_108036_20210610_20210615_02_T1_B'
#dataset = 'T54SUD_20210601T012659_B0'
dataset = 'T54SUD_20200815T012659_B'

# Get image date to band x
#r = 2
#r =[ '02', '03', '04', '08' ]
r =[  '05', '06', '07','8A', '11', '12']
# Set number of cluster
cluster_num = 12

listToStr = ' '.join(map(str, r))

# Set image file name
#imagefile = city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")
imagefile = city + '_bandnum_' + listToStr + '_clusternum_' + str(cluster_num) + '_' + now

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
# 10m
'''
x=2200
y=1000
h=1000
w=500
'''
# 20m
x=1150
y=500
h=350
w=200

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

#rm_k_means.k_cal_show_image(X,city,dataset,r,cluster_num,x,y,h,w,now)

#rm_k_means.k_get_elbow(X,cluster_num)
#rm_k_means.k_get_hist(X,cluster_num,r)

rm_x_means.x_cal_plot(X,city,r,cluster_num,h,w,now)
#rm_x_means_thermo.x_cal_thermo(X,city,r,cluster_num,h,w,now,yeardate)

print ('Start: ' + now)
print ('End: ' + datetime.datetime.now().strftime("%Y%m%d-%H%M"))


#############
dataset = 'T54SUD_20210716T012701_B'
now = now1.strftime("%Y%m%d-%H%M") + '_20210716_'

# Set image file name
imagefile = city + '_bandnum_' + listToStr + '_clusternum_' + str(cluster_num) + '_' + now
X = rm_k_means.k_get_raw_data(city,dataset,r,x,y,h,w)

#rm_k_means.k_cal_show_image(X,city,dataset,r,cluster_num,x,y,h,w,now)

#rm_k_means.k_get_elbow(X,cluster_num)
#rm_k_means.k_get_hist(X,cluster_num,r)

rm_x_means.x_cal_plot(X,city,r,cluster_num,h,w,now)
#rm_x_means_thermo.x_cal_thermo(X,city,r,cluster_num,h,w,now,yeardate)

print ('Start: ' + now)
print ('End: ' + datetime.datetime.now().strftime("%Y%m%d-%H%M"))