import rm_k_means
import rm_x_means
import datetime
from osgeo import gdal, gdal_array

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

###################################
X = rm_k_means.k_get_raw_data(city,dataset,r,x,y,h,w)

rm_k_means.k_cal_show_image(X,city,dataset,r,cluster_num,x,y,h,w,now)

#rm_k_means.k_get_elbow(X,cluster_num)
#rm_k_means.k_get_hist(X,cluster_num,r)

rm_x_means.x_cal_plot(X,city,r,cluster_num,h,w,now)