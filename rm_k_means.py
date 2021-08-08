import numpy as np
import rasterio
from sklearn import cluster
import matplotlib.pyplot as plt
import collections
from scipy import stats

def k_get_raw_data(city,dataset,r,x,y,h,w):

    mimage = np.zeros((h, w, len(r)), dtype=int)

    for i in range(len(r)):
        b = r[i]
        print(b + ' file')

        #print('====== Extract band' + str(b))

        with rasterio.open(city + '\\' + dataset + b + '.jp2') as src:
            data = src.read()# numpy

        # Check data size
        print('Shape: ' + str(data.shape))
        print('Dimension: ' + str(data.ndim))    
        print('Dataset:' + dataset + b )


        img = data[0][y:y+h, x:x+w]

        #print('Extracted Image Shape: ' + str(img.shape))
        mimage[:, :, i] = img
        #print('Merged Flatted shape:'+ str(mimage.shape))    
        #print('Merged Flatted data:'+ str(mimage[:, :, i][:2]))    # show first two rows


    new_shape = (mimage.shape[0] * mimage.shape[1], mimage.shape[2])
    #print('New shape: ' + str(new_shape))

    X = mimage[:, :, :len(r)].reshape(new_shape)

    #print('X shape:' + str(X.shape))
    return X

def k_cal_show_image(X,city,dataset,r,cluster_num,x,y,h,w,now):

    # Set image file name
    listToStr = ' '.join(map(str, r))
    imagefile = 'images/k-' + city + '_bandnum_' + listToStr+ '_clusternum_' + str(cluster_num) + '_' + now

    # Clustering
    k_means = cluster.KMeans(n_clusters=cluster_num)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster_new = X_cluster.reshape([h,w])

    print('X_cluster iteratins:' + str(k_means.n_iter_))
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
    fig = plt.figure(dpi=400)
    fig.suptitle('K-Means ' + city + ': cluster:' + str(cluster_num) + ' iteration:' + str(k_means.n_iter_))
    plt.get_current_fig_manager().full_screen_toggle()
    plt.rcParams["font.size"] = 6
    #ax = plt.gca()
    #ax.title.set_fontsize(8)

    plt.subplot(122)
    import matplotlib.patches as mpatches
    print('unique color:' + str(cluster_size[:,0]))
    print('unique color:' + str(cluster_size[:,1]))

    # Set color map
    im = plt.imshow(X_cluster_new,cmap='turbo')
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
    plt.title('% of clusters')
    plt.pie(x=cluster_size[:,1], labels=cluster_size[:,0], autopct='%.2f%%',colors=colors)
    plt.tight_layout()

    plt.savefig(imagefile + '.png')
    #plt.show()

    k_get_hist(X,cluster_num,r,X_cluster,imagefile,city,cluster_size)

def k_get_hist(X,cluster_num,r,X_cluster,imagefile,city,cluster_size):

    '''
    # Clustering
    k_means = cluster.KMeans(n_clusters=cluster_num)
    k_means.fit(X)

    X_cluster = k_means.labels_
    '''
    print('X_cluster data:' + str(X_cluster))
    print('X_cluster shape:' + str(X_cluster.shape))
    print('X shape:' + str(X.shape))

    X_cluster = X_cluster.reshape([X.shape[0],1])
    print('X_cluster shape:' + str(X_cluster.shape))

    #X = X[X_cluster,np.newaxis]
    X = np.hstack((X,X_cluster))


    print('X merged shape:' + str(X.shape))
    print('X merged data cluster info:' + str(X[:,-1]))

    #get real cct from each band  based on the last column info
    print('X extracted data cluster info:' + str(X[:,-1]))

    # Array cluster, band , std, cv
    cluster_std = np.zeros((len(r), cluster_num))
    cluster_cv = np.zeros((len(r), cluster_num))

    d = 3

    for i  in range(cluster_num):

        if i % d == 0:
            a = 0
            # Set subplot
            fig = plt.figure(figsize=(19.0, 10.0/0.96))
            fig.subplots(d, len(r))  # d = rows, r= columns: number of band
            fig.suptitle(city + ': cluster:' + str(cluster_num) + ' : ' + str(i/d+1) ,fontsize=10)
            plt.rcParams["font.size"] = 7
            plt.get_current_fig_manager().full_screen_toggle()

        # Get if last one is 1
        index = np.where(X[:, -1] == i)

        print('Cluster ' + str(i) + ': ' + str(X[index].shape))
        print(X[index][:2]) # show first two rows

        for j in range(len(r)): # 
            b = r[j]
            dataset = X[index][:,j] # cluster i, band j
            print('Cluster ' + str(i) + ' Band ' + b + ': ' + str(dataset.shape))
            print('STD: ' + str(f'{np.std(dataset):.1f}'))
            print(dataset[:5])    # first 5 rows

            plt.subplot(d,len(r),j+1+a*len(r)) # row d, r band, 
            plt.title('Cluster:' + str(i) + ' Band:' + b + ' STD:' + str(f'{np.std(dataset):.1f}'))
            plt.hist(dataset, bins=15)  # histogram

            #  Add STD to cluster_std
            cluster_std[j,i] = f'{np.std(dataset):.1f}'
            # Add CV to cluster_cv
            cluster_cv[j,i] = f'{stats.variation(dataset):.3f}'

        print('a=' + str(a))
        a += 1 # add 1 to increment

        if i % d == d-1 or i == cluster_num-1:
            plt.tight_layout()
            plt.savefig(imagefile + '-' + str(i+1) + '-hist.png')
            #plt.show(block=False)

    # Plot std /  band / cluster
    print('cluster_std shape:' + str(cluster_std.shape))
    #print('cluster_std data:' + str(cluster_std))

    ex_cluster = cluster_size[np.argmin(cluster_size[:,1]),0]
    print('Lowest num index:' + str(np.argmin(cluster_size[:,1])))
    print('Lowest num cluster:' + str(cluster_size[np.argmin(cluster_size[:,1]),0]))

    #plt.close("all") # Close all prvisou ones
    listToStr = ' '.join(map(str, r))
    fig = plt.figure(figsize=(19.0, 10.0/0.96))
    plt.xlabel("Band")
    plt.ylabel("Standard Devisation")
    plt.title('K-Means Cluster:' + str(cluster_num) + ' Band:' + listToStr + ' Standard Divisation')
    plt.xticks(list(range(0, len(r))),list(range(1, len(r)+1)))
    for c in range(cluster_num):
        if c != ex_cluster:
            plt.plot(cluster_std[:,c], label="Cluster {}".format(c), marker="o", linestyle = "--")
    plt.legend()
    plt.savefig(imagefile + '-' + str(i+1) + '-std.png')
    #plt.show(block=False)

    # Plot CV chart
    fig = plt.figure(figsize=(19.0, 10.0/0.96))
    plt.xlabel("Band")
    plt.ylabel("Standard Devisation")
    plt.title('K-Means Cluster:' + str(cluster_num) + ' Band:' + listToStr + ' Coefficient of Variation')
    plt.xticks(list(range(0, len(r))),list(range(1, len(r)+1)))
    for c in range(cluster_num):
        if c != ex_cluster:
            plt.plot(cluster_cv[:,c], label="Cluster {}".format(c), marker="o", linestyle = "--")
    plt.legend()
    plt.savefig(imagefile + '-' + str(i+1) + '-cv.png')
    #plt.show(block=False)

def k_get_elbow(X,cluster_num):
    distortions = []

    for i  in range(1,cluster_num+1):                # 1~10クラスタまで一気に計算 
        km = cluster.KMeans(n_clusters=i)
        km.fit(X)                         # クラスタリングの計算を実行
        distortions.append(km.inertia_)   # km.fitするとkm.inertia_が求まる
        #y_km = km.fit_predict(X)

    plt.plot(range(1,cluster_num+1),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


    # Get each cluster each band histogram. y num of items, x cct, x by 100 to 200
'''
    plt.plot(range(1,11),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
'''