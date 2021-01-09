import numpy as np
import matplotlib.pyplot as plt
import collections
import pyclustering
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.utils import draw_clusters
from scipy import stats

def x_cal_plot(X,city,r,cluster_num,h,w,now):
    # Set image file name
    imagefile = 'images/x-' + city + '_bandnum_' + str(r) + '_clusternum_' + str(cluster_num) + '_' + now.strftime("%Y%m%d-%H%M")


    # クラスタ数2から探索させてみる
    initial_centers = kmeans_plusplus_initializer(X, 2).initialize()
    # クラスタリングの実行
    instances = xmeans(X, initial_centers, kmax=cluster_num, tolerance=0.025, criterion=0, ccore=True)
    instances.process()
    # クラスタはget_clustersで取得できる
    clusters = instances.get_clusters()
    centers = instances.get_centers()

    print('X shape:' + str(X.shape))
    X_cluster_new = np.zeros(X.shape[0], dtype=int)
    X_cluster_info = np.zeros([2,len(clusters)], dtype=int)
    print('X_cluster_new shape:' + str(X_cluster_new.shape))
    print('X_cluster_info shape:' + str(X_cluster_info.shape))
    print(X_cluster_info) 
    items = 0

    for i in range(len(clusters)):
        print('Cluster ' + str(i) + ': ' + str(len(clusters[i])))
        print('Cluster ' + str(i) + ' Center : ' + str(centers[i]))
        print('Cluster ' + str(i) + ' first 10: ' + str(clusters[i][:10]))
        print('Cluster ' + str(i) + ' last 10: ' + str(clusters[i][-10:]))
        items += len(clusters[i])

        X_cluster_info[0:,i] = i
        X_cluster_info[1:,i] = len(clusters[i])

        # Insert cluster num into X_cluster_new
        for j in range(len(clusters[i])):
            X_cluster_new[clusters[i][j]] = i

    X_cluster_new = X_cluster_new.reshape([h,w])
    print('Cluster: ' + str(len(clusters)))
    print('Cluster total items: ' + str(items))
    print('X_cluster_new shape: ' + str(X_cluster_new.shape))
    #print(X_cluster_new[:5]) #最初の5行だけprintする
    print(X_cluster_info) 

    #############################################
    # IMSHOW
    fig = plt.figure(dpi=400)
    #fig.suptitle(city + ': cluster:' + str(len(clusters)) + ' iteration:' + str(k_means.n_iter_))
    fig.suptitle('X-Means ' + city + ': cluster:' + str(len(clusters)) )
    plt.get_current_fig_manager().full_screen_toggle()
    plt.rcParams["font.size"] = 6

    plt.subplot(122)
    import matplotlib.patches as mpatches
    im = plt.imshow(X_cluster_new,cmap='jet') # for normal map use jet
    # get the colors of the values, according to the colormap used by imshow
    colors = [ im.cmap(im.norm(value)) for value in range(len(clusters))]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=i)) for i in range(len(clusters)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )    

    ###################################
    # PIE
    print('Cluster Size: ' + str(len(clusters)))
    plt.subplot(121)
    plt.title('% of clusters')
    plt.pie(x=X_cluster_info[1,:], labels=X_cluster_info[0,:], autopct='%.2f%%',colors=colors)
    plt.tight_layout()



    '''
    plt.subplot(133)
    centers = np.array(centers)
    print('Cluster Center : ' + str(centers.shape))
    for i in range(len(centers)):
        print('Cluster Center : ' + str(i) + str(centers[i]))
    plt.scatter(centers[:,0],centers[:,1],s=250, marker='*',c='red')    
    '''


    plt.savefig(imagefile + '.png')
    #plt.show(block=False)

    x_get_hist(X,cluster_num,r,X_cluster_new,imagefile,city,X_cluster_info)
    '''
        ############################################
    # 結果をプロット
    z_xm = np.ones(X.shape[0])
    for k in range(len(clusters)):
        z_xm[clusters[k]] = k+1

    plt.subplot(121)
    plt.scatter(X[:,0],X[:,1], c=z_xm)
    centers = np.array(centers)
    plt.scatter(centers[:,0],centers[:,1],s=250, marker='*',c='red')    
'''
def x_get_hist(X,cluster_num,r,X_cluster_new,imagefile,city,X_cluster_info):

    print('X_cluster_new data:' + str(X_cluster_new))
    print('X_cluster shape:' + str(X_cluster_new.shape))
    print('X shape:' + str(X.shape))

    X_cluster_new = X_cluster_new.reshape([X.shape[0],1])
    print('X_cluster shape:' + str(X_cluster_new.shape))

    #X = X[X_cluster,np.newaxis]
    X = np.hstack((X,X_cluster_new))


    print('X merged shape:' + str(X.shape))
    print('X merged data cluster info:' + str(X[:,-1]))

    #get real cct from each band  based on the last column info
    print('X extracted data cluster info:' + str(X[:,-1]))

    # Array cluster, band , std, cv
    cluster_std = np.zeros((r, cluster_num))
    cluster_cv = np.zeros((r, cluster_num))

    d = 3

    for i  in range(cluster_num):

        if i % d == 0:
            a = 0
            # Set subplot
            fig = plt.figure(figsize=(19.0, 10.0/0.96))
            fig.subplots(d, r)  # d = rows, r= columns: number of band
            fig.suptitle(city + ': cluster:' + str(cluster_num) + ' : ' + str(i/d+1) ,fontsize=10)
            plt.rcParams["font.size"] = 7
            plt.get_current_fig_manager().full_screen_toggle()

        # 最終列の値が1のデータのみ取得
        index = np.where(X[:, -1] == i)

        print('Cluster ' + str(i) + ': ' + str(X[index].shape))
        print(X[index][:2]) # show first two rows

        for j in range(r): # 
            dataset = X[index][:,j] # cluster i, band j
            print('Cluster ' + str(i) + ' Band ' + str(j+1) + ': ' + str(dataset.shape))
            print('STD: ' + str(f'{np.std(dataset):.1f}'))
            print('CV: ' + str(f'{stats.variation(dataset):.3f}'))
            print(dataset[:5])    # first 5 rows

            plt.subplot(d,r,j+1+a*r) # row d, r band, 
            plt.title('Cluster:' + str(i) + ' Band:' + str(j+1) + ' STD:' + str(f'{np.std(dataset):.1f}'))
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

    print(X_cluster_info) 
    ex_cluster = X_cluster_info[0,np.argmin(X_cluster_info[1,:])]
    print('Lowest num index:' + str(np.argmin(X_cluster_info[1,:])))
    print('Lowest num cluster:' + str(X_cluster_info[0,np.argmin(X_cluster_info[1,:])]))

    # Plot STD chart
    fig = plt.figure(figsize=(19.0, 10.0/0.96))
    plt.xlabel("Band")
    plt.ylabel("Standard Devisation")
    plt.title('X-Means Cluster:' + str(cluster_num) + ' Band:' + str(r) + ' Standard Divisation')
    plt.xticks(list(range(0, r)),list(range(1, r+1)))
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
    plt.title('X-Means Cluster:' + str(cluster_num) + ' Band:' + str(r) + ' Coefficient of Variation')
    plt.xticks(list(range(0, r)),list(range(1, r+1)))
    for c in range(cluster_num):
        if c != ex_cluster:
            plt.plot(cluster_cv[:,c], label="Cluster {}".format(c), marker="o", linestyle = "--")
    plt.legend()
    plt.savefig(imagefile + '-' + str(i+1) + '-cv.png')
    #plt.show(block=False)