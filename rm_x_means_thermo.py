import numpy as np
import matplotlib.pyplot as plt
import collections
import pyclustering
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.utils import draw_clusters
from scipy import stats

def x_cal_thermo(X,city,r,cluster_num,h,w,now,yeardate):
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
    X_cluster_info = np.zeros([4,len(clusters)], dtype=int)
    print('X_cluster_new shape:' + str(X_cluster_new.shape))
    print('X_cluster_info shape:' + str(X_cluster_info.shape))
    print(centers) 
    items = 0

    for i in range(len(clusters)):
        print('Cluster ' + str(i) + ': ' + str(len(clusters[i])))
        print('Cluster ' + str(i) + ' Center : ' + str(centers[i][0]))
        print('Cluster ' + str(i) + ' first 10: ' + str(clusters[i][:10]))
        print('Cluster ' + str(i) + ' last 10: ' + str(clusters[i][-10:]))
        items += len(clusters[i])

        X_cluster_info[0:1,i] = i
        X_cluster_info[1:2,i] = len(clusters[i])
        X_cluster_info[2:3,i] = centers[i][0]

    # Sorty cluster by actual band 10 center assending
    X_cluster_info = X_cluster_info[:, X_cluster_info[2,:].argsort()]

    # Add sort num
    for i in range(len(clusters)):
        X_cluster_info[3:4,i] = i
        cn = int(X_cluster_info[0:1,i])
        print('Cluster to replace:' + str(cn) + ', New Cluster' + str(i))
        # Insert cluster num into X_cluster_new
        for j in range(len(clusters[cn])):
            X_cluster_new[clusters[cn][j]] = i

    X_cluster_new = X_cluster_new.reshape([h,w])
    print('Cluster: ' + str(len(clusters)))
    print('Cluster total items: ' + str(items))
    print('X_cluster_new shape: ' + str(X_cluster_new.shape))
    print(X_cluster_info) 

    #############################################
    # IMSHOW
    fig = plt.figure(dpi=400)
    #fig.suptitle(city + ': cluster:' + str(len(clusters)) + ' iteration:' + str(k_means.n_iter_))
    fig.suptitle('X-Means ' + yeardate + ' ' + city + ': cluster:' + str(len(clusters)) )
    plt.get_current_fig_manager().full_screen_toggle()
    plt.rcParams["font.size"] = 6

    plt.subplot(122)
    import matplotlib.patches as mpatches
    im = plt.imshow(X_cluster_new,cmap='Reds') # for normal map use jet
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
    plt.pie(x=X_cluster_info[1,:], labels=X_cluster_info[3,:], autopct='%.2f%%',colors=colors)
    plt.tight_layout()



    '''
    plt.subplot(133)
    centers = np.array(centers)
    print('Cluster Center : ' + str(centers.shape))
    for i in range(len(centers)):
        print('Cluster Center : ' + str(i) + str(centers[i]))
    plt.scatter(centers[:,0],centers[:,1],s=250, marker='*',c='red')    
    '''


    plt.savefig(imagefile + '-heat.png')
    #plt.show(block=False)

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
