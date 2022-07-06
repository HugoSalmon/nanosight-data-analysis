


from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema


from pathlib import Path


from seaborn import heatmap


import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
from scipy.stats import norm
import numpy as np
 
from ML_tools.distance_matrix_computation import compute_distance_matrix_distribs, compute_distance_matrix

from scipy.stats import rv_histogram

from paths import datapath
data_prod_path = Path(datapath, 'Prod hASC Lea')

import pandas
from sklearn.cluster import KMeans

from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import itertools
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
directory = "Prod hASC Lea"
from ML_tools.hierarchical_clustering import plot_hierarchical_clustering
from constants import quantile
import ot
import seaborn as sns
from ML_tools.hierarchical_clustering import colors_dendrogram
# from pyemd import emd
from ot import unbalanced












    


def plot_paired_matrix(distance_matrix_dataframe, path_to_save_fig, title, index_order=None):

    fig, ax = plt.subplots(1, 1, figsize=(20,20))

    if index_order:
        distance_matrix_dataframe = distance_matrix_dataframe.iloc[index_order, index_order]

    heatmap(distance_matrix_dataframe, ax=ax, xticklabels=True, yticklabels=True,
        annot = True, annot_kws={"size":10}, cbar_kws={'label': "Distance", 'shrink':0.3},#, "location":"top", "use_gridspec":False},
        fmt=".3", 
        cmap="Blues", linewidths=0.1, linecolor='gray')
 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    ax.set_xticklabels([col for col in ax.get_xticklabels()], rotation=45, ha="left", rotation_mode="anchor", fontsize=13)

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(Path(path_to_save_fig, title+"_paired_matrix.pdf"))
    
          


    

def plot_clustering_results(list_distribs, bin_centers, labels, normalized, list_names, path_to_save, title):

    if normalized:
        bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
        bins =  [0] + list(bin_centers + bin_diffs/2)
                
        list_area = [np.sum(distrib * bin_diffs) for distrib in list_distribs]
        normalized_distribs = [distrib / list_area[u] for u, distrib in enumerate(list_distribs)]

    
        distribs_to_plot = normalized_distribs
    else:
        distribs_to_plot = list_distribs

    fig, ax = plt.subplots(1, figsize=(20,10))
    # fig2, ax2 = plt.subplots(1 + len(list(itertools.combinations(np.unique(labels), 2))),1, figsize=(20,10), sharex=True, sharey=True)

    colors = colors_dendrogram
    
    stop = 250
    
    distribs_to_plot = np.array(distribs_to_plot)
    
    fig3, ax3 = plt.subplots(len(np.unique(labels)),1, figsize=(10,10), sharey=True)
    
    for i in range(len(np.unique(labels))):
        
        
        distribs_cluster = distribs_to_plot[(labels==i)]
        
        
        names_cluster = np.array(list_names)[np.where(labels==i)]
        print("Cluster", i+1)
        print(names_cluster)
        
        distribs_cluster = distribs_cluster[:,:stop]
        bin_centers = bin_centers[:stop]
        
        color_cluster = colors[i]
        
        mean_distrib_cluster = np.mean(distribs_cluster, axis=0)
        std_distrib_cluster = np.std(distribs_cluster, axis=0) 

        ste_distrib_cluster = std_distrib_cluster / np.sqrt(len(distribs_cluster))
        
        ci = quantile*ste_distrib_cluster

        ax.plot(bin_centers, mean_distrib_cluster, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")")   
    
        ax.fill_between(x=bin_centers, y1=mean_distrib_cluster-ci, y2=mean_distrib_cluster+ci, color=color_cluster, alpha=.1)
        ax.legend(fontsize=15)
        
    
        # ax3[i].fill_between(x=bin_centers, y1=mean_distrib_cluster-ci, y2=mean_distrib_cluster+ci, color=color_cluster, alpha=.1)
        
    
        for v, distrib in enumerate(distribs_cluster):
            
            ax.plot(bin_centers, distrib, color=color_cluster, alpha=0.2)
            
            # if v==0:
            #     ax2[-1].plot(bin_centers, distrib, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")")
            # else:
            #     ax2[-1].plot(bin_centers, distrib, color=color_cluster)
                
            ax3[i].plot(bin_centers, distrib, alpha=0.3, label=names_cluster[v])
            ax3[i].legend(fontsize=8)
                
        ax3[i].plot(bin_centers, mean_distrib_cluster, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")")   

    # for k, (u,v) in enumerate(list(itertools.combinations(np.unique(labels), 2))):
        
    #     distribs_cluster1 = distribs_to_plot[(labels==u)]
    #     distribs_cluster2 = distribs_to_plot[(labels==v)]
    #     distribs_cluster1 = distribs_cluster1[:,:stop]
    #     distribs_cluster2 = distribs_cluster2[:,:stop]
        
    #     color_cluster1 = colors[u]
    #     color_cluster2 = colors[v]
        
    #     for z, distrib in enumerate(distribs_cluster1):
    #         if z==0:
    #             ax2[k].plot(distrib, color=color_cluster1, label="Cluster "+str(u+1)+" ("+"n="+str(np.sum(labels==u))+")")
    #         else:
    #             ax2[k].plot(distrib, color=color_cluster1)
    
    #     for z, distrib in enumerate(distribs_cluster2):
    #         if z==0:
    #             ax2[k].plot(distrib, color=color_cluster2, label="Cluster "+str(v+1)+" ("+"n="+str(np.sum(labels==v))+")")
    #         else:
    #             ax2[k].plot(distrib, color=color_cluster2)
    
    #     ax2[k].legend(fontsize=15)  
    
    # ax2[-1].legend(fontsize=13)

    # fig2.savefig(Path(str(path_to_save)+"_paired_comparison.pdf"))
    
    fig.tight_layout()        
    fig.savefig(Path(path_to_save, title+"_mean_distribs.pdf"))
    
    fig3.tight_layout()
    fig3.savefig(Path(path_to_save, title+"_distribs.pdf"))
    
    plt.close(fig)
    plt.close(fig3)
        
#
#""""""""""""""""""""""""""""""""
#""" Spectral clustering """
#""""""""""""""""""""""""""""""""
#
#
###
##n_neighbors = int(np.log(len(list_prods)))
##
##
##connectivity = kneighbors_graph(distance_matrix, n_neighbors=n_neighbors,
##                                            include_self=True, metric="precomputed")
##
##
##
##affinity_matrix = 0.5 * (connectivity + connectivity.T)
##
##
##
##normed_dist = distance_matrix / np.max(distance_matrix)
##affinity_matrix = 1 - normed_dist
##
##
##laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True,
##                                  return_diag=True)
##
##
##eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian)
##
##fig, ax = plt.subplots(1)
##ax.scatter(np.arange(len(eigenvalues)), eigenvalues)
##ax.set_title("Eigenvalues of the laplacian matrix")
##
#### 2 eigenvalues before gap
####
####
####
####
##n_components = 3
##n_clusters = n_components
##
##
##
##clustering_method = SpectralClustering(n_clusters = n_clusters, \
##                                               n_components = n_components, \
##                                               affinity = 'precomputed', \
##                                               n_neighbors = None, random_state=random_state)
##
##clust = clustering_method.fit(affinity_matrix)
##
##labels = clust.labels_
##print(silhouette_score(distance_matrix, labels))
##
##
##

#