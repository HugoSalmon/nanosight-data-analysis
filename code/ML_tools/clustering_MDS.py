










from stat_tools.fit_gaussian_mixture import CustomGaussianMixture
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema


from pathlib import Path
from paths import figpath


from seaborn import heatmap

import dash
from dash import html
from dash import dcc
from plotly.tools import mpl_to_plotly

from load_distrib import get_mean_hist, get_hist
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
import itertools
import webbrowser
from threading import Timer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
from scipy.stats import norm
import dash_bootstrap_components as dbc
from sklearn.metrics import silhouette_score
import pickle
import os
from paths import datapath

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from skimage.io import imread
import plotly.express as px

from load_files import dic_files, file_exists
import numpy as np
from plotly.subplots import make_subplots
 
from distribution_comparison import kolmogorov_smirnoff_test,\
                                custom_KS_distance, W_dist, energy_dist, \
                                Wasserstein_distance_non_normalised, KS_distance_non_normalised
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
from hierarchical_clustering import plot_hierarchical_clustering

#import ot


prods = os.listdir(Path(datapath, directory))
productions = [prod for prod in prods if "organisation fichier" not in prod and "M8" not in prod]

dilutions = ["d5", "d10", "d20", "d50", "d100", "d500", "d1000"]
dilutions = [dil + "-" for dil in dilutions] + [dil + " " for dil in dilutions]

times = ["t0", "t2", "t4", "t4-UC", "t4-UCbis"]
replicates = ["01", "02", "03"]

random_state = 50


dist = "KS"
#dist = "Wasserstein"
#dist="energy"




distribs = []
distribs_normalized = []
list_prods = []
for production in productions:  
    for time in times:
        if not file_exists(production, time):
            continue

        df = get_mean_hist(production, time)
        concentration = df["mean_c"].values
        concentration_normalized = df["mean_c_normalized"].values

        
        bin_centers = df["Bin centre (nm)"].values
        if len(bin_centers)!=1000:
            continue
        
        bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
        bins = np.array([0] + list(bin_centers + bin_diffs/2))



        if True:#production+"_"+time in cluster_2:
            distribs_normalized.append(df["mean_c_normalized"].values)
            distribs.append(df["mean_c"].values)
        
            list_prods.append(production+"_"+time)
            
distribs = np.array(distribs)        
distribs_normalized = np.array(distribs_normalized)        
        
  
######## Scaling non normalized


dist = "Wasserstein"   

if dist=="KS":
    dist_func = lambda x, y : KS_distance_non_normalised(x, y)

elif dist=="Wasserstein":
    dist_func = lambda x, y : Wasserstein_distance_non_normalised(x, y)

elif dist=="energy":
    dist_func = lambda x, y : energy_dist(bin_centers, bin_centers, x, y)
    
    
distance_matrix = np.asarray([[dist_func(distribs[u], distribs[v]) for u in range(len(list_prods))] for v in range(len(list_prods))])


list_prods_names = ["\n".join(prod.replace("_","-").split("-")) for prod in list_prods]

distance_matrix_dataframe = pandas.DataFrame(distance_matrix, index=list_prods_names, columns=list_prods_names)


from sklearn.manifold import MDS

embedding = MDS(n_components=5, dissimilarity="precomputed")
X_transformed_non_normalized = embedding.fit_transform(distance_matrix)







  
######## Scaling non normalized


dist = "KS"   

if dist=="KS":
    dist_func = lambda x, y : custom_KS_distance(x, y)

elif dist=="Wasserstein":
    dist_func = lambda x, y : W_dist(bin_centers, bin_centers, x, y)

elif dist=="energy":
    dist_func = lambda x, y : energy_dist(bin_centers, bin_centers, x, y)
    
    
distance_matrix = np.asarray([[dist_func(distribs_normalized[u], distribs_normalized[v]) for u in range(len(list_prods))] for v in range(len(list_prods))])


list_prods_names = ["\n".join(prod.replace("_","-").split("-")) for prod in list_prods]

distance_matrix_dataframe = pandas.DataFrame(distance_matrix, index=list_prods_names, columns=list_prods_names)


from sklearn.manifold import MDS

embedding = MDS(n_components=5, dissimilarity="precomputed")
X_transformed_normalized = embedding.fit_transform(distance_matrix)


X_final = np.concatenate([X_transformed_non_normalized, X_transformed_normalized], axis=1)

X_final = pandas.DataFrame(X_final, columns=["Feature "+str(i) for i in range(X_final.shape[1])])

X_final.index = list_prods

for col in X_final.columns:
    
    X_final[col] = ( X_final[col] - np.mean(X_final[col]) ) / np.std(X_final[col])

""""""""""""""""""""""""""""""""
""" hierarchical clustering """
""""""""""""""""""""""""""""""""
#

#silhouettes = []
#candidates = np.arange(2, 6)
#for n_clusters in candidates:      
#    index_, labels, silhouette = plot_hierarchical_clustering(distance_matrix_dataframe, \
#                        metric="precomputed", linkage_method="complete", threshold=0.7, \
#                        distance_matrix=True, criterion="maxclust", max_clust = n_clusters)
#    
#    silhouettes.append(silhouette)
#    
#best = np.argmax(silhouettes)
#n_clusters = candidates[best]




threshold = 0.6
index_, labels, silhouette = plot_hierarchical_clustering(X_final, \
                    metric="euclidean", linkage_method="ward", threshold=threshold, \
                    distance_matrix=True, criterion="threshold", path=Path(figpath, "clustering_"+str(threshold)+"_"+dist+".pdf"))


index_ = np.array(index_)
ordered_labels = np.array(labels[index_])

silhouettes = silhouette_samples(X_final, labels, metric="euclidean")
for cluster in np.unique(labels):
    index_cluster = (labels==cluster)
    silhouette_cluster = silhouettes[index_cluster]
    print(cluster, np.mean(silhouette_cluster), np.std(silhouette_cluster))

    
    

""""""""""""""""""""""""""""""""
""" Spectral clustering """
""""""""""""""""""""""""""""""""


##
#n_neighbors = int(np.log(len(list_prods)))
#
#
#connectivity = kneighbors_graph(distance_matrix, n_neighbors=n_neighbors,
#                                            include_self=True, metric="precomputed")
#
#
#
#affinity_matrix = 0.5 * (connectivity + connectivity.T)
#
#
#
#normed_dist = distance_matrix / np.max(distance_matrix)
#affinity_matrix = 1 - normed_dist
#
#
#laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True,
#                                  return_diag=True)
#
#
#eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian)
#
#fig, ax = plt.subplots(1)
#ax.scatter(np.arange(len(eigenvalues)), eigenvalues)
#ax.set_title("Eigenvalues of the laplacian matrix")
#
### 2 eigenvalues before gap
###
###
###
###
#n_components = 3
#n_clusters = n_components
#
#
#
#clustering_method = SpectralClustering(n_clusters = n_clusters, \
#                                               n_components = n_components, \
#                                               affinity = 'precomputed', \
#                                               n_neighbors = None, random_state=random_state)
#
#clust = clustering_method.fit(affinity_matrix)
#
#labels = clust.labels_
#print(silhouette_score(distance_matrix, labels))
#
#
#
#



#
""" Plot results """


fig, ax = plt.subplots(1, figsize=(20,10))

fig2, ax2 = plt.subplots(1 + len(list(itertools.combinations(np.unique(labels), 2))),1, figsize=(20,10), sharex=True, sharey=True)


colors = ["royalblue", "indianred", "mediumseagreen","orange","darkblue", "violet"]


#np.random.shuffle(labels)

stop = 250

for i in range(len(np.unique(labels))):
    distribs_cluster = distribs[(labels==i)]
    
    prod_cluster = np.array(list_prods)[np.where(labels==i)]
    print("Cluster", i+1)
    print(prod_cluster)
    
    distribs_cluster = distribs_cluster[:,:stop]
    bin_centers = bin_centers[:stop]
    
    color_cluster = colors[i]
    
    mean_distrib_cluster = np.mean(distribs_cluster, axis=0)
    ste_distrib_cluster = np.std(distribs_cluster, axis=0) / np.sqrt(len(distribs_cluster))
    
        
    
#    weights = np.array([1/len(distribs_cluster)]*len(distribs_cluster))
#    
#    # l2bary
#    bary_l2 = distribs_cluster.T.dot(weights)
#
#    n = len(bin_centers)
#    M = ot.utils.dist0(n)
#    M /= M.max()    
#    
#    # wasserstein
#    reg = 1e-3
#    ot.tic()
#    bary_wass = ot.bregman.barycenter(distribs_cluster.T, M, reg, weights)
#    ot.toc()

#    ax.plot(bin_centers, bary_wass, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")", linestyle="--")   

    
    ci = 1.96*ste_distrib_cluster
#    ci = ste_distrib_cluster

    ax.plot(bin_centers, mean_distrib_cluster, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")")   

    ax.fill_between(x=bin_centers, y1=mean_distrib_cluster-ci, y2=mean_distrib_cluster+ci, color=color_cluster, alpha=.1)
    ax.legend(fontsize=15)

    for v, distrib in enumerate(distribs_cluster):
        if v==0:
            ax2[-1].plot(bin_centers, distrib, color=color_cluster, label="Cluster "+str(i+1)+" ("+"n="+str(np.sum(labels==i))+")")
        else:
            ax2[-1].plot(bin_centers, distrib, color=color_cluster)
        
ax2[-1].legend(fontsize=13)
    

for k, (u,v) in enumerate(list(itertools.combinations(np.unique(labels), 2))):
    
    distribs_cluster1 = distribs[(labels==u)]
    distribs_cluster2 = distribs[(labels==v)]
    distribs_cluster1 = distribs_cluster1[:,:stop]
    distribs_cluster2 = distribs_cluster2[:,:stop]
    
    color_cluster1 = colors[u]
    color_cluster2 = colors[v]
    
    for z, distrib in enumerate(distribs_cluster1):
        if z==0:
            ax2[k].plot(distrib, color=color_cluster1, label="Cluster "+str(u+1)+" ("+"n="+str(np.sum(labels==u))+")")
        else:
            ax2[k].plot(distrib, color=color_cluster1)

    for z, distrib in enumerate(distribs_cluster2):
        if z==0:
            ax2[k].plot(distrib, color=color_cluster2, label="Cluster "+str(v+1)+" ("+"n="+str(np.sum(labels==v))+")")
        else:
            ax2[k].plot(distrib, color=color_cluster2)

    ax2[k].legend(fontsize=15)  
    
    
fig2.savefig(Path(figpath,"clustering_"+str(threshold)+"_"+dist+"paired_comparison.pdf"))
fig.savefig(Path(figpath,"clustering_"+str(threshold)+"_"+dist+"barycenters.pdf"))






#
#""" DBSCAN """
#
#from sklearn.cluster import DBSCAN
#clustering = DBSCAN(eps=3, min_samples=2, metric="precomputed").fit(distance_matrix)
#clustering.labels_