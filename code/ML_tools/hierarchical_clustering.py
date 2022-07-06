
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager
rcParams['font.family'] = 'serif'


from seaborn import heatmap
from pathlib import Path

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, dendrogram, set_link_color_palette
from sklearn.metrics import silhouette_score



import matplotlib.patches as mpatches

colors_dendrogram = ['indianred','royalblue','mediumseagreen','skyblue','darkturquoise','mediumvioletred', 'darksalmon','blue']

def compute_hierarchical_tree(data, metric, linkage_method="ward"):

    n = len(data)

    if metric!="precomputed":
        flattened_distance_matrix = pdist(data, metric = metric)
        square_distance_matrix = squareform(flattened_distance_matrix)
    else:
#        unsquareform = lambda a: a[np.nonzero(np.triu(a))]
#        flattened_distance_matrix = unsquareform(data)
        flattened_distance_matrix = squareform(data) # function works vice versa
        square_distance_matrix = data

    tree = linkage(flattened_distance_matrix, method=linkage_method)

    tree_order = get_tree_order(tree, n + n-2)
    
    
    ordered_dist_matrix = np.zeros((n,n))    

    a,b = np.triu_indices(n,k=1)
    ordered_dist_matrix[a,b] = square_distance_matrix[ [tree_order[i] for i in a], [tree_order[j] for j in b]]
    ordered_dist_matrix[b,a] = ordered_dist_matrix[a,b]
    
    return tree, tree_order, ordered_dist_matrix

def get_tree_order(linkage_matrix, cluster_index):
  
    n = len(linkage_matrix) + 1

    if cluster_index < n:
        return [cluster_index]
    else:
        left = int(linkage_matrix[cluster_index-n, 0])
        right = int(linkage_matrix[cluster_index-n, 1])
        return (get_tree_order(linkage_matrix, left) + get_tree_order(linkage_matrix, right))




#def plot_results(table, ax):
#
#    fig, ax = plt.subplots(2, 1, figsize=(16,13))
#    heat_map = heatmap(table[index_], ax=ax[0])
#    dend = plot_dendrogram(ax[1], res_linkage, table.index, threshold=0.7)
#    ax[0].set_aspect("equal")
#    ax[0].xaxis.set_tick_params(labelsize=2)
#    ax[0].yaxis.set_tick_params(labelsize=2)
#
#
#






# def optimize_hierarchical_clustering(distance_matrix_dataframe, threshold=0.7, labelsize=10, linkage_method="complete",
#                                          path_to_save_fig=None, maxclust=4, criterion="threshold"):


#     fig, ax = plt.subplots(1)


      
#     index_, labels, silhouette = plot_hierarchical_clustering(distance_matrix_dataframe, \
#                         metric="precomputed", linkage_method=linkage_method, threshold=threshold, \
#                         distance_matrix=True, criterion="threshold", 
#                         path=path_to_save_fig,
#                         labelsize=labelsize, max_clust=maxclust, plot=True)
        
#     print("silhouette score", silhouette)
        
#     index_ = np.array(index_)
#     ordered_labels = np.array(labels[index_])
    
#     # silhouettes = silhouette_samples(distance_matrix, labels, metric="precomputed")
#     # for cluster in np.unique(labels):
#     #     index_cluster = (labels==cluster)
#     #     silhouette_cluster = silhouettes[index_cluster]
#     #     print("silhouette score cluster", cluster, ":", np.mean(silhouette_cluster), "std=", np.std(silhouette_cluster))

#     return distance_matrix_dataframe, index_, labels, silhouette

def run_hierarchical_clustering(table, metric, linkage_method, criterion="maxclust", threshold=0.7, max_clust=2, labelsize=8, optimize=True):

    if optimize:
        
        fig, ax = plt.subplots(1)
        all_silhouettes = []
        thresholds = np.arange(0.025, 1, 0.025)
        
        for thres in thresholds:
              
            results_clustering = run_hierarchical_clustering(table, \
                                metric=metric, linkage_method=linkage_method, threshold=thres, \
                                criterion="threshold", optimize=False)
                
            if results_clustering["silhouette"] is None:
                silhouette = 0
            else:
                silhouette = results_clustering["silhouette"]
            
            all_silhouettes.append(silhouette)

            ax.scatter(thres, silhouette)
            
        all_silhouettes = np.array(all_silhouettes)
        best = all_silhouettes.argmax()
        threshold = thresholds[best]
        criterion = "threshold"
    

    data = np.array(table)

    if metric!="precomputed":
        flattened_distance_matrix = pdist(data, metric = metric)
        square_distance_matrix = squareform(flattened_distance_matrix)
    else:
        flattened_distance_matrix = squareform(data) # function works vice versa
        square_distance_matrix = data


 
    linkage_matrix, ordered_index_dendrogram, ordered_dist_matrix = compute_hierarchical_tree(data, metric=metric, linkage_method=linkage_method)

    ordered_names_dendrogram = np.array(table.index)[np.array(ordered_index_dendrogram)]

    corresp_index = {str(table.index[i]):i for i in range(len(table.index))}


    if criterion=="threshold":
        labels = fcluster(linkage_matrix, t=threshold*np.max(linkage_matrix[:,2]), criterion="distance")

    elif criterion=="maxclust":
 
        labels = fcluster(linkage_matrix, t=max_clust, criterion=criterion)


    labels-=1

    if len(np.unique(labels))>1 and len(np.unique(labels))<len(square_distance_matrix):
        silhouette = silhouette_score(square_distance_matrix, labels, metric='precomputed')
        # print("silhouette score : "+str(silhouette_hierarchique))
    else:
        silhouette = None
        
    ordered_dist_matrix = pandas.DataFrame(ordered_dist_matrix, index=ordered_names_dendrogram, columns=ordered_names_dendrogram)   

    if threshold is None:

        distance_threshold=linkage_matrix[-n_clusters+1,2]  
        threshold = distance_threshold / np.max(linkage_matrix[:,2])


    return {"linkage_matrix":linkage_matrix, 
            "ordered_index_dendrogram":ordered_index_dendrogram, 
            "ordered_dist_matrix":ordered_dist_matrix, 
            "labels":labels, 
            "silhouette":silhouette, 
            "threshold":threshold}




def plot_hierarchical_clustering(table, results_clustering, labelsize=13, title="", path_to_save_fig=None):   

    fig, ax = plt.subplots(2, 1, figsize=(20,10))

    plot_heatmap(results_clustering["ordered_dist_matrix"], ax=ax[0])
    ax_dend = ax[1]
    
    fig2, ax2 = plt.subplots(1, figsize=(15,15))
    heatmap(results_clustering["ordered_dist_matrix"], ax=ax2, xticklabels=True, yticklabels=True,
        annot = True, annot_kws={"size":10}, cbar_kws={'label': "Distance", 'shrink':0.3},#, "location":"top", "use_gridspec":False},
        fmt=".3", 
        cmap="Blues", linewidths=0.1, linecolor='gray'
        )
         
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=13)
    ax2.set_xticklabels([col for col in ax2.get_xticklabels()], rotation=45, ha="left", rotation_mode="anchor", fontsize=13)

    ax2.set_aspect("equal")
    fig2.tight_layout()
    

    # corresp_index = {str(table.index[i]):i for i in range(len(table.index))}
    

    dend = plot_dendrogram(ax_dend, results_clustering["linkage_matrix"], labels=table.index, threshold=results_clustering["threshold"])

    ax_dend.set_xticklabels(ax_dend.get_xticklabels(), fontsize=labelsize, rotation=45, rotation_mode="anchor", ha="right")

    
    fig.tight_layout()

    if path_to_save_fig is not None:
        fig2.savefig(Path(path_to_save_fig, title+"_distance_matrix.pdf"))
    
        fig.savefig(Path(path_to_save_fig, title+"_hierarchical_clustering.pdf"))
        
        plt.close(fig)
        plt.close(fig2)
        
        results_clustering["ordered_dist_matrix"].to_csv(Path(path_to_save_fig, title+"_distance_matrix.csv"))



# def compute_n_clusters_linkage_matrix(linkage_matrix, n_data):
    
#     n_cluster_values = []
    
#     n_clust = n_data
    
#     clusters = {}
    
#     n_threshold = n_data
    
#     for i in range(len(linkage_matrix)):
        
#         ids = linkage_matrix[i,:2]
        
#         n_clust -=1

#         # if ids[0] < n_data and ids[1] < n_data:
#         #     n_clust -=1
            
#         # elif ids[0] >= n_data and ids[1] >= n_data:
#         #     n_clust -=1

#         n_cluster_values.append(n_clust)
        
#     return np.array(n_cluster_values)



def plot_heatmap(table, ax):

    heatmap(table, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")




def plot_dendrogram(ax, res_linkage, labels, threshold=0.7):
    
    set_link_color_palette(colors_dendrogram)
    dend = dendrogram(res_linkage, ax=ax, labels=labels, above_threshold_color='k', color_threshold=threshold*np.max(res_linkage[:,2]))
    
    return dend





if __name__=="__main__":     
    
    
    cluster1 = 10 + np.random.randn(5)
    cluster2 = np.random.randn(10)
    data = np.concatenate([cluster1, cluster2]).reshape(-1,1)
    individuals = ["I"+str(i) for i in range(15)]
    table = pandas.DataFrame(data, index=individuals)

    

    """ clustering hierarchique """

    results = run_hierarchical_clustering(table, metric="euclidean", \
                                                              linkage_method="average", \
                                                              threshold=0.7, \
                                                              criterion="threshold",
                                                              optimize=True)

    plot_hierarchical_clustering(table, results, labelsize=13, title="", 
                                 path_to_save_fig=None)
    