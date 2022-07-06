
import numpy as np
import pandas
from scipy.integrate import simpson

from scipy.stats import wasserstein_distance
from ML_tools.distribution_comparison import kolmogorov_smirnoff_test_weighted
                                            



def compute_distance_matrix(list_data, list_names, distance):
    
    if distance=="euclidean":
        dist_func = lambda x, y: np.linalg.norm(x-y)

    distance_matrix = np.asarray([[dist_func(list_data[u], list_data[v]) \
        for u in range(len(list_data))] for v in range(len(list_data))])

    distance_matrix_dataframe = pandas.DataFrame(distance_matrix, 
                                                 index=list_names, 
                                                 columns=list_names)
    
    return distance_matrix_dataframe

def compute_distance_matrix_distribs(list_distribs, bin_centers, normalized, distance, list_names, ax):

    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
    list_area = [simpson(x=bin_centers, y=distrib) for distrib in list_distribs]
    normalized_distribs = [distrib / list_area[u] for u, distrib in enumerate(list_distribs)]

    if distance=="Wasserstein" and normalized:
        dist_func = lambda x, y: wasserstein_distance(bin_centers, bin_centers, x, y)

    if distance=="emd" and not normalized:
        dist_func = lambda x, y: compute_emd(bin_centers, x,y)

    if distance=="Kolmogorov" and normalized:
        dist_func = lambda x, y: kolmogorov_smirnoff_test_weighted(bin_centers, bin_centers, x, y)[0]

    if normalized:
        distance_matrix = np.asarray([[dist_func(normalized_distribs[u], normalized_distribs[v]) \
            for u in range(len(normalized_distribs))] for v in range(len(normalized_distribs))])
    else:
        distance_matrix = np.asarray([[dist_func(list_distribs[u], list_distribs[v]) \
            for u in range(len(list_distribs))] for v in range(len(list_distribs))])

    distance_matrix_dataframe = pandas.DataFrame(distance_matrix, 
                                                 index=list_names, 
                                                 columns=list_names)     
    
    for i in range(len(normalized_distribs)):
        ax.plot(normalized_distribs[i], label=list_names[i])
    ax.legend(fontsize=15)

    return distance_matrix_dataframe       



def compute_distance_matrix_mixte_wasserstein_euclidean(list_distribs, bin_centers, list_data, list_names):
        
        
    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)

    wasserstein = lambda x, y: compute_wasserstein_distance(bin_centers, bin_centers, x, y)
    
    euclidean = lambda x, y: np.linalg.norm(x-y)
            

                
    list_area = [np.sum(distrib * bin_diffs) for distrib in list_distribs]
    normalized_distribs = [distrib / list_area[u] for u, distrib in enumerate(list_distribs)]

    
    wasserstein_distance_matrix = np.asarray([[wasserstein(normalized_distribs[u], normalized_distribs[v]) \
        for u in range(len(normalized_distribs))] for v in range(len(normalized_distribs))])
    
     
    euclidean_distance_matrix = np.asarray([[euclidean(list_area[u], list_area[v]) \
        for u in range(len(list_area))] for v in range(len(list_area))])
    
               
        
    wasserstein_distance_matrix = wasserstein_distance_matrix / np.max(wasserstein_distance_matrix)
    euclidean_distance_matrix = euclidean_distance_matrix / np.max(euclidean_distance_matrix)
    
    
    weight1 = 0.5
    weight2 = 0.5
    
    distance_matrix = weight1*wasserstein_distance_matrix + weight2*euclidean_distance_matrix
        

    distance_matrix_dataframe = pandas.DataFrame(distance_matrix, 
                                                 index=list_names, 
                                                 columns=list_names)
    
    return distance_matrix_dataframe


def compute_test_diff_matrix_distribs(list_distribs, bin_centers, normalized, test, list_names):

    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
    list_area = [np.sum(distrib * bin_diffs) for distrib in list_distribs]
    normalized_distribs = [distrib / list_area[u] for u, distrib in enumerate(list_distribs)]

    if test=="Kolmogorov" and normalized:
        dist_func = lambda x, y: kolmogorov_smirnoff_test_weighted(bin_centers, bin_centers, x, y)[1]

    distance_matrix = np.asarray([[dist_func(list_distribs[u], list_distribs[v]) \
        for u in range(len(list_distribs))] for v in range(len(list_distribs))])

    distance_matrix_dataframe = pandas.DataFrame(distance_matrix, 
                                                 index=list_names, 
                                                 columns=list_names)     

    return distance_matrix_dataframe       
