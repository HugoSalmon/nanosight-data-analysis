

from scipy.stats import norm

confidence = 0.95
quantile = norm.ppf((1 + confidence)/2)


dir_plots = "data_illustrations"
dir_csv_exports = "data_csv_exports"
text_button_data_illustrations = "Data illustrations"
text_button_csv_export = "Data csv export"
dir_test = "Kolmogovov_Smirnov_test"
text_button_test = "Kolmogorov-Smirnov two-sample test"
text_button_cluster_attributes = "Clustering size concentration attributes"
text_button_cluster_distributions = "Clustering size concentration distributions"
dir_distributions_clustering = "clustering_size_concentration_distributions"
dir_attributes_clustering = "clustering_size_concentration_attributes"