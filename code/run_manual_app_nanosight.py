

from pathlib import Path
from paths import datapath
from app_nanosight import App

    
# """ Lea """

path = Path(datapath, "data_Sarah_Reddy")

data_app = App(data_dir=path,
                autosampler=False,
                dilution_prefix="d",
                replicate_prefix="-")

data_app.run_manual()
# data_app.plot_nanosight()
data_app.export_data()
# data_app.run_clustering_total_concentration_nanosight()



# data_app.plot_nanosight_kolmogorov_test_matrix()



# """ BALF """

# path = Path(datapath, "data_Sarah_Reddy","BALFs")    

# data_app = App(videodrop_dir=None,
#                 data_dir=path,
#                 autosampler=False,
#                 # dilution_videodrop="d",
#                 dilution="d")

# data_app.run_manual()



# data_app.export_nanosight()

# data_app.plot_nanosight_distributions()


# data_app.run_clustering_normalized_nanosight_size_concentration_distributions_wasserstein()


# data_app.run_clustering_total_concentration_nanosight()
# data_app.run_clustering_combined_distances_nanosight()


# data_app.plot_nanosight_kolmogorov_test_matrix()


