




import numpy as np
from paths import datapath
from pathlib import Path
import os


def is_date(string):
    
    split = string.split("-")
    
    if len(split)!=3:
        return False

    if np.sum([split[u].isdigit() for u in range(len(split))])!=3:
        return False
        
    if len(split[0])!=4 or len(split[1])!=2 or len(split[2])!=2:
        return False
    
    return True



def is_time(string):
    
    split = string.split("-")
    
    if len(split)!=3:
        return False
    
    if np.sum([split[u].isdigit() for u in range(len(split))])!=3:
        return False
    
    if len(split[0])!=2 or len(split[1])!=2 or len(split[2])!=2:
        return False
    
    return True



def list_nanosight_files_in_directory(directory_path, raw_suffix="_1"):


    path_dic = {}
    list_dir = []
    for (dirpath, dirnames, filenames) in os.walk(directory_path):
        list_dir += dirnames
        for file in filenames:
            path_dic[file] = Path(dirpath, file)
            

    experiments = [file.replace("-ExperimentSummary.csv","") for file in path_dic 
                    if "ExperimentSummary.csv" in file and "~lock" not in file]


    files_dic = {}
 
    for experiment in experiments:

        last_element = experiment.split(" ")[-1]
        
        datetime_at_the_end = (is_date(last_element) or is_time(last_element))

        experiment_root = experiment
            
        while datetime_at_the_end:

            experiment_root = experiment_root.replace(" "+last_element, "")

            last_element = experiment_root.split(" ")[-1]
            
            datetime_at_the_end = (is_date(last_element) or is_time(last_element))
            
 
        all_tracks_files = [path_dic[file] for file in path_dic
                    if experiment_root in file and "AllTracks" in file and "~lock" not in file]      
        
        particle_data_files = [path_dic[file] for file in path_dic
                    if experiment_root in file and "ParticleData" in file and "~lock" not in file]      

        summary_files = [path_dic[file] for file in path_dic
                    if experiment_root in file and "Summary" in file 
                    and "ExperimentSummary" not in file and "~lock" not in file]        

        experiment_summary_file =  path_dic[experiment + "-ExperimentSummary.csv"]

 
        if experiment + "-ExperimentSummary"+raw_suffix+".csv" in path_dic:
            experiment_summary_raw_file =  path_dic[experiment + "-ExperimentSummary"+raw_suffix+".csv"]
        else:
            experiment_summary_raw_file = None
            
            
        particle_data_raw_files = [file for file in particle_data_files if raw_suffix+".csv" in str(file)]
        particle_data_files = [file for file in particle_data_files if raw_suffix+".csv" not in str(file)]
 
        
        files_dic[experiment_root] = {"all_tracks_file":all_tracks_files,
                                 "particle_data_raw_file":particle_data_raw_files,
                                 "particle_data_file":particle_data_files,
                                 "summary_file":summary_files,
                                 "experiment_summary_file":experiment_summary_file,
                                 "experiment_summary_raw_file":experiment_summary_raw_file}
   
    return files_dic




            
            
            
            
            
        
        
