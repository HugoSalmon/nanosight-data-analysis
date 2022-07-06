

from nanosight.list_nanosight_files import list_nanosight_files_in_directory
from nanosight.read_nanosight_files import read_experiment_summary_file

from pathlib import Path
import numpy as np
import pandas
import itertools
from collections import OrderedDict



def extract_nanosight_experiment_measures(directory_path, autosampler=False, dilution_prefix="dilution", raw_suffix="_1", keep_videos=True):

    files_dic = list_nanosight_files_in_directory(directory_path, raw_suffix=raw_suffix)
    
    bin_centers = [] #for bin sizes verif
    
    sorted_name_experiments = sorted(list(files_dic.keys()))
        
    dilutions = []
    
    for i, name_experiment in enumerate(sorted_name_experiments):
        
        try:
            splitted = name_experiment.split(" ")
            dilution_string = [s for s in splitted if dilution_prefix in s][-1]
            dilution_factor = int(dilution_string.replace(dilution_prefix,""))
            # print(name_experiment, dilution_factor)
            dilutions.append(dilution_factor)

        except:
            # print(name_experiment, "Warning, no dilution factor found, set to 1")
            dilution_factor = 1
            dilutions.append("Not found")
            
                
        experiment_summary_file = files_dic[name_experiment]["experiment_summary_file"]    
        experiment_summary_table, results_concentration, results_reliable = read_experiment_summary_file(Path(directory_path, experiment_summary_file), autosampler=autosampler)
        
        bin_centers.append(experiment_summary_table["Bin centre (nm)"].values)


        for col in [col for col in experiment_summary_table.columns if col!="Bin centre (nm)"]:
            experiment_summary_table[col] = experiment_summary_table[col] * dilution_factor

        experiment_summary_table.columns = [col+" "+name_experiment if col!="Bin centre (nm)" else col for col in experiment_summary_table.columns]

        results_concentration = results_concentration * dilution_factor
                
        if not autosampler:
#            experiment_summary_table.columns = [col.replace("Concentration (particles / ml)", "Concentration " for col in experiment_summary_table.columns]
            cols_videos = [col for col in experiment_summary_table if "Video" in col]    
            u = experiment_summary_table[cols_videos].std(axis=1)
            experiment_summary_table["Standard error "+name_experiment] = u  / np.sqrt(len(cols_videos))
            experiment_summary_table["Standard deviation "+name_experiment] = u



        ### Raw data

        if files_dic[name_experiment]["experiment_summary_raw_file"] is not None:   

            raw_experiment_summary_file = files_dic[name_experiment]["experiment_summary_raw_file"]    
            raw_experiment_summary_table, data_infos, results_reliable = read_experiment_summary_file(Path(directory_path, raw_experiment_summary_file), autosampler=autosampler)
            bin_centers.append(raw_experiment_summary_table["Bin centre (nm)"].values)

            for col in [col for col in raw_experiment_summary_table.columns if col!="Bin centre (nm)"]:
                raw_experiment_summary_table[col] = raw_experiment_summary_table[col] * dilution_factor
    
            raw_experiment_summary_table.columns = ["Raw "+col+" "+name_experiment if col!="Bin centre (nm)" else col for col in raw_experiment_summary_table.columns]
            
            if not autosampler:
                cols_videos = [col for col in raw_experiment_summary_table if "Video" in col]
                u = raw_experiment_summary_table[cols_videos].std(axis=1) 
                raw_experiment_summary_table["Raw Standard error "+name_experiment] = u  / np.sqrt(len(cols_videos))
                raw_experiment_summary_table["Raw Standard deviation "+name_experiment] = u   

            raw_experiment_summary_table.drop("Bin centre (nm)", axis=1, inplace=True)
            experiment_summary_table = pandas.concat([experiment_summary_table, raw_experiment_summary_table], axis=1)                        

        if i==0:
            concatenated_results = experiment_summary_table
        else:
            experiment_summary_table.drop("Bin centre (nm)", axis=1, inplace=True)
            concatenated_results = pandas.concat([concatenated_results, experiment_summary_table], axis=1)


        results_concentration.index = [name_experiment]
        if i==0:
            
            concatenated_results_concentration = results_concentration
        else:
            concatenated_results_concentration = pandas.concat([concatenated_results_concentration, results_concentration], axis=0)

        results_reliable.index = [name_experiment]
        if i==0:
            
            concatenated_results_reliable = results_reliable
        else:
            concatenated_results_reliable = pandas.concat([concatenated_results_reliable, results_reliable], axis=0)
        
        
        
    ##verif equal bins
    for i, j in list(itertools.combinations(np.arange(len(bin_centers)), 2)):
        
        if len(bin_centers[i])!=len(bin_centers[j]):
            raise ValueError("Error: different bin sizes", data_path, sorted_name_experiments[i], sorted_name_experiments[j])
        
        if ((np.array(bin_centers[i])==np.array(bin_centers[j])).sum()) != (len(np.array(bin_centers[i]))):
            raise ValueError("Error: different bin sizes", data_path, sorted_name_experiments[i], sorted_name_experiments[j])
    
    
    if not keep_videos:
        for col in [col for col in concatenated_results.columns if "Video" in col]:
            concatenated_results.drop(col, axis=1, inplace=True)
            
    
    return sorted_name_experiments, concatenated_results, concatenated_results_concentration, dilutions, concatenated_results_reliable






    