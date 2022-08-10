

from nanosight.list_nanosight_files import list_nanosight_files_in_directory
from nanosight.read_nanosight_files import read_experiment_summary_file

from pathlib import Path
import numpy as np
import pandas
import itertools
from collections import OrderedDict



def extract_nanosight_experiment_measures(directory_path, dilution_prefix="dilution", raw_suffix="_1", keep_videos=True):

    files_dic = list_nanosight_files_in_directory(directory_path, raw_suffix=raw_suffix)
    
    bin_centers = [] #for bin sizes verif
    
    sorted_name_experiments = sorted(list(files_dic.keys()))
        
    dilutions = []
        
    for i, name_experiment in enumerate(sorted_name_experiments):

        try:
            dilution_chain = name_experiment.split(dilution_prefix)[-1]
            
            dilution_factor = ""
            
            for s in range(len(dilution_chain)):
                if dilution_chain[s].isdigit():
                    dilution_factor += dilution_chain[s]
                    
                else:
                    break


            dilution_factor = int(dilution_factor)
            
            # print(name_experiment, dilution_factor)
            dilutions.append(dilution_factor)

        except:
            # print(name_experiment, "Warning, no dilution factor found, set to 1")
            dilution_factor = 1
            dilutions.append("Not found")
            
            
                
        experiment_summary_file = files_dic[name_experiment]["experiment_summary_file"]    
        results_distributions, results_concentrations, results_reliable, results_size = read_experiment_summary_file(Path(directory_path, experiment_summary_file))
 
    
        
        bin_centers.append(results_distributions["Bin centre (nm)"].values)

        for col in [column for column in results_distributions.columns if "Bin centre" not in column]:
            results_distributions[col] = results_distributions[col] * dilution_factor  
            
        results_distributions["Standard deviation"] = results_distributions["Standard error"]*np.sqrt(5)

        results_distributions.columns = [col+" "+name_experiment if col!="Bin centre (nm)" else col for col in results_distributions.columns]

        


 
                
        # if not autosampler:
        #     cols_videos = [col for col in results_distributions if "Video" in col]    
        #     u = results_distributions[cols_videos].std(axis=1)
        #     results_distributions["Standard error "+name_experiment] = u  / np.sqrt(len(cols_videos))
        #     results_distributions["Standard deviation "+name_experiment] = u



        # ### Raw data

        # if files_dic[name_experiment]["experiment_summary_raw_file"] is not None:   

        #     raw_experiment_summary_file = files_dic[name_experiment]["experiment_summary_raw_file"]    
        #     raw_experiment_summary_table, data_infos, results_reliable = read_experiment_summary_file(Path(directory_path, raw_experiment_summary_file), autosampler=autosampler)
        #     bin_centers.append(raw_experiment_summary_table["Bin centre (nm)"].values)

        #     for col in [col for col in raw_experiment_summary_table.columns if col!="Bin centre (nm)"]:
        #         raw_experiment_summary_table[col] = raw_experiment_summary_table[col] * dilution_factor
    
        #     raw_experiment_summary_table.columns = ["Raw "+col+" "+name_experiment if col!="Bin centre (nm)" else col for col in raw_experiment_summary_table.columns]
            
        #     if not autosampler:
        #         cols_videos = [col for col in raw_experiment_summary_table if "Video" in col]
        #         u = raw_experiment_summary_table[cols_videos].std(axis=1) 
        #         raw_experiment_summary_table["Raw Standard error "+name_experiment] = u  / np.sqrt(len(cols_videos))
        #         raw_experiment_summary_table["Raw Standard deviation "+name_experiment] = u   

        #     raw_experiment_summary_table.drop("Bin centre (nm)", axis=1, inplace=True)
        #     experiment_summary_table = pandas.concat([experiment_summary_table, raw_experiment_summary_table], axis=1)                        



        results_concentrations.insert(0, "Average of Total Concentration", results_concentrations.mean(axis=1).values[0])
        results_concentrations.insert(1, "Std of Total Concentration", results_concentrations.std(axis=1).values[0])      
       

        if i==0:
            concatenated_distributions = results_distributions.copy()
        else:
            concatenated_distributions = pandas.merge(concatenated_distributions, results_distributions, on="Bin centre (nm)", how="inner")


        results_concentrations.index = [name_experiment]
        if i==0:
            
            concatenated_concentrations = results_concentrations.copy()
        else:
            concatenated_concentrations = pandas.concat([concatenated_concentrations, results_concentrations], axis=0)
            
            
        



        results_reliable.drop("key", axis=1, inplace=True)
            
        validity = pandas.DataFrame(results_reliable.iloc[[1],:])
        particles_per_frame = pandas.DataFrame(results_reliable.iloc[:1,:]).astype(float)
                
        validity.index = [name_experiment]
        particles_per_frame.index = [name_experiment]
        
        
        if i==0: 
            concatenated_particles_per_frame = particles_per_frame.copy()
        else:
            concatenated_particles_per_frame = pandas.concat([concatenated_particles_per_frame, particles_per_frame], axis=0)
        
        if i==0:
            concatenated_validity = validity
        else:
            concatenated_validity = pandas.concat([concatenated_validity, validity], axis=0)
        

        for k, index in enumerate(results_size.index):
            
            
            res = [results_size.loc[index, col] for col in results_size.columns]

            new_df = pandas.DataFrame(np.array(res).reshape(1,-1), columns = [index+" "+col for col in results_size.columns])
                        
            new_df.insert(0, "Average of "+index, np.mean(res))
            new_df.insert(1, "Std of "+index, np.std(res))


            if k==0:
                
                new_results_size = new_df
            else:
                new_results_size = pandas.concat([new_results_size, new_df], axis=1)

        new_results_size.index = [name_experiment]

        
        if i==0:
            concatenated_sizes = new_results_size.copy()
        else:
            concatenated_sizes = pandas.concat([concatenated_sizes, new_results_size], axis=0)
            
                                         
                
    ##verif equal bins
    for i, j in list(itertools.combinations(np.arange(len(bin_centers)), 2)):
        
        if len(bin_centers[i])!=len(bin_centers[j]):
            raise ValueError("Error: different bin sizes", data_path, sorted_name_experiments[i], sorted_name_experiments[j])
        
        if ((np.array(bin_centers[i])==np.array(bin_centers[j])).sum()) != (len(np.array(bin_centers[i]))):
            raise ValueError("Error: different bin sizes", data_path, sorted_name_experiments[i], sorted_name_experiments[j])
    
    
    if not keep_videos:
        for col in [col for col in concatenated_results_distributions.columns if "Video" in col]:
            concatenated_results_distributions.drop(col, axis=1, inplace=True)
            
            
            
    
    return sorted_name_experiments, concatenated_distributions, concatenated_concentrations, dilutions, concatenated_particles_per_frame, concatenated_validity, concatenated_sizes






    