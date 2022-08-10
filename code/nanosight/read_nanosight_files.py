

import os
import numpy as np
import pandas



def read_experiment_summary_file(filepath):
    
    
    """
    Read an ExperimentSummary.csv file
        
        Parameters
        ----------
        filepath: path of the file
    
        Returns
        ----------
        a Pandas dataframe containing experiment concentration data
        
    """

    if not os.path.exists(filepath):
        raise ValueError("File not found", filepath)

    
    try:

        from csv import reader
        # open file in read mode
        with open(filepath, 'r', encoding="ISO-8859-1") as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            rows = []
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                rows.append(row)

        rows = np.array(["-".join(row) for row in rows]).astype(str)
        
        rows_series=  pandas.Series(rows)
        
        """ Read concentration distributions """

        is_start = rows_series.apply(lambda s: True if s=="Graph Data-" else False)
        
        start_index = np.where(is_start)[0][0]

        results_distributions = pandas.read_csv(filepath, skiprows=start_index+1, sep=",", encoding = "ISO-8859-1", 
                                header=0)
            
        for col in [column for column in results_distributions.columns if "Unnamed:" in column]:
            results_distributions.drop(col, axis=1, inplace=True)


        for i in range(len(results_distributions)):

            
            if str(results_distributions.iloc[i]["Bin centre (nm)"])=="nan" or str(results_distributions.iloc[i]["Bin centre (nm)"])=="Percentile":
                stop_index = i
                
                break

        results_distributions = results_distributions.iloc[:stop_index]

        results_distributions = results_distributions.astype(float)

        ### Rename columns  
        cols_videos = [col for col in results_distributions.columns if "Concentration (particles / ml)" in col] 
        dic_rename = {col: "Concentration (particles / ml) Video " + str(int(col.split(".")[1])+1) if "." in col else "Concentration (particles / ml) Video 1" for col in cols_videos}
        dic_rename.update({"Standard Error": "Standard error"})
        results_distributions.rename(columns = dic_rename, inplace=True)
        
        cols_videos = [col for col in results_distributions.columns if "Video" in col]
        
        ### Add concentration average if does not exist (for autosampler files)

        if "Concentration average" not in results_distributions.columns:

            results_distributions["Concentration average"] = results_distributions[cols_videos].mean(axis=1)
            results_distributions["Standard error"] = results_distributions[cols_videos].std(axis=1) / np.sqrt(len(cols_videos))

        results_distributions = results_distributions[["Bin centre (nm)", "Concentration average", "Standard error"] + cols_videos]
        
        ### Get other results

        details_experiment = pandas.read_csv(filepath, sep=",", usecols=range(2), encoding = "ISO-8859-1")            
            
        is_start_results = rows_series.apply(lambda s: True if "[Results]" in s  else False)
        start_results_index = np.where(is_start_results)[0][0]
        results = pandas.read_csv(filepath, sep=",", skiprows=start_results_index, usecols=range(6), encoding = "ISO-8859-1")   
        results.columns = ["key", "Video 1", "Video 2", "Video 3", "Video 4", "Video 5"]
        
        where_key = (results["key"]=="Concentration (Particles / ml)")
        results_concentration = pandas.DataFrame(results[where_key])
        results_concentration.reset_index(inplace=True)
        results_concentration.drop("index", axis=1, inplace=True)
        results_concentration.drop("key", axis=1, inplace=True)
        results_concentration = results_concentration.astype(float)

        results_concentration.rename(columns={"Video "+str(k): "Total Concentration Video "+str(k) for k in range(1, 6)}, inplace=True)
        
        name_details = np.array(results)[:,0]
     
        where_particles_per_frame = (name_details == "Particles per frame")
        
        infos_particles_per_frame = results[where_particles_per_frame].values[0]
        
        where_validity = (name_details == "Validity of concentration measurement")
        infos_validity = results[where_validity].values[0]
           
        where_key = ((results["key"]=="Validity of concentration measurement") | (results["key"]=="Particles per frame"))
        results_reliable = pandas.DataFrame(results[where_key].iloc[:2,:])
        results_reliable.reset_index(inplace=True, drop=True)
    
        # index_size = np.where(results["key"]=="[Size Data]")[0][0]
        # size_results = results[index_size:]
        
        index_size = np.where(results["key"]=="[Size Data]")[0][0]
        index_end_size = np.where(results["key"]=="Graph Data")[0][0]
    
        size_results = results[index_size:index_end_size]
        where_key = ((size_results["key"]=="Mean") | (size_results["key"]=="Mode") | (size_results["key"]=="SD") | (size_results["key"]=="D10") | (size_results["key"]=="D50") | (size_results["key"]=="D90"))
    
        results_size = pandas.DataFrame(size_results[where_key])
        index = results_size["key"]
        results_size.drop("key", axis=1, inplace=True)
            
        results_size.index = np.array(index)
        results_size = results_size.astype(float)
                
        results_size.index = ["Size "+name for name in results_size.index]

    except:
        raise ValueError("Error when reading file", filepath)

    return results_distributions, results_concentration, results_reliable, results_size





def read_summary_file(filepath):
    
    """
        Read an Summary.csv file
        
        Parameters
        ----------
        filepath: path of the file
    
        Returns
        ----------
        a Pandas dataframe containing concentration data for one specific video
        
    """    

    if not os.path.exists(filepath):
        raise ValueError("File not found", filepath)
    
    try:
        table = pandas.read_csv(filepath, skiprows=92, sep=",", encoding = "ISO-8859-1", 
                            header=0, usecols=range(2)) 
        details_experiment = pandas.read_csv(filepath, sep=",", usecols=range(2), encoding = "ISO-8859-1")

    except:
        raise ValueError("Error when reading file", filepath)
        
    infos_experiments = {}

    for i in range(len(details_experiment)):
        
        for key in ["Total frams analysed", "Mean", "Temperature/C", "Viscosity/cP", "Frame rate/fps", "Concentration (Particles / ml)"]:
            if details_experiment.iloc[i,0] == key:
                infos_experiments[key] = float(details_experiment.iloc[i,1])

        for key in ["Particles per frame"]:
            if details_experiment.iloc[i,0] == key:
                infos_experiments[key] = details_experiment.iloc[i,1]

        # print(details_experiment.iloc[i])  
        
        if i>100:
            break
                        
    
    for i in range(len(table)):
        
        if table.iloc[i]["Bin centre (nm)"]=="Percentile":
            
            table = table.iloc[:i]
            break

    table = table.astype(float)

    return table, infos_experiments


def read_particle_data_file(filepath):
    
    """
        Read an ParticleData.csv file
        
        Parameters
        ----------
        filepath: path of the file
    
        Returns
        ----------
        a Pandas dataframe containing info of tracked particles for one specific video
    
    """    

    if not os.path.exists(filepath):
        raise ValueError("File not found", filepath)
    
    try:
        table = pandas.read_csv(filepath, sep=",", encoding = "ISO-8859-1", 
                            header=0)
        
    except:
        raise ValueError("Error when reading file", filepath)

    return table
    
def read_all_tracks_file(filepath):
    
    """
        Read an AllTracks.csv file
        
        Parameters
        ----------
        filepath: path of the file
    
        Returns
        ----------
        a Pandas dataframe containing info of tracked particles for one specific video
    
    """    
    
    if not os.path.exists(filepath):
        raise ValueError("File not found", filepath)
    
    try:
        table = pandas.read_csv(filepath, sep=",", encoding = "ISO-8859-1", 
                            header=0)
        
    except:
        raise ValueError("Error when reading file", filepath)

    return table
