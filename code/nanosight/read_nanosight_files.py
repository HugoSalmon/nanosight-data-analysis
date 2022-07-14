

import os
import numpy as np
import pandas



def read_experiment_summary_file(filepath, autosampler=False):
    
    
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

        rows = np.array(["-".join(row) for row in rows[:300]]).astype(str)
        rows_series=  pandas.Series(rows)
        
        is_start = rows_series.apply(lambda s: True if "Bin centre (nm)" in s  else False)
    
        start_index = np.where(is_start)[0][0]

        is_start_results = rows_series.apply(lambda s: True if "[Results]" in s  else False)

        start_results_index = np.where(is_start_results)[0][0]
    
        if not autosampler:
    
            table = pandas.read_csv(filepath, skiprows=start_index, sep=",", encoding = "ISO-8859-1", 
                                header=0, usecols=range(8)) 
    
            details_experiment = pandas.read_csv(filepath, sep=",", usecols=range(2), encoding = "ISO-8859-1")
             
            results = pandas.read_csv(filepath, sep=",", skiprows=start_results_index, usecols=range(8), encoding = "ISO-8859-1")   
 
        else:
            
            table = pandas.read_csv(filepath, skiprows=start_index, sep=",", encoding = "ISO-8859-1", 
                                header=0, usecols=range(3)) 

            details_experiment = pandas.read_csv(filepath, sep=",", usecols=range(2), encoding = "ISO-8859-1")

        results = pandas.read_csv(filepath, sep=",", skiprows=start_results_index, usecols=range(6), encoding = "ISO-8859-1")   
        results.columns = ["key", "Video 1", "Video 2", "Video 3", "Video 4", "Video 5"]
        
        where_key = (results["key"]=="Concentration (Particles / ml)")
        results_concentration = pandas.DataFrame(results[where_key])
        results_concentration.reset_index(inplace=True)
        results_concentration.drop("index", axis=1, inplace=True)
        results_concentration.drop("key", axis=1, inplace=True)
        results_concentration = results_concentration.astype(float)
        

        concentration_average = results_concentration.mean(axis=1).values[0]
        concentration_std = results_concentration.std(axis=1).values[0]
        results_concentration["Average Concentration (Particles / ml)"] = concentration_average
        
        results_concentration["Standard Deviation"] = concentration_std         
        
    except:
        raise ValueError("Error when reading file", filepath)
        

    for i in range(len(table)):
    
        
        if str(table.iloc[i]["Bin centre (nm)"])=="Percentile" or str(table.iloc[i]["Bin centre (nm)"])=="nan":
            
            table = table.iloc[:i]
            break

    table = table.astype(float)

    ### Rename columns  
    cols_videos = [col for col in table.columns if "Concentration (particles / ml)" in col]
    table = table[["Bin centre (nm)", "Concentration average", "Standard Error"] + cols_videos]
    new_names_cols_videos = ["Concentration (particles / ml) Video "+str(j+1) for j in range(len(cols_videos))]
    table.columns = ["Bin centre (nm)", "Concentration average", "Standard error"] + new_names_cols_videos 
    
    name_details = np.array(results)[:,0]
 
    where_particles_per_frame = (name_details == "Particles per frame")
    
    infos_particles_per_frame = results[where_particles_per_frame].values[0]
    
    where_validity = (name_details == "Validity of concentration measurement")
    infos_validity = results[where_validity].values[0]

  
        
    where_key = ((results["key"]=="Validity of concentration measurement") | (results["key"]=="Particles per frame"))
    results_reliable = pandas.DataFrame(results[where_key].iloc[:2,:])
    results_reliable.reset_index(inplace=True)
    results_reliable.drop("index", axis=1, inplace=True)
    results_reliable.drop("key", axis=1, inplace=True)
        
    infos_validity = results_reliable.iloc[1,:].values


    results_reliable = pandas.DataFrame(results_reliable.iloc[:1,:]).astype(float)
    
    if np.sum(results_reliable.iloc[0,:]<30) > 0:
        reliable = "unreliable"
    else:
        reliable = "reliable"

    results_reliable["Average particles per frame"] = results_reliable.mean(axis=1)
    results_reliable.loc[:,"Is reliable"] = reliable

    return table, results_concentration, results_reliable




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
