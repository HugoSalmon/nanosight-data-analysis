import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
  


from scipy.integrate import simpson




from scipy.stats import norm

confidence = 0.95
quantile = norm.ppf((1 + confidence)/2)




def plot_nanosight_size_distribution_replicates(concatenated_results, global_name, replicate_names, save_path_fig):
    
    fig, ax = plt.subplots(1, len(replicate_names)+1, figsize=(20,8))

    bin_centers = concatenated_results["Bin centre (nm)"].values
    
    for k, name in enumerate(replicate_names):
        

        concentration_average = concatenated_results["Concentration average "+name].values
        concentration_ste = concatenated_results["Standard error "+name].values
        ci = quantile*concentration_ste

        bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
        bins =  [0] + list(bin_centers + bin_diffs/2)
        
        start = 0
        stop = len(bin_centers)
    
        ax[k].plot(bin_centers[start:stop], concentration_average[start:stop], color="blue", label="Average concentration")
        ax[k].fill_between(x=bin_centers[start:stop], y1=concentration_average[start:stop]-ci[start:stop], y2=concentration_average[start:stop]+ci[start:stop], color='b', alpha=.3, label="95%-Confidence interval")

        ax[k].legend(fontsize=13)
        ax[k].tick_params(axis="both", labelsize=13)
        
        
        
    concentration_average = concatenated_results["Concentration average "+global_name].values
    concentration_ste = concatenated_results["Standard error "+global_name].values
    ci = quantile*concentration_ste
    ax[-1].plot(bin_centers[start:stop], concentration_average[start:stop], color="blue", label="Average concentration")
    ax[-1].fill_between(x=bin_centers[start:stop], y1=concentration_average[start:stop]-ci[start:stop], y2=concentration_average[start:stop]+ci[start:stop], color='b', alpha=.3, label="95%-Confidence interval")


    ax[-1].set_title("Average over replicates", fontsize=15)
          
    ax[-1].legend(fontsize=15)
    ax[-1].tick_params(axis="both", labelsize=15)
            
    fig.suptitle(global_name, fontsize=18)

    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig)
    




def plot_nanosight_size_densities(concatenated_results, name, save_path_fig):

    fig, ax = plt.subplots(1, figsize=(8,6))
    ax = [ax]

    bin_centers = concatenated_results["Bin centre (nm)"].values
    concentration_average = concatenated_results["Concentration average "+name].values
    concentration_ste = concatenated_results["Standard error "+name].values
    ci = quantile*concentration_ste

    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
    
    start = 0
    stop = 300


#################" ATTENTION
    area = np.trapz(concentration_average, bin_centers)
    normalized_distrib = concentration_average / area
    concentration_average = normalized_distrib
    #########

    ax[0].plot(bin_centers[start:stop], normalized_distrib[start:stop], color="blue", label="Average concentration")
    # ax[0].fill_between(x=bin_centers[start:stop], y1=concentration_average[start:stop]-ci[start:stop], y2=concentration_average[start:stop]+ci[start:stop], color='blue', alpha=0.1, label="95%-Confidence interval")

                    
    # ax[0].legend(fontsize=13)
    ax[0].tick_params(axis="both", labelsize=15)
    ax[0].set_ylabel("Density", fontsize=18)

    ax[0].set_xlabel("Size (nm)", fontsize=18)
    


    # fig.suptitle(name, fontsize=15)
    # fig.suptitle("Particles sizes distribution", fontsize=18)
    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig)


def plot_nanosight_size_distribution(concatenated_results, name, save_path_fig):


    fig, ax = plt.subplots(1, figsize=(16,10))
    ax = [ax]

    bin_centers = concatenated_results["Bin centre (nm)"].values
    concentration_average = concatenated_results["Concentration average "+name].values
    concentration_ste = concatenated_results["Standard error "+name].values
    ci = quantile*concentration_ste

    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
    
    start = 0
    stop = len(bin_centers)


    ax[0].plot(bin_centers[start:stop], concentration_average[start:stop], color="blue", label="Average concentration")
    ax[0].fill_between(x=bin_centers[start:stop], y1=concentration_average[start:stop]-ci[start:stop], y2=concentration_average[start:stop]+ci[start:stop], color='blue', alpha=0.1, label="95%-Confidence interval")

                    
    ax[0].legend(fontsize=13)
    ax[0].tick_params(axis="both", labelsize=13)
    ax[0].set_title("Distribution estimation", fontsize=15)
    ax[0].set_ylabel("Concentration (particles/mL)", fontsize=15)

    ax[0].set_xlabel("Size (nm)", fontsize=15)


    fig.suptitle(name, fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig)




def plot_all_conds_nanosight(results_table, dic_exp, list_conds, total_concentrations,
                              colors_time=None, lim_left=None, 
                              lim_right=None, savepath=None):


    fig, axes = plt.subplots(len(list_conds),2, figsize=(25,15), sharex=True, sharey=False)
            
    if len(list_conds)==1:
        ax = axes.reshape(1,-1)
    else:
        ax = axes
        
    colors = ["cyan", "dodgerblue", "darkblue", "orange", "tomato", "darkred"]    + ["cyan", "dodgerblue", "darkblue", "orange", "tomato", "darkred"]  + ["cyan", "dodgerblue", "darkblue", "orange", "tomato", "darkred"]  + ["cyan", "dodgerblue", "darkblue", "orange", "tomato", "darkred"]  

    colors += 5*colors

    bin_centers = results_table["Bin centre (nm)"].values
    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
        
    for i, key in enumerate(list_conds):
        
        for n, name in enumerate(dic_exp[key]):

            concentration = results_table["Concentration average "+name].values

            # area = np.sum(concentration * bin_diffs)
            area = simpson(x=bin_centers, y=concentration)

            normalized_concentration = concentration / area
            c_totale = total_concentrations.loc[name]["Average of Total Concentration"]
            
            # reliable = reliable_results.loc[name]["Is reliable"]
            
            if colors_time is not None:
                color = colors_time[n]
                
            else:
                color=colors[n]


            ax[i,0].plot(bin_centers, concentration, color=color, label=name + " C=%.2e"%c_totale)  
                        
            ax[i,1].plot(bin_centers, normalized_concentration, color=color, label=name+" (normalized)")  


    for i in range(len(list_conds)):
        
        if lim_left is not None:
            ax[i,0].set_xlim(left=lim_left)
        if lim_right is not None:
            ax[i,0].set_xlim(right=lim_right)

        ax[i,0].legend(fontsize=13, loc="upper right")
        ax[i,1].legend(fontsize=13, loc="upper right")

        ax[i,0].set_ylabel("Concentration (Particles/ml)", fontsize=13)
            
        ax[i,0].tick_params(labelsize=13)
        ax[i,1].tick_params(labelsize=13)

        
    lims = [ax[i,0].get_ylim()[1] for i in range(len(list_conds))]

    for i in range(len(list_conds)):
        ax[i,0].set_ylim([0,np.max(lims)])
    
    lims_normalized = [ax[i,1].get_ylim()[1] for i in range(len(list_conds))]

    for i in range(len(list_conds)):
        ax[i,1].set_ylim([0,np.max(lims_normalized)])
        
        
    ax[-1,0].set_xlabel("Diameter size (nm)", fontsize=13)
    ax[-1,1].set_xlabel("Diameter size (nm)", fontsize=13)
    
    ax[0,1].set_title("Normalized distributions", fontsize=15)

    fig.tight_layout()
    
    if savepath is not None:
        fig.savefig(savepath)  
        
    plt.close(fig)
        
        
        
        

def plot_comparison_nanosight(results_table, cond1, cond2, dic_exp, total_concentrations, reliable_results, 
                              colors_cond=None, colors_time=None, lim_left=None, 
                              lim_right=None, savepath=None, ylim_max=None, raw=False):

    fig, ax = plt.subplots(3,2, figsize=(20,10), sharex=True, sharey=False)

    bin_centers = results_table["Bin centre (nm)"].values
    bin_diffs = np.array([bin_centers[0]*2] + list(bin_centers[1:] - bin_centers[:-1]))
    bins =  [0] + list(bin_centers + bin_diffs/2)
        
    for i, key in enumerate([cond1, cond2]):

        for n, name in enumerate(dic_exp[key]):

            if raw:
                concentration = results_table["Raw Concentration average "+name].values
            else:
                concentration = results_table["Concentration average "+name].values

            area = np.sum(concentration * bin_diffs)
            normalized_concentration = concentration / area
            c_totale = total_concentrations.loc[name]["Average Concentration (Particles / ml)"]
            
            reliable = reliable_results.loc[name]["Is reliable"]

            
            if n==0:
                ax[0,0].plot(bin_centers, concentration, color=colors_cond[i], label=key)
                ax[0,1].plot(bin_centers, normalized_concentration, color=colors_cond[i], label=key+" (normalized)")
            else:
                ax[0,0].plot(bin_centers, concentration, color=colors_cond[i]) 
                ax[0,1].plot(bin_centers, normalized_concentration, color=colors_cond[i])

            ax[1+i,0].plot(bin_centers, concentration, color=colors_time[n], label=name + " C=%.2e"%c_totale + " -"+reliable)  
            ax[1+i,1].plot(bin_centers, normalized_concentration, color=colors_time[n], label=name+" (normalized)" + " -"+reliable)   

    for i in range(3):
        if lim_left is not None:
            ax[i,0].set_xlim(left=lim_left)
        if lim_right is not None:
            ax[i,0].set_xlim(right=lim_right)
            
        if ylim_max is not None:
            ax[i,0].set_ylim(top=ylim_max)

        ax[i,0].legend(fontsize=13)
        ax[i,1].legend(fontsize=13)

        if raw:
            ax[i,0].set_ylabel("Raw Concentration (Particles/ml)", fontsize=13)
        else:
            ax[i,0].set_ylabel("Concentration (Particles/ml)", fontsize=13)

        ax[i,0].tick_params(labelsize=13)
        ax[i,1].tick_params(labelsize=13)

    ax[1,0].set_ylim(ax[0,0].get_ylim())
    ax[2,0].set_ylim(ax[0,0].get_ylim())
    ax[1,1].set_ylim(ax[0,1].get_ylim())
    ax[2,1].set_ylim(ax[0,1].get_ylim())

    ax[-1,0].set_xlabel("Diameter size (nm)", fontsize=13)
    ax[-1,1].set_xlabel("Diameter size (nm)", fontsize=13)
    
    ax[0,1].set_title("Normalized distributions", fontsize=15)

    fig.tight_layout()
    
    if savepath is not None:
        fig.savefig(savepath)
