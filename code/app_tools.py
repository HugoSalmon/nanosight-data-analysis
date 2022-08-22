import os
from pathlib import Path
from collections import OrderedDict


def get_replicates(name_experiments, replicate_prefix="-"):
    
    replicates = OrderedDict({})

    for i, name_experiment in enumerate(name_experiments):

        is_replicate = replicate_prefix in name_experiment
                
        if is_replicate:

            replicate_chain = name_experiment.split(" ")[-1]

            str_num = replicate_chain.split(replicate_prefix)[-1]

            shorted_name = name_experiment.replace(replicate_prefix+str_num,"")
            
            if shorted_name not in replicates:
                replicates[shorted_name] = [name_experiment]
            else:
                replicates[shorted_name].append(name_experiment)
                
        else:
            replicates[name_experiment] = [name_experiment]

                          
    return replicates



def create_missing_dir(list_sequential_dirs):
    
    for i in range(len(list_sequential_dirs)):
        
        path = Path(*list_sequential_dirs[:i+1])

        if not os.path.exists(path):
            os.mkdir(Path(path))
