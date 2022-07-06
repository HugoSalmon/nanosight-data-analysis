




from pathlib import Path
import os





source = Path().resolve().parent




for dir_ in ["figures", "results", "data"]:
    if not os.path.exists(Path(source, dir_)):
        print("No " + dir_ + " directory in the nanosight-videodrop-data-analysis directory." \
                        " Creating one.")
    
        os.mkdir(Path(source, dir_))


datapath = Path(source, "data")
savepath = Path(source, "save")
figpath = Path(source, "figures") 
codepath = Path(source, "code")
resultspath = Path(source, "results")
nanosight_app_path_results = Path(source, "nanosight_app_results")
videodrop_app_path_results = Path(source, "videodrop_app_results")
