




from pathlib import Path
import os





source = Path().resolve().parent




for dir_ in ["nanosight-app-results"]:
    if not os.path.exists(Path(source, dir_)):
        print("No " + dir_ + " directory in the nanosight-videodrop-data-analysis directory." \
                        " Creating one.")
    
        os.mkdir(Path(source, dir_))


datapath = Path(source, "data")
codepath = Path(source, "code")
nanosight_app_path_results = Path(source, "nanosight_app_results")
