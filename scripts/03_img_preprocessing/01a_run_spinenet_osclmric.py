import sys
sys.path.append('/work/robinpark/SpineNet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import spinenet
import glob 
import os

from tqdm import tqdm
from spinenet import SpineNet
from spinenet.io import load_dicoms_from_folder, save_vert_dicts_to_csv

scan_folder = '/datasets/SpineDatasets/MRI/OSCLMRIC/Scans'

spnt = SpineNet(device='cuda:0', verbose=True)

# Get all T2 sagittal scans 
t2_s1_scans = glob.glob('/datasets/SpineDatasets/MRI/OSCLMRIC/Scans/*/*/T2/S1') 

# Save out IVD volumes
for folder in t2_s1_scans:
    p = pathlib.Path(folder)
    pat_id, date, t2, s1 = p.parts[-4:]
    try:
        scan = load_dicoms_from_folder(
            folder,
            require_extensions=False
        )
        vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
        ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)
        print(f'Executing {pat_id}_{date}...{len(vert_dicts)} vertebrae detected; {[vert_dict["predicted_label"] for vert_dict in vert_dicts]}')
        for i, ivd_dict in enumerate(ivd_dicts):
            ivd_level = ivd_dict['level_name'].replace('-','_')
            ivd_volume = ivd_dict['volume']
            ivd_array_path = f"/work/robinpark/AutoLabelClassifier/data/osclmric_ivd_arrays/{pat_id}"
            pat_subfolder = f"{ivd_array_path}/{date}_{t2}_{s1}"
            # create directory if it doesn't exist
            if not os.path.exists(ivd_array_path):
                os.mkdir(ivd_array_path)
            if not os.path.exists(pat_subfolder):
                os.mkdir(pat_subfolder)
            np.save(f"{pat_subfolder}/{ivd_level}.npy", ivd_volume)
    except:
        print(f'Executing {pat_id}_{date}...no vertebrae detected; nothing saved out')