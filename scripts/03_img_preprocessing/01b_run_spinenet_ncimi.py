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

scan_folder = '/work/robinpark/NCIMI_clean/paired_scans_reports'

spnt = SpineNet(device='cuda:0', verbose=True)

# Get sagittal series 
df_metadata = pd.read_csv('/work/robinpark/NCIMI_clean/patient_metadata.csv', index_col=0, low_memory=False)

df_t2_sag = df_metadata.loc[
    ((df_metadata.protocol_name.str.find('t2') > -1) & (df_metadata.protocol_name.str.find('sag') > -1)) |
    ((df_metadata.series_desc.str.find('t2') > -1) & (df_metadata.series_desc.str.find('sag') > -1))]

dedup_series = df_t2_sag[['ser_id_coded','main_direction']].drop_duplicates().drop_duplicates(
    subset='ser_id_coded', keep=False)

sagittal_series = dedup_series.loc[dedup_series.main_direction=='sagittal']

df_sagittal = df_metadata.loc[df_metadata.ser_id_coded.isin(sagittal_series['ser_id_coded'])]

# Limit to sagittal volumes with less than 30 slices
df_slice_counts = df_sagittal[['ser_id_coded','filename']].groupby('ser_id_coded').count()
df_slice_cln = df_slice_counts.loc[(df_slice_counts.filename < 30)]

df_sagittal_cln = df_sagittal.loc[df_sagittal.ser_id_coded.isin(df_slice_cln.index.to_list())]

df_unique_sag_ser = df_sagittal_cln[['pat_id_coded',
                                     'study_id_coded',
                                     'ser_id_coded']].drop_duplicates().reset_index(drop=True)

# Save out IVDs
for index, row in tqdm(df_unique_sag_ser.iterrows(), total=df_unique_sag_ser.shape[0]):
    pat_id = row['pat_id_coded']
    stu_id = row['study_id_coded']
    ser_id = row['ser_id_coded']

    folder = f"{scan_folder}/{pat_id}/{stu_id}/{ser_id}"

    try: 
        scan = load_dicoms_from_folder(
            folder, 
            require_extensions=False) 
        vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
        ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)

        print(f'Executing {ser_id}...{len(vert_dicts)} vertebrae detected; {[vert_dict["predicted_label"] for vert_dict in vert_dicts]}')

        # save_vert_dicts_to_csv(vert_dicts, f'{folder}/{ser_id}_vert_dicts.csv')
        for i, ivd_dict in enumerate(ivd_dicts):
            ivd_level = ivd_dict['level_name'].replace('-','_')
            ivd_volume = ivd_dict['volume']
            ivd_array_path = f"/work/robinpark/NCIMI_clean/ncimi_ivd_arrays/{pat_id}"
            pat_subfolder = f"{ivd_array_path}/{stu_id}_{ser_id}"
            if not os.path.exists(ivd_array_path):
                os.mkdir(ivd_array_path)
            if not os.path.exists(pat_subfolder):
                os.mkdir(pat_subfolder)
            np.save(f"{pat_subfolder}/{ivd_level}.npy", ivd_volume)
    except:
        print(f'Executing {ser_id}...no vertebrae detected; nothing saved out')