import glob
import numpy as np
import pandas as pd
import pydicom 
import pickle
import random
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split

#  Import SpineNet
import sys
sys.path.append('/work/robinpark/SpineNet')

# SEED
seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


cln_path = '/work/robinpark/PID010A_clean'
metadata_path = f'{cln_path}/patient_metadata.csv'
unique_reports_path = f'{cln_path}/segmented_unique_reports.csv'
labels_path = '/work/robinpark/AutoLabelClassifier/data/report_labels'
ivd_arrays_path = '/work/robinpark/AutoLabelClassifier/data/ncimi_ivd_arrays'


# Import extracted labels 
filepath = f"{labels_path}/con_lora_base_2step_ALL_CANCER_new_template_yesno_scores_have_prompt"
df_labels1 = pd.read_csv(f'{filepath}1.csv',index_col=0)
df_labels2 = pd.read_csv(f'{filepath}2.csv',index_col=0)
df_labels3 = pd.read_csv(f'{filepath}3.csv',index_col=0)
df_labels4 = pd.read_csv(f'{filepath}4.csv',index_col=0)

df_labels = pd.concat([df_labels1, df_labels2, df_labels3, df_labels4]).reset_index(drop=True) 

print('Importing and thresholding automated labels...')
# Threshold based on normalised scores
def norm_scores_yes(row):
    scores = torch.tensor([row['yes_score'], row['no_score']])
    return F.softmax(scores,dim=0)[0].item()

def norm_scores_no(row):
    scores = torch.tensor([row['yes_score'], row['no_score']])
    return F.softmax(scores,dim=0)[1].item()
    
df_labels['yes_norm'] = df_labels.apply(lambda x: norm_scores_yes(x), axis=1)
df_labels['no_norm'] = df_labels.apply(lambda x: norm_scores_no(x), axis=1)

df_labels['results']=0
df_labels.loc[df_labels.yes_norm>0.0005798592464998364, 'results'] = 1

# Get unique reports
df_all = pd.read_csv(unique_reports_path)
df_merged = df_labels.merge(df_all[['report_no_hist','study_id_coded']], on='report_no_hist', how='left')
df_merged = df_merged.drop_duplicates(subset='study_id_coded', keep='first')

print('Creating list of scans...')
# Get list of all scans
scans = glob.glob(f'{ivd_arrays_path}/*/*/*.npy')
df_paths = pd.DataFrame(scans,columns=['path_long'])

# Split column path into multiple columns
df_paths['path'] = df_paths['path_long'].str.slice(58,)

# Expand path to three columns
df_paths['pat_id'] = df_paths['path'].str.slice(0,7)
df_paths['study_id'] = df_paths['path'].str.slice(8,15)
df_paths['ser_id'] = df_paths['path'].str.slice(16,24)
df_paths['level'] = df_paths['path'].str.slice(25,).str.replace('.npy','')

# Sort by pat_id and study_id and keep the last two ser_id in the group
df_subset = df_paths[['pat_id','study_id','ser_id']].drop_duplicates().sort_values(
    by=['pat_id','study_id','ser_id']).groupby(['pat_id','study_id']).tail(2)

# Only keep pat_id and study_ids in df_subset (last two series in group)
df_paths_subset = df_paths.merge(df_subset, on=['pat_id','study_id','ser_id'], how='inner')

# Define list of vertebrae to keep (currently thoracic and lumbar)
list_vert = list(df_paths_subset.level.unique())
list_vert_keep = [x for x in list_vert if 'L' in x or 'T' in x]

# Import metadata
df_metadata = pd.read_csv(metadata_path, index_col=0, low_memory=False)

# Keep T2 vertebrae that are not localisers 
df_t2 = df_metadata.loc[(df_metadata.series_desc.str.find('t2')>-1) & (df_metadata.main_direction=='sagittal')]
df_t2_stu = df_t2[['pat_id_coded','study_id_coded','ser_id_coded']].drop_duplicates().reset_index(drop=True)

df_paths_subset_t2 = df_paths_subset.merge(
    df_t2_stu, left_on=['pat_id','study_id','ser_id'], 
    right_on=['pat_id_coded','study_id_coded','ser_id_coded'], how='inner') 

# Only keep levels in list
df_paths_subset_t2 = df_paths_subset_t2.loc[df_paths_subset_t2.level.isin(list_vert_keep)]

# Get unique levels
df_keys = df_paths_subset_t2[['pat_id','study_id','ser_id','level']].drop_duplicates()

print('Organising scans with same patient ID and date...')
# Create an ndarray of ndarrays with same pat_id and date
ivd_dicts = {}
for scan in list(df_paths_subset_t2.path_long):
    pat_id, stu_ser, level = scan.split('/')[-3:] # 
    # stu, ser = stu_ser.split('_')
    ivd_dicts[f'{pat_id}_{stu_ser}_{level}'] = np.load(scan) # _{level}

# Get unique patient/study ID and create merged variable
df_keys = df_paths[['pat_id','study_id']].drop_duplicates().reset_index(drop=True)
df_keys['pat_stu_id'] = df_keys['pat_id'] + '_' + df_keys['study_id']

print('Removing invalid reports...')
# Remove cases that have invalid reports
df_merged = df_merged.loc[(df_merged.report_no_hist.str.lower().str.find('no report') == -1) & 
                          (df_merged.report_no_hist.str.lower().str.find('external source') == -1)]

# Get manually annotated labels
df_test_subj1 = pd.read_csv(f'{cln_path}/segmented_test_manually_labeled_set.csv',index_col=0)
df_test_subj2 = pd.read_csv(f'{cln_path}/segmented_train_manually_labeled_set.csv',index_col=0)
df_test_subj = pd.concat([df_test_subj1, df_test_subj2]).reset_index(drop=True)
df_test_subj = df_test_subj.loc[df_test_subj.cancer_in_image != -1].reset_index(drop=True)
df_test_subj = df_test_subj.merge(df_keys, 
                                  left_on='study_id_coded', 
                                  right_on='study_id', 
                                  how='inner').drop_duplicates().reset_index(drop=True)

df_test_subj['pat_stu_id'] = df_test_subj['pat_id'] + '_' + df_test_subj['study_id']

df_merged2 = df_merged.merge(df_keys, left_on='study_id_coded', right_on='study_id', how='left')

df_merged2 = df_merged2.loc[~df_merged2.pat_id.isna()]
df_merged2['pat_stu_id'] = df_merged2['pat_id'] + '_' + df_merged2['study_id']
df_merged2 = df_merged2.drop_duplicates().reset_index(drop=True)

# Define splits 
all_pat_list = list(df_merged2.pat_id.drop_duplicates())
test_pat_list = list(df_test_subj['pat_id'].unique())
test_pat_stu_list = list(df_test_subj['pat_stu_id'].unique())
train_pat_list = list(set(all_pat_list) - set(test_pat_list))

list_pat_id_date_level = list(ivd_dicts.keys())

# Split train pat list into train and val
# train_pat_list, val_pat_list = train_test_split(train_pat_list, test_size=0.2, random_state=seed)

# # Create a dictionary with patient splits
# ncimi_train_val_test_split = {}
# ncimi_train_val_test_split['train'] = train_pat_list
# ncimi_train_val_test_split['val'] = val_pat_list
# ncimi_train_val_test_split['test'] = test_pat_list

# # Save the splits
# with open(f'{ivd_arrays_path}/ncimi_train_val_test_split.pkl', 'wb') as handle:
#     pickle.dump(ncimi_train_val_test_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{ivd_arrays_path}/ncimi_train_val_test_split.pkl', 'rb') as handle:
    ncimi_train_val_test_split = pickle.load(handle)

train_pat_list = ncimi_train_val_test_split['train']
val_pat_list = ncimi_train_val_test_split['val']
test_pat_list = ncimi_train_val_test_split['test']

train_pat_stu_list = [i for i in df_merged2.pat_stu_id if i[:7] in train_pat_list]
val_pat_stu_list = [i for i in df_merged2.pat_stu_id if i[:7] in val_pat_list]
all_pat_stu_list = test_pat_stu_list + train_pat_stu_list + val_pat_stu_list

print('Creating arrays for each split of data...')
# Create arrays for each pat_id_date: one for labels and one for ivd arrays
ivd_train_array = []
ivd_val_array = []
ivd_test_array = []

label_train_array = []
label_train_scores = []
label_train_pat_stu = []

label_val_array = []
label_val_scores = []
label_val_pat_stu = []

label_test_array = []
compare_test_array = []
label_test_scores = []
label_test_pat_stu = []
label_test_report = []
label_test_con = []

test_reports = []
test_pat_id_date = []
val_pat_id_date = []
train_pat_id_date = []
ivd_test_array_names = []

for pat_id_date_level in list_pat_id_date_level:
    pat_stu_id = pat_id_date_level[:15]
    pat_stu_ser_id = pat_id_date_level[:24]       
    if pat_stu_id in all_pat_stu_list:
        if pat_stu_id in test_pat_stu_list:
            ivd_test_array_names.append(pat_id_date_level)
            ivd_test_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_test_array.append(df_test_subj.loc[df_test_subj.pat_stu_id==pat_stu_id].cancer_in_image.item())
            compare_test_array.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].results.item())
            label_test_scores.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].yes_norm.item())
            test_pat_id_date.append(pat_stu_ser_id)
            label_test_report.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].report_no_hist.item())
            label_test_con.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].pred_conclusion.item())
            label_test_pat_stu.append(pat_stu_id)
        elif pat_stu_id in val_pat_stu_list:
            ivd_val_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_val_array.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].results.item())
            label_val_scores.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].yes_norm.item())
            val_pat_id_date.append(pat_stu_ser_id)
            label_val_pat_stu.append(pat_stu_id)
        elif pat_stu_id in train_pat_stu_list:
            ivd_train_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_train_array.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].results.item())
            label_train_scores.append(df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].yes_norm.item())
            train_pat_id_date.append(pat_stu_ser_id)
            label_train_pat_stu.append(pat_stu_id)
    else:
        continue

print('Creating dictionaries for data summarisation...')
# Save the arrays
ncimi_array_dict = {}
ncimi_array_dict['ivd_train_array'] = ivd_train_array
ncimi_array_dict['ivd_val_array'] = ivd_val_array
ncimi_array_dict['ivd_test_array'] = ivd_test_array

ncimi_array_dict['label_train_array'] = label_train_array
ncimi_array_dict['label_train_scores'] = label_train_scores
ncimi_array_dict['label_train_pat_stu'] = label_train_pat_stu

ncimi_array_dict['label_val_array'] = label_val_array
ncimi_array_dict['label_val_scores'] = label_val_scores
ncimi_array_dict['label_val_pat_stu'] = label_val_pat_stu

ncimi_array_dict['label_test_array'] = label_test_array
ncimi_array_dict['compare_test_array'] = compare_test_array
ncimi_array_dict['label_test_scores'] = label_test_scores
ncimi_array_dict['label_test_pat_stu'] = label_test_pat_stu
ncimi_array_dict['label_test_report'] = label_test_report
ncimi_array_dict['label_test_con'] = label_test_con

ncimi_array_dict['test_reports'] = test_reports
ncimi_array_dict['test_pat_id_date'] = test_pat_id_date
ncimi_array_dict['val_pat_id_date'] = val_pat_id_date
ncimi_array_dict['train_pat_id_date'] = train_pat_id_date
ncimi_array_dict['ivd_test_array_names'] = ivd_test_array_names

print('Saving arrays...')
# Pickle the dictionary
with open(f'{ivd_arrays_path}/ncimi_arrays_dict.pkl', 'wb') as handle:
    pickle.dump(ncimi_array_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create arrays for each pat_id_date: one for labels and one for ivd arrays
train_samples = {}
val_samples = {}
test_samples = {}

for pat_id_date_level in list_pat_id_date_level:
    pat_stu_id = pat_id_date_level[:15]
    if pat_stu_id in all_pat_stu_list:
        result = df_merged2.loc[df_merged2.pat_stu_id==pat_stu_id].results.item()
        key = pat_id_date_level
        
        if key in list(ivd_dicts.keys()):
            if key[:15] in test_pat_stu_list:
                test = df_test_subj.loc[df_test_subj.pat_stu_id==pat_stu_id].cancer_in_image.item()
                test_samples[key] = test
                
            elif key[:15] in val_pat_stu_list:
                val_samples[key] = result

            else:
                train_samples[key] = result
        else:
            continue

samples = {}
samples['train_samples'] = train_samples
samples['val_samples'] = val_samples
samples['test_samples'] = test_samples

# Pickle the dictionary
with open(f'{ivd_arrays_path}/ncimi_samples_dict.pkl', 'wb') as handle:
    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')