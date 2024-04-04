import glob
import numpy as np
import pandas as pd
import os
import random
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve


# DEFINE SEEDS
seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

all_reports_path = '/work/robinpark/PID010A_clean/all_osclmric_reports.csv'
test_reports_path = '/work/robinpark/AutoLabelClassifier/data/osclmric_reports'
labels_path = '/work/robinpark/AutoLabelClassifier/data/report_labels'
ivd_arrays_path = '/work/robinpark/AutoLabelClassifier/data/osclmric_ivd_arrays'

# IVD-level Labels
def import_labels(level):
    df_labels = pd.read_csv(f'{labels_path}/con_lora_base_2step_ALL_STENOSIS{level}_new_template_yesno_scores_have_prompt.csv',index_col=0).reset_index(drop=True)
    df_labels['level'] = level
    return df_labels 

print('Importing and thresholding automated labels...')
df_labels_l5s1 = import_labels('L5S1')
df_labels_l4l5 = import_labels('L4L5')
df_labels_l3l4 = import_labels('L3L4')

# Threshold scores to create binary labels
def norm_scores_yes(row):
    scores = torch.tensor([row['yes_score'], row['no_score']])
    return F.softmax(scores,dim=0)[0].item()

def norm_scores_no(row):
    scores = torch.tensor([row['yes_score'], row['no_score']])
    return F.softmax(scores,dim=0)[1].item()

df_threshold_list = [
    (df_labels_l3l4, 0.0009224715176969767),
    (df_labels_l4l5, 0.0009928893996402621), 
    (df_labels_l5s1, 0.0009309677989222109),
    ]

for df_labels, threshold in df_threshold_list:
    df_labels['yes_norm'] = df_labels.apply(lambda x: norm_scores_yes(x), axis=1)
    df_labels['no_norm'] = df_labels.apply(lambda x: norm_scores_no(x), axis=1)

    df_labels['results']=0
    df_labels.loc[df_labels.yes_norm>threshold, 'results'] = 1

# Merge with reports file to get identifiers 
df_all = pd.read_csv(all_reports_path)
df_labels_l3l4 = df_labels_l3l4.merge(df_all, left_index=True, right_index=True, how='left')
df_labels_l4l5 = df_labels_l4l5.merge(df_all, left_index=True, right_index=True, how='left')
df_labels_l5s1 = df_labels_l5s1.merge(df_all, left_index=True, right_index=True, how='left')

# Concatenate thresholded labels
df_labels = pd.concat([df_labels_l3l4, df_labels_l4l5, df_labels_l5s1]).reset_index(drop=True) 

print('Creating list of scans...')
# Create list of scans
t2_s1_scans = []
for level in ['L3_L4', 'L5_S1', 'L4_L5']: 
    t2_s1_scans_level = glob.glob(f'{ivd_arrays_path}/*/*/{level}.npy')
    t2_s1_scans.extend(t2_s1_scans_level)

print('Organising scans with same patient ID and date...')
# Create an ndarray of ndarrays with same pat_id and date
ivd_dicts = {}
for t2_s1_scan in t2_s1_scans:
    pat_id, date, level = t2_s1_scan.split('/')[-3:] # 
    date = date.split('_')[0]
    level = level.split('.')[0].replace('_', '')
    ivd_dicts[f'{pat_id}_{date}_{level}'] = np.load(t2_s1_scan) # _{level}

# Add channel dimension
for key in ivd_dicts:
    ivd_dicts[key] = ivd_dicts[key][np.newaxis, :]

print('Removing invalid reports...')
# Remove labels with invalid reports
df_labels = df_labels.loc[(df_labels.report_no_hist.str.lower().str.find('no report') == -1) & 
                          (df_labels.report_no_hist.str.lower().str.find('external source') == -1)]

# Get unique patient, date and level
df_merged = df_labels
df_merged['pat_id_date_level'] = df_merged['pat_id'] + '_' + df_merged['date'] + '_' + df_merged['level']
df_merged = df_merged.drop_duplicates(subset='pat_id_date_level').reset_index(drop=True)

list_pat_id_date_level = df_merged['pat_id_date_level'].to_list()

# Get list of test subjects and manual labels 
df_test_subj = pd.read_csv(f'{test_reports_path}/osclmric_test_subjects.csv')
df_test_labels = pd.read_csv(f'{test_reports_path}/test_manual_labels.csv')

# Create column that merges pat ID and date
df_test_labels['pat_id'] = df_test_labels['pat_id_date_level'].str.split('_').str[0]
df_test_labels['date'] = df_test_labels['pat_id_date_level'].str.split('_').str[1]
df_test_labels['pat_id_date'] = df_test_labels['pat_id'] + '_' + df_test_labels['date']

all_pat_list = list(df_merged.pat_id.drop_duplicates())
test_pat_list = list(df_test_labels['pat_id'].unique())
test_pat_date_list = list(df_test_labels['pat_id_date'].unique())
train_pat_list = list(set(all_pat_list) - set(test_pat_list))

## Split train pat list into train and val

# train_pat_list, val_pat_list = train_test_split(train_pat_list, test_size=0.1, random_state=seed)

# # create a dictionary with patient splits
# osclmric_train_val_test_split = {}
# osclmric_train_val_test_split['train'] = train_pat_list
# osclmric_train_val_test_split['val'] = val_pat_list
# osclmric_train_val_test_split['test'] = test_pat_list

# # save the splits
# with open('/work/robinpark/AutoLabelClassifier/data/osclmric_ivd_arrays/osclmric_train_val_test_split.pkl', 'wb') as handle:
#     pickle.dump(osclmric_train_val_test_split, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Import split data
with open(f'{ivd_arrays_path}/osclmric_train_val_test_split.pkl', 'rb') as handle:
    osclmric_train_val_test_split = pickle.load(handle)

train_pat_list = osclmric_train_val_test_split['train']
val_pat_list = osclmric_train_val_test_split['val']
test_pat_list = osclmric_train_val_test_split['test']

print('Creating arrays for each split of data...')
# Create arrays for each pat_id_date
ivd_train_array = []
ivd_val_array = []
ivd_test_array = []

label_train_array = []
label_train_scores = []

label_val_array = []
label_val_scores = []

label_test_array = []
compare_test_array = []
label_test_scores = []
label_test_report = []
label_test_con = []

test_reports = []
test_pat_id_date = []

for pat_id_date_level in list_pat_id_date_level:
    if pat_id_date_level in ivd_dicts:
        if pat_id_date_level[:-5] in test_pat_date_list:
            ivd_test_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_test_array.append(df_test_labels.loc[df_test_labels.pat_id_date_level==pat_id_date_level].label.item())
            compare_test_array.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].results.item())
            test_reports.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].report.item())
            label_test_scores.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].yes_norm.item())
            test_pat_id_date.append(pat_id_date_level)
            label_test_report.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].report_no_hist.item())
            label_test_con.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].pred_conclusion.item())
        elif pat_id_date_level[:6] in val_pat_list:
            ivd_val_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_val_array.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].results.item())
            label_val_scores.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].yes_norm.item())
        elif pat_id_date_level[:6] in train_pat_list:
            ivd_train_array.append(torch.Tensor(ivd_dicts[pat_id_date_level]))
            label_train_array.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].results.item())
            label_train_scores.append(df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].yes_norm.item())
    else:
        continue

osclmric_array_dict = {}
osclmric_array_dict['ivd_train_array'] = ivd_train_array
osclmric_array_dict['ivd_val_array'] = ivd_val_array
osclmric_array_dict['ivd_test_array'] = ivd_test_array

osclmric_array_dict['label_train_array'] = label_train_array
osclmric_array_dict['label_train_scores'] = label_train_scores

osclmric_array_dict['label_val_array'] = label_val_array
osclmric_array_dict['label_val_scores'] = label_val_scores

osclmric_array_dict['label_test_array'] = label_test_array
osclmric_array_dict['compare_test_array'] = compare_test_array
osclmric_array_dict['label_test_scores'] = label_test_scores
osclmric_array_dict['label_test_report'] = label_test_report
osclmric_array_dict['label_test_con'] = label_test_con
osclmric_array_dict['test_reports'] = test_reports
osclmric_array_dict['test_pat_id_date'] = test_pat_id_date

print('Saving arrays...')
# Pickle the dictionary
with open(f'{ivd_arrays_path}/osclmric_arrays_dict.pkl', 'wb') as handle:
    pickle.dump(osclmric_array_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Creating dictionaries for data summarisation...')
# Create dictionaries of samples for each split of the data (IVD as key, label as value)
train_samples = {}
val_samples = {}
test_samples = {}

for pat_id_date_level in list_pat_id_date_level:
    result = df_merged.loc[df_merged.pat_id_date_level==pat_id_date_level].results.item()
    key = pat_id_date_level
    
    if pat_id_date_level in list(ivd_dicts.keys()):
        if key[:-5] in test_pat_date_list:
            test = df_test_labels.loc[df_test_labels.pat_id_date_level==pat_id_date_level].label.item()
            test_samples[key] = test
            
        elif key[:6] in val_pat_list:
            val_samples[key] = result

        elif key[:6] in train_pat_list:
            train_samples[key] = result
    else:
        continue

samples = {}
samples['train_samples'] = train_samples
samples['val_samples'] = val_samples
samples['test_samples'] = test_samples

# Pickle the dictionary
with open(f'{ivd_arrays_path}/osclmric_samples_dict.pkl', 'wb') as handle:
    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')

