import pandas as pd

root_dir = '/work/robinpark/AutoLabelClassifier/data/ncimi_reports'

# Import data
df_test = pd.read_csv(f'{root_dir}/segmented_test_manually_labeled_set.csv',index_col=0)
df_train = pd.read_csv(f'{root_dir}/segmented_train_manually_labeled_set.csv',index_col=0)

# Drop outdated field
df_test = df_test.drop(columns=['study_id_coded'])
df_train = df_train.drop(columns=['study_id_coded'])

# Import updated study IDs
df_unique_reports = pd.read_csv('/work/robinpark/NCIMI_clean/segmented_unique_reports.csv', index_col=0, low_memory=False)
df_unique_reports = df_unique_reports[['study_id_coded','report_no_hist']]

# Merge, excluding any unmatched
df_test = df_test.merge(df_unique_reports, on='report_no_hist')
df_train = df_train.merge(df_unique_reports, on='report_no_hist')

# Drop any duplicated
df_test = df_test.drop_duplicates(subset=['report_no_hist'])
df_train = df_train.drop_duplicates(subset=['report_no_hist'])

# Drop any in each other 
df_test = df_test.drop(df_test[df_test.report_no_hist.isin(df_train.report_no_hist)].index)
df_train = df_train.drop(df_train[df_train.report_no_hist.isin(df_test.report_no_hist)].index)

# Pick 150 examples from each dataframe, stratified by cancer_in_image
df_test_cancer = df_test[df_test.cancer_in_image == 1].sample(75, random_state=0)
df_test_no_cancer = df_test[df_test.cancer_in_image == 0].sample(75, random_state=0)
df_test_sample = pd.concat([df_test_cancer, df_test_no_cancer])
df_test_sample = df_test_sample[['study_id_coded','centre','report_no_hist']]

df_train_cancer = df_train[df_train.cancer_in_image == 1].sample(75, random_state=0)
df_train_no_cancer = df_train[df_train.cancer_in_image == 0].sample(75, random_state=0)
df_train_sample = pd.concat([df_train_cancer, df_train_no_cancer])
df_train_sample = df_train_sample[['study_id_coded','centre','report_no_hist']]

df_test_sample.to_csv(f'{root_dir}/ncimi_sampled_test_set.csv')
df_train_sample.to_csv(f'{root_dir}/ncimi_sampled_train_set.csv')

