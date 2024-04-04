import pandas as pd

df_orig = pd.read_csv('/work/robinpark/PID010A_clean/OSCLMRIC_reports_any_stenosis_edited.csv')
df_no_sten = pd.read_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_no_stenosis.csv')

df_orig = df_orig.rename(columns={'Report':'report'})
df_no_sten = df_no_sten.loc[~df_no_sten.no_stenosis.isna()]
df_no_sten['no_sten'] = 0
df_no_sten = df_no_sten.rename(columns={'no_sten':'result'})

df_merge = pd.concat([df_orig, df_no_sten])

# Shuffle the rows
df_merge = df_merge.sample(frac=1, random_state=0).reset_index(drop=True)

# Split into two dataframes
df_train = df_merge.iloc[:int(len(df_merge)*0.5)]
df_test = df_merge.iloc[int(len(df_merge)*0.5):]

print(df_train.value_counts('result'))
print(df_test.value_counts('result'))

df_train.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/osclmric_reports_labeled_train.csv', index=False)
df_test.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/osclmric_reports_labeled_test.csv', index=False)
