import pandas as pd

df_spon = pd.read_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_spon.csv')

df_spon = df_spon.loc[df_spon.global_label != -1]

# Shuffle the rows
df_spon = df_spon.sample(frac=1, random_state=0).reset_index(drop=True)

# Split into two dataframes
df_train = df_spon.iloc[:int(len(df_spon)*0.5)]
df_test = df_spon.iloc[int(len(df_spon)*0.5):]

print(df_train.value_counts('global_label'))
print(df_test.value_counts('global_label'))

df_train.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_spon_train.csv', index=False)
df_test.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_spon_test.csv', index=False)
