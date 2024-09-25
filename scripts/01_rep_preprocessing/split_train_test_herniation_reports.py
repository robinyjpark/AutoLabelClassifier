import pandas as pd

df_hern = pd.read_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_herniation.csv')

df_hern = df_hern.loc[df_hern.label != -1]

# Shuffle the rows
df_hern = df_hern.sample(frac=1, random_state=0).reset_index(drop=True)

# Split into two dataframes
df_train = df_hern.iloc[:int(len(df_hern)*0.5)]
df_test = df_hern.iloc[int(len(df_hern)*0.5):]

print(df_train.value_counts('label'))
print(df_test.value_counts('label'))

df_train.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/herniation_reports_labeled_train.csv', index=False)
df_test.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/herniation_reports_labeled_test.csv', index=False)
