import pandas as pd 

from glob import glob
from sklearn.model_selection import train_test_split


df_no_ce = pd.read_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_no_cauda_equina.csv')
df_ce = pd.read_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_cauda_equina.csv')

df_all = pd.concat([df_no_ce,df_ce])
df_all = df_all.loc[df_all.global_label != -1]

# Split in half
df_train, df_test = train_test_split(df_all, test_size=0.5, random_state=42, stratify=df_all.global_label)

df_train.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_cauda_equina_train.csv', index=False)
df_test.to_csv('/work/robinpark/AutoLabelClassifier/data/osclmric_reports/manually_labeled_cauda_equina_test.csv', index=False)