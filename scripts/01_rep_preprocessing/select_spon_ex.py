import pandas as pd 

dst = '/work/robinpark/AutoLabelClassifier/data/osclmric_reports/'

all_osc = pd.read_csv('/work/robinpark/PID010A_clean/all_osclmric_reports.csv')

df_spon = all_osc.loc[(all_osc.report.str.lower().str.find('spondylolisthesis') > -1)]
df_no_spon_mention = all_osc.loc[(all_osc.report.str.lower().str.find('spondylolisthesis') == -1)]

# Pick 120 from each dataset with seed
df_spon = df_spon.sample(80, random_state=42)
df_no_spon_mention = df_no_spon_mention.sample(80, random_state=42)

df_concat = pd.concat([df_spon, df_no_spon_mention])

df_concat.to_csv(f'{dst}/manually_labeled_spon_raw.csv', index=False)
