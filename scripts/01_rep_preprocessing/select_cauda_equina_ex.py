import pandas as pd 


dst = '/work/robinpark/AutoLabelClassifier/data/osclmric_reports/'


all_osc = pd.read_csv('/work/robinpark/PID010A_clean/all_osclmric_reports.csv')

df_no_cauda = all_osc.loc[all_osc.report.str.lower().str.find('no cauda') > -1]

df_cauda = all_osc.loc[
    ((all_osc.report.str.lower().str.find('cauda equina compression') > -1) |
    (all_osc.report.str.lower().str.find('compression of cauda equina') > -1))
    & 
    (all_osc.report.str.lower().str.find('no compression of cauda') == -1) & 
    (all_osc.report.str.lower().str.find('normal cauda') == -1) & 
    (all_osc.report.str.lower().str.find('normal conus and cauda') == -1) & 
    (all_osc.report.str.lower().str.find('no cauda') == -1) & 
    (all_osc.report.str.lower().str.find('? cauda') == -1) & 
    (all_osc.report.str.lower().str.find('?cauda') == -1) & 
    (all_osc.report.str.lower().str.find('no evidence of cauda') == -1) & 
    (all_osc.report.str.lower().str.find('cauda equina are normal') == -1) & 
    (all_osc.report.str.lower().str.find('cauda equina are unremarkable') == -1) & 
    (all_osc.report.str.lower().str.find('cauda equina is unremarkable') == -1) &
    (all_osc.report.str.lower().str.find('cauda equina?') == -1) & 
    (all_osc.report.str.lower().str.find('craniocaudal') == -1) 
]

# Pick 100 from each dataset with seed
df_no_cauda = df_no_cauda.sample(100, random_state=42)
df_no_cauda['guess'] = 0
#df_no_cauda.to_csv(f'{dst}/manually_labeled_no_cauda_equina.csv', index=False)

df_cauda['guess'] = 1 
df_cauda.to_csv(f'{dst}/manually_labeled_cauda_equina_raw.csv', index=False)