import pandas as pd 

clean_path = '/work/robinpark/PID010A_clean'

li_reports = []

for i in range(100):
    j = i+1
    report = pd.read_excel(clean_path + '/OSCLMRIC_100_annotated_reports.xlsx',
                            sheet_name=f'Report{j}').drop(columns=['Unnamed: 12'])
    
    for column in report.columns:
    # Create a new column that assigns 1 if any value in the existing column is equal to 1
        report[column + '_new'] = report[column].apply(lambda x: 1 if x == 1 else 0)

    # Get the max of the new columns 
    report['result'] = report.filter(like='_new').max(axis=1)
    # Drop intermediate columns
    report = report.drop(columns=report.filter(like='_new').columns)

    report['Report'] = report['Report'][8]

    report = report[['Report','segment','result']]
    report = report.loc[report.segment.isin(['T12-L1','L1-L2','L2-L3','L3-L4','L4-L5','L5-S1'])]
    # Pivot to one row per report and segment as columns 
    report = report.pivot(index='Report', columns='segment', values='result').reset_index()
    li_reports.append(report)

osclmric_reports = pd.concat(li_reports).reset_index(drop=True)

osclmric_reports.to_csv('/work/robinpark/PID010A_clean/OSCLMRIC_reports_stenosis_by_ivd.csv')