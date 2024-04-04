import pandas as pd 

from glob import glob

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

    # Get the result per report
    sum_report = report[['Report','result']].groupby('Report').sum().reset_index()
    sum_report.loc[sum_report['result']>0,'result'] = 1

    li_reports.append(sum_report)

osclmric_reports = pd.concat(li_reports).reset_index(drop=True)

osclmric_reports.to_csv('/work/robinpark/PID010A_clean/OSCLMRIC_reports_any_stenosis.csv')