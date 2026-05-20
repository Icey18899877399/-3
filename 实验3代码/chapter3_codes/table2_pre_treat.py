import pandas as pd
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

#table2
table_2 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_hd_report.csv', sep=',')
table_2.columns = [str.lower(i) for i in table_2.columns]

table_2.query_reason.value_counts()
query_reason_dict = {
    '担保资格审查': '1',
    '贷款审批': '2',
    '贷后管理': '3'
}
table_2['query_reason'] = table_2['query_reason'].map(query_reason_dict)

type_dw_dummy = pd.get_dummies(table_2[['report_id', 'query_reason']])
df_query_reason = type_dw_dummy.groupby('report_id').mean()
df_query_reason.columns = ['report_'+ i for i in df_query_reason.columns]

query_org_dummy = pd.get_dummies(table_2[['report_id', 'query_org']])
df_query_org_dummy = query_org_dummy.groupby('report_id').mean()
df_query_org_dummy.columns = ['report_'+ i for i in df_query_org_dummy.columns]

table_2_handled = pd.merge(df_query_reason, df_query_org_dummy, left_index=True, right_index=True)

table_2_handled.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_2.csv')





