import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

#table8
table_8 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data/contest_ext_crd_qr_recordsmr.tsv', sep='\t')
table_8.columns = [str.lower(i) for i in table_8.columns]

table_8['type_id'] = table_8.type_id.astype('str')

type_dw_dummy = pd.get_dummies(table_8[['report_id', 'type_id']])
df_query_reason = type_dw_dummy.groupby('report_id').mean()
df_query_reason.columns = ['recordsmr_'+ i for i in df_query_reason.columns]

df_continuous_mean = table_8.loc[:, ['report_id', 'sum_dw']]
df_continuous_mean = df_continuous_mean.groupby('report_id').mean()

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

table_8_handled = pd.merge(df_query_reason, df_continuous_mean, left_index=True, right_index=True)
table_8_handled.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_8.csv')
