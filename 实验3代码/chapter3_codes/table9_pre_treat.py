import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
#table9
table_9 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data/contest_ext_crd_qr_recorddtlinfo.tsv', sep='\t')
table_9.columns = [str.lower(i) for i in table_9.columns]

table_9['query_reason'] = table_9.query_reason.astype('str')

type_dw_dummy = pd.get_dummies(table_9[['report_id', 'query_reason']])
df_query_reason = type_dw_dummy.groupby('report_id').mean()
df_query_reason.columns = ['recordsmr_'+ i for i in df_query_reason.columns]

# table_9 中没有 sum_dw 列，使用每个 report_id 的查询次数作为数值特征
# 这样可以保留 numeric 特征并与独热编码结果合并

df_continuous_mean = table_9.groupby('report_id').size().to_frame('query_count')

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)), columns=columns, index=indexs)

table_9_handled = pd.merge(df_query_reason, df_continuous_mean, left_index=True, right_index=True)

table_9_handled.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_9.csv')