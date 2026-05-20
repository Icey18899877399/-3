from collections import Counter
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

table_6 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_is_sharedebt.csv', sep=',')
table_6.columns = [str.lower(i) for i in table_6.columns]

type_dw_dict = {
'未销户贷记卡信息汇总': 'DJK',
'未结清贷款信息汇总': 'DK',
'未销户准贷记卡信息汇总': 'ZDJK'
}

table_6['type_dw'] = table_6['type_dw'].map(type_dw_dict)

type_dw_dummy = pd.get_dummies(table_6[['report_id', 'type_dw']])
df_type_dw = type_dw_dummy.groupby('report_id').mean()
df_type_dw.columns = ['sharedebt_'+ i for i in df_type_dw.columns]

df_continuous_mean = table_6.drop(['type_dw'], axis=1)
df_continuous_mean = df_continuous_mean.groupby('report_id').mean()

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

table_6_handled = pd.merge(df_type_dw, df_continuous_mean, left_index=True, right_index=True)
table_6_handled.head()
table_6_handled.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_6.csv')
