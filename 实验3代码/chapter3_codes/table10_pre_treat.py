import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
#table10
table_10 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_cd_ln_spl.tsv', sep='\t')
table_10.columns = [str.lower(i) for i in table_10.columns]
table_10.type_dw.value_counts()
type_dw_dict = {
    '提前还款（全部）': '1',
    '提前还款（部分）': '2',
    '其他': '3',
    '展期（延期）':'4',
    '担保人代还 ':'5'
}

table_10['type_dw'] = table_10['type_dw'].map(type_dw_dict)

type_dw_dummy = pd.get_dummies(table_10[['report_id', 'type_dw']])
df_type_dw = type_dw_dummy.groupby('report_id').mean()
df_type_dw.columns = ['spl_'+ i for i in df_type_dw.columns]

df_continuous_mean = table_10.loc[:, ['report_id', 'changing_months', 'changing_amount']]
df_continuous_mean = df_continuous_mean.groupby('report_id').mean()

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

table_10_handled = pd.merge(df_type_dw, df_continuous_mean, left_index=True, right_index=True)

table_10_handled.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_10.csv')