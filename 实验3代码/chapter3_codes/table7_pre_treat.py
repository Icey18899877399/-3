from collections import Counter
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

table_7 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data/contest_ext_crd_is_ovdsummary.csv', sep=',')
table_7.columns = [str.lower(i) for i in table_7.columns]


type_dw_dict = {
'贷款逾期': 'DQ',
'贷记卡逾期': 'DJK',
'准贷记卡60天以上透支': 'ZDJK'
}

table_7['type_dw'] = table_7['type_dw'].map(type_dw_dict)

type_dw_dummy = pd.get_dummies(table_7[['report_id', 'type_dw']])
df_type_dw = type_dw_dummy.groupby('report_id').mean()
df_type_dw.columns = ['ovdsummary_'+ i for i in df_type_dw.columns]

df_continuous_mean = table_7.drop(['type_dw'], axis=1)
df_continuous_mean = df_continuous_mean.groupby('report_id').mean()

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs


table_7_handled = pd.merge(df_type_dw, df_continuous_mean, left_index=True, right_index=True)
table_7_handled.shape

table_7_handled.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_7.csv')




