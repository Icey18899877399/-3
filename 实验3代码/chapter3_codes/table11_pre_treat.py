import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
#table11

table_11 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data/contest_ext_crd_cd_lnd_ovd.csv', sep=',')
table_11.columns = [str.lower(i) for i in table_11.columns]

df_continuous_mean = table_11.loc[:, ['report_id', 'last_months', 'amount']]
df_continuous_mean = df_continuous_mean.groupby('report_id').mean()


ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

df_continuous_mean.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_11.csv')