import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df_table_all = pd.read_csv("/home/user/-3/实验3代码/chapter3_data_handled/train_all.csv", index_col=0)
# merge 修复后只剩一列 LOAN_DATE
if 'LOAN_DATE' in df_table_all.columns:
    df_table_all = df_table_all.drop(['LOAN_DATE'], axis=1)
df_table_all = df_table_all.dropna(axis=1, how='all')

columns = df_table_all.columns
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_table_all = pd.DataFrame(imr.fit_transform(df_table_all.values))
df_table_all.columns = columns

df_table_all.to_csv("/home/user/-3/实验3代码/chapter3_data_handled/trainafter.csv")
print('trainafter shape:', df_table_all.shape)
