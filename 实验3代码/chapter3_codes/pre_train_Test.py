import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df_table_all = pd.read_csv("D:\\文档\\机器学习\\实验3代码\\chapter3_data_handled\\test_all.csv", index_col=0)
df_table_all = df_table_all.drop(['LOAN_DATE_x'], axis=1)
df_table_all = df_table_all.drop(['LOAN_DATE_y'], axis=1)
df_table_all = df_table_all.dropna(axis=1,how='all')

columns = df_table_all.columns
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_table_all = pd.DataFrame(imr.fit_transform(df_table_all.values))
df_table_all.columns = columns
df_table_all.to_csv("D:/文档/机器学习/实验3代码/chapter3_data_handled/testafter.csv")
