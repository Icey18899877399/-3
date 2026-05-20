import pandas as pd
import numpy as np

df_table_1_train = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_1_train.csv', index_col=0)
df_table_1_test = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_1_test.csv', index_col=0)
df_table_2 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_2.csv', index_col=0)
df_table_3 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_3.csv', index_col=0)
df_table_4 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_4.csv', index_col=0)
df_table_5 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_5.csv', index_col=0)
df_table_6 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_6.csv', index_col=0)
df_table_7 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_7.csv', index_col=0)
df_table_8 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_8.csv', index_col=0)
#df_table_9 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_9.csv', index_col=0)
df_table_10 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_10.csv', index_col=0)
df_table_11 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_11.csv', index_col=0)

# 训练集：table_1_train 为主表，依次左连接 table_2 ~ table_11
df_all = pd.merge(df_table_1_train, df_table_2, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_3, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_4, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_5, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_6, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_7, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_8, how='left', left_index=True, right_index=True)
#df_all = pd.merge(df_all, df_table_9, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_10, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_11, how='left', left_index=True, right_index=True)

print('train_all shape:', df_all.shape)
df_all.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/train_all.csv')

# 测试集：table_1_test 为主表
df_all = pd.merge(df_table_1_test, df_table_2, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_3, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_4, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_5, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_6, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_7, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_8, how='left', left_index=True, right_index=True)
#df_all = pd.merge(df_all, df_table_9, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_10, how='left', left_index=True, right_index=True)
df_all = pd.merge(df_all, df_table_11, how='left', left_index=True, right_index=True)

print('test_all shape:', df_all.shape)
df_all.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/test_all.csv')
