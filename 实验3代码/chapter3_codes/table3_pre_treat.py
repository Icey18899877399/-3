from collections import Counter
import pandas as pd
import numpy as np
import time


table_3 = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data/contest_ext_crd_cd_ln.tsv', sep='\t')
table_3.shape
table_3.head()

# 变量替换
state_dict = {
    '结清':'JQ',
    '逾期':'YQ',
    '正常':'NM',
    '呆账':'DZ',
    '转出':'ZC',
    '销户':'XH',
    '冻结':'FZ',
    '止付':'EP',
    '未激活':'IA'
}
type_dw_dict = {
    '个人经营性贷款':'PBL',
    '个人汽车贷款':'PAL',
    '个人商用房（包括商住两用）贷款':'PCHL',
    '个人消费贷款':'PCL',
    '个人住房贷款':'PHL',
    '个人住房公积金贷款':'IHL',
    '个人助学贷款':'ILE',
    '农户贷款':'FL',
    '其他贷款':'OTHERS'
}
guarantee_type_dict = {
    '保证':'G',
    '抵押担保':'G_US',
    '农户联保':'G_PL',
    '其他担保':'G_OTHERS',
    '信用/免担保':'G_NG',
    '信用免担保':'G_NG',
    '质押（含保证金）担保':'G_PG_IM',
    '组合（不含保证）担保':'G_CG_WW',
    '组合（含保证）担保':'G_CG_IW'
}
payment_rating_dict = {
    '按半年归还':'P_H',
    '按季归还':'P_S',
    '按年归还':'P_Y',
    '按其他方式归还':'P_OHTERS',
    '按日归还':'P_D',
    '按月归还':'P_M',
    '按周归还':'P_W',
    '不定期归还':'P_U',
    '一次性归还':'P_O'
}
class5_state_dict = {
    '次级':'C_I',
    '关注':'C_F',
    '可疑':'C_D',
    '未知':'C_U',
    '正常':'C_N'

}


# 检查
print(table_3['state'].map(state_dict).value_counts())
print(table_3['type_dw'].map(type_dw_dict).value_counts())
print(table_3['guarantee_type'].map(guarantee_type_dict).value_counts())
print(table_3['payment_rating'].map(payment_rating_dict).value_counts())
print(table_3['class5_state'].map(class5_state_dict).value_counts())


table_3['state'] = table_3['state'].map(state_dict)
table_3['type_dw'] = table_3['type_dw'].map(type_dw_dict)
table_3['guarantee_type'] = table_3['guarantee_type'].map(guarantee_type_dict)
table_3['payment_rating'] = table_3['payment_rating'].map(payment_rating_dict)
table_3['class5_state'] = table_3['class5_state'].map(class5_state_dict)
table_3['currency'] = list(map(lambda x: 1 if x == '人民币' else 0, table_3['currency']))

#归并到report id
from sklearn.preprocessing import OneHotEncoder
table_3.isnull().sum()
#独热编码
state_dummy = pd.get_dummies(table_3[['report_id', 'state']])
df_state = state_dummy.groupby('report_id').mean()

type_dw_dummy = pd.get_dummies(table_3[['report_id', 'type_dw']])
df_type_dw = type_dw_dummy.groupby('report_id').mean()

guarantee_type_dummy = pd.get_dummies(table_3[['report_id', 'guarantee_type']])
df_guarantee_type = guarantee_type_dummy.groupby('report_id').mean()

payment_rating_dummy = pd.get_dummies(table_3[['report_id', 'payment_rating']])
df_payment_rating = payment_rating_dummy.groupby('report_id').mean()

# 注意，行和不唯一是因为有缺失值，缺失值在转换为虚拟变量时全变成了0
class5_state_dummy = pd.get_dummies(table_3[['report_id', 'class5_state']])
df_class5_state = class5_state_dummy.groupby('report_id').mean()


df_currency = table_3.loc[:, ['report_id', 'currency']].groupby('report_id').mean()
(df_currency !=1)['currency'].value_counts()
# 全为1， 无意义


df_count = table_3.loc[:, ['report_id', 'payment_rating']].groupby('report_id').count()
df_count.columns = ['loan_count']

#连续变量处理
df_continuous = table_3.loc[:, ['report_id','payment_cyc', 'credit_limit_amount', 'balance', 'remain_payment_cyc', 'scheduled_payment_amount', 
                               'actual_payment_amount', 'curr_overdue_cyc', 'curr_overdue_amount']]
df_continuous_mean = df_continuous.groupby('report_id').mean()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

#所有变量merge
table_3_handled = pd.merge(df_state, df_type_dw, left_index=True, right_index=True)
table_3_handled = pd.merge(table_3_handled, df_guarantee_type, left_index=True, right_index=True)
table_3_handled = pd.merge(table_3_handled, df_payment_rating, left_index=True, right_index=True)
table_3_handled = pd.merge(table_3_handled, df_class5_state, left_index=True, right_index=True)
table_3_handled = pd.merge(table_3_handled, df_continuous_mean, left_index=True, right_index=True)
table_3_handled = pd.merge(table_3_handled, df_count, left_index=True, right_index=True)

table_3_handled.to_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/table_3.csv')



