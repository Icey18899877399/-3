from collections import Counter
import pandas as pd
import numpy as np
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

table_3 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_cd_ln.tsv', sep='\t')
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

#table5
table_5 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_is_creditcue.csv', sep=',')
table_5.columns = [str.lower(i) for i in table_5.columns]
#字符转化为小写

df_continuous_mean = table_5.loc[:, ['house_loan_count', 'commercial_loan_count', 'other_loan_count', 'loancard_count',
                'standard_loancard_count', 'announce_count', 'dissent_count']]
df_continuous_mean.index = table_5['report_id']

ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

df_continuous_mean.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_5.csv')