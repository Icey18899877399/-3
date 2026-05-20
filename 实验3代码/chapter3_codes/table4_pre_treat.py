import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

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

#table4
table_4 = pd.read_csv(r'D:\文档\机器学习\实验3代码\chapter3_data\contest_ext_crd_cd_lnd.tsv', sep='\t')

table_4.loc[:, ['state', 'currency', 'guarantee_type', 'cardtype']].head()

table_4.loc[:, ['credit_limit_amount', 'share_credit_limit_amount', 'used_credit_limit_amount',
                'latest6_month_used_avg_amount', 'used_highest_amount', 'scheduled_payment_amount',
                'actual_payment_amount', 'curr_overdue_cyc', 'curr_overdue_amount']].head()

table_4['state'] = table_4['state'].map(state_dict)
table_4['guarantee_type'] = table_4['guarantee_type'].map(guarantee_type_dict)
table_4['currency'] = list(map(lambda x: 1 if x == '人民币' else 0, table_4['currency']))

#独热编码
state_dummy = pd.get_dummies(table_4[['report_id', 'state']])
df_state = state_dummy.groupby('report_id').mean()
df_state.columns = ['ind_'+ i for i in df_state.columns]

guarantee_type_dummy = pd.get_dummies(table_4[['report_id', 'guarantee_type']])
df_guarantee_type = guarantee_type_dummy.groupby('report_id').mean()
df_guarantee_type.columns = ['ind_'+ i for i in df_guarantee_type.columns]

df_count = table_4.loc[:, ['report_id', 'state']].groupby('report_id').count()
df_count.columns = ['loan_count']

currency_dummy = table_4[['report_id', 'currency']]
df_currency = currency_dummy.groupby('report_id').mean()
df_currency.columns = ['ind_'+ i for i in df_currency.columns]

#连续变量
df_continuous = table_4.loc[:, ['report_id', 'credit_limit_amount', 'share_credit_limit_amount', 'used_credit_limit_amount',
                'latest6_month_used_avg_amount', 'used_highest_amount', 'scheduled_payment_amount',
                'actual_payment_amount', 'curr_overdue_cyc', 'curr_overdue_amount']]
df_continuous_mean = df_continuous.groupby('report_id').mean()


ss = StandardScaler()
columns = df_continuous_mean.columns
indexs =  df_continuous_mean.index
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
df_continuous_mean = pd.DataFrame(ss.fit_transform(imr.fit_transform(df_continuous_mean.values)))
df_continuous_mean.columns = columns
df_continuous_mean.index = indexs

table_4_handled = pd.merge(df_state, df_guarantee_type, left_index=True, right_index=True)
table_4_handled = pd.merge(table_4_handled, df_count, left_index=True, right_index=True)
table_4_handled = pd.merge(table_4_handled, df_currency, left_index=True, right_index=True)
table_4_handled = pd.merge(table_4_handled, df_continuous_mean, left_index=True, right_index=True)

table_4_handled.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_4.csv')
