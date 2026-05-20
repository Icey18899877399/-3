import pandas as pd
import numpy as np
import time
# import numba 加速处理
from collections import Counter
fh = r"D:\文档\机器学习\实验3代码\chapter3_data\contest_basic_train.tsv"

train = pd.read_csv(fh, sep="\t", header=0)
train.isnull().sum()
### 1. 变量属性数值化与缺失值处理
# 根据身份证号获取性别(男1，女0)
train['gender'] = list(map(lambda x:1 if int(str(x)[-2]) % 2 ==1 else 0, train['ID_CARD']))

#地区归并到省，取地区编码前两位
#缺失值用身份证前两位替代（出生地）
province=list(map(lambda x, y:y[:2] if np.isnan(x) else str(x)[:2], train['WORK_PROVINCE'], train['ID_CARD']))
train['WORK_PROVINCE']=province

# 为了省内存(axis=1)
train=train.drop("ID_CARD", axis=1)


# 用字典对特征类别进行标注
is_local_repalce_dict = {'本地籍':"0","非本地籍":"1"}
train["IS_LOCAL"] = train["IS_LOCAL"].map(is_local_repalce_dict)

train['MARRY_STATUS'].value_counts()
marry_status_dict = {"已婚":"0", "未婚":"1", "离异":"2", "离婚":"2", "其他":"4", "丧偶":"5"}
train["MARRY_STATUS"] = train["MARRY_STATUS"].map(marry_status_dict)

# 2. 开始给名义变量独热编码（one hot encoding），产生dummy
# 注意其中省份为两位数编码，但是原数据读进去的是浮点型，需要先转成字符串
train_part = pd.get_dummies(train.loc[:,["AGENT","IS_LOCAL","WORK_PROVINCE","MARRY_STATUS"]],drop_first=True)
train[train_part.columns]=train_part

if 'AGENT' in train.columns:
    train.drop( ['AGENT', 'IS_LOCAL','WORK_PROVINCE','MARRY_STATUS'],axis=1, inplace=True)

#给有序变量定义编码
train['EDU_LEVEL'].value_counts()
edu_level_dict={"初中":0,"高中":1,"专科及以下":2,"专科":2,"本科":3,"硕士研究生":4,"博士研究生":4,"硕士及以上":4,"其他":5}
if train["EDU_LEVEL"].dtype !="float64":
    train["EDU_LEVEL"]=train["EDU_LEVEL"].map(edu_level_dict)
    
# 处理时间：目前有如下思路
## 1. 找一个给定日期，得到距离该日期的天数，作为变量：贷款时间
## 2. 会不会说，月初和月末贷款有一定的差异性，因为发工资一般是在月初
due_date=pd.to_datetime("2017/12/31")
train["LOAN_DATE"]=pd.to_datetime(train["LOAN_DATE"])
train["LOAN_TIME"]=list(map(lambda x: (due_date - x).days, train["LOAN_DATE"]))

#缺失值统计
train.isnull().sum()

# 3. 异常值检测
# 可能出现异常值的变量：
###  SALARY
train['SALARY'].describe()

# 没有异常值

# 4. 连续指标标准化
## 需要标准化的变量：
### 1. SALARY
### 2. LOAN_TIME
### 3. EDU_LEVEL
def scalar(series):
    theMean = series.mean()
    theStd = series.std()
    return series.map(lambda x: (x-theMean) / theStd)

train['SALARY'] = scalar(train['SALARY'])
train['LOAN_TIME'] = scalar(train['LOAN_TIME'])
train['EDU_LEVEL'] = scalar(train['EDU_LEVEL'])
df_train = train.copy()




# test  测试集

fh=r"D:\文档\机器学习\实验3代码\chapter3_data\contest_basic_test.tsv"
train=pd.read_csv(fh,sep="\t", header=0)
train.isnull().sum()

# 根据身份证号获取性别(男1，女0)
train['gender']=list(map(lambda x:1 if int(str(x)[-2]) % 2 ==1 else 0, train['ID_CARD']))

# 地区归并到省，取地区编码前两位
# 缺失值用身份证前两位替代（出生地）
province=list(map(lambda x, y:y[:2] if np.isnan(x) else str(x)[:2],train['WORK_PROVINCE'],train['ID_CARD']))
train['WORK_PROVINCE']=province

## 为了省内存(axis=1)

train=train.drop("ID_CARD",axis=1)


# 用字典对特征类别进行标注
is_local_repalce_dict ={'本地籍':"0","非本地籍":"1"}
train["IS_LOCAL"]=train["IS_LOCAL"].map(is_local_repalce_dict)

train['MARRY_STATUS'].value_counts()
marry_status_dict={"已婚":"0","未婚":"1","离异":"2","离婚":"2","其他":"4","丧偶":"5"}
train["MARRY_STATUS"]=train["MARRY_STATUS"].map(marry_status_dict)

# 开始给名义变量独热编码（one hot encoding），产生dummy
# 注意其中省份为两位数编码，但是原数据读进去的是浮点型，需要先转成字符串
train_part=pd.get_dummies(train.loc[:,["AGENT","IS_LOCAL","WORK_PROVINCE","MARRY_STATUS"]],drop_first=True)
train[train_part.columns]=train_part
if 'AGENT' in train.columns:
    train.drop( ['AGENT', 'IS_LOCAL','WORK_PROVINCE','MARRY_STATUS'],axis=1, inplace=True)

# 给有序变量定义编码
train['EDU_LEVEL'].value_counts()
edu_level_dict={"初中":0,"高中":1, "专科及以下":2, "专科":2, "本科":3, "硕士研究生":4, "博士研究生":4, "硕士及以上":4, "其他":5}
if train["EDU_LEVEL"].dtype !="float64":
    train["EDU_LEVEL"]=train["EDU_LEVEL"].map(edu_level_dict)
    
# 处理时间：目前有如下思路
## 1. 找一个给定日期，得到距离该日期的天数，作为变量：贷款时间
## 2. 会不会说，月初和月末贷款有一定的差异性，因为发工资一般是在月初
due_date = pd.to_datetime("2017/12/31")
train["LOAN_DATE"] = pd.to_datetime(train["LOAN_DATE"])
train["LOAN_TIME"] = list(map(lambda x: (due_date - x).days, train["LOAN_DATE"]))

# 缺失值统计
train.isnull().sum()

# 异常值检测
## 可能出现异常值的变量：
###  SALARY
train['SALARY'].describe()
# 没有异常值

# 连续指标标准化
## 需要标准化的变量：
### 1. SALARY
### 2. LOAN_TIME
### 3. EDU_LEVEL
def scalar(series):
    theMean = series.mean()
    theStd = series.std()
    return series.map(lambda x: (x-theMean) / theStd)

train['SALARY'] = scalar(train['SALARY'])
train['LOAN_TIME'] = scalar(train['LOAN_TIME'])
train['EDU_LEVEL'] = scalar(train['EDU_LEVEL'])


df_test = train.copy()


df_train.index = df_train.REPORT_ID
df_test.index = df_test.REPORT_ID
del df_train['REPORT_ID']
del df_test['REPORT_ID']

# 简单的更改train为test，变得到了测试集
# 但是发现测试集变量要多余训练集，这是因为有的离散变量取值训练集里面没有，那就直接删去测试集中的虚拟变量

train_columns = df_train.columns
test_columns = df_test.columns

intersection = list(set(train_columns) & set(test_columns))
df_test = df_test.loc[:, intersection]
df_train = df_train.loc[:, intersection+['Y']]
# print(df_test.shape)
# print(df_train.shape)

df_train.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_1_train.csv')
df_test.to_csv(r'D:\文档\机器学习\实验3代码\chapter3_data_handled\table_1_test.csv')


