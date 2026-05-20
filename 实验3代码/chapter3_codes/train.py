
from sklearn.impute import SimpleImputer




# from sklearn.preprocessing import Imputer

# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pip
pip.main(['install', "imblearn"])
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np

df_table_all = pd.read_csv(r'train_all.csv', index_col=0)
df_table_all = df_table_all.drop(['LOAN_DATE'], axis=1)
df_table_all = df_table_all.dropna(axis=1,how='all')

columns = df_table_all.columns
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_table_all = pd.DataFrame(imr.fit_transform(df_table_all.values))
df_table_all.columns = columns

X = df_table_all.drop('Y', axis=1).values
Y = np.array(df_table_all['Y'])

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3, random_state=0)

data_pairs = [(X_train, y_train, X_test, y_test)]

st = SMOTETomek()
# X_train_st, y_train_st = st.fit_sample(X_train, y_train)
X_train_st, y_train_st = st.fit_resample(X_train, y_train)
sfm = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))
# sfm = SelectFromModel(LogisticRegression(penalty="l1", C=1.0))
sfm.fit(X_train, y_train)
X_train_tiny = sfm.transform(X_train)
X_test_tiny = sfm.transform(X_test)

sfm.fit(X_train_st, y_train_st)
X_train_st_tiny = sfm.transform(X_train_st)
X_test_st_tiny = sfm.transform(X_test)

sfm = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'))
sfm.fit(X_train, y_train)
X_train_tiny = sfm.transform(X_train)
X_test_tiny = sfm.transform(X_test)

sfm.fit(X_train_st, y_train_st)
X_train_st_tiny = sfm.transform(X_train_st)
X_test_st_tiny = sfm.transform(X_test)

model1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model1.fit(X_train_st_tiny, y_train_st)
y_pred = model1.predict_proba(X_test_st_tiny)
y=y_pred[:,1]

np.savetxt('123.txt',(y,y_test))


c = np.vstack((a,b))
print("c="+str(c))
