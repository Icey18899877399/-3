from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
from sklearn.metrics import auc, roc_curve
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r'/home/user/-3/实验3代码/chapter3_data_handled/trainafter.csv', index_col=0)
X = df.drop(['Y'], axis=1).values
y = df['Y'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 处理类别不平衡
st = SMOTETomek()
X_train_res, y_train_res = st.fit_resample(X_train, y_train)

# L1特征选择（只保留一次）
sfm = SelectFromModel(LogisticRegression(penalty="l1", C=1.0, solver='liblinear'))
sfm.fit(X_train_res, y_train_res)

X_train_sel = sfm.transform(X_train_res)
X_test_sel = sfm.transform(X_test)

# 训练模型
model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
model.fit(X_train_sel, y_train_res)

# 预测概率
y_prob = model.predict_proba(X_test_sel)[:, 1]

# 保存结果（去掉无意义的转置操作）
output = np.column_stack((np.arange(len(y_test)), y_prob, y_test))
np.savetxt(
    r'/home/user/-3/实验3代码/chapter3_data_handled/LR.txt',
    output,
    fmt="%d %0.9f %d"
)

# ROC绘图
def report_auc(y_true, y_prob, title="", output_name="", lw=2):
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=lw, label='ROC (AUC = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(r"/home/user/-3/实验3代码/chapter3_data_handled/LR_L1_" + output_name, dpi=800)
    plt.close()

# 绘图
report_auc(y_test, y_prob, 'Logistic with L1 penalty', 'LG')