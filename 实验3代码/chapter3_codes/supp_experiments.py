"""补充实验：A. 梯度提升模型扩展（XGBoost / LightGBM / GBDT）
              B. 消融实验（采样策略 / 特征选择 / 标准化）

所有图采用顶刊（Nature / Cell）风格：Wong 色盲友好色板，去顶/右边框，等等。
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# =========================== 全局美化配置 (Nature 风格) ===========================
# Wong 色盲友好色板 — Nature 推荐
PALETTE = {
    'Logistic (L1)':  '#0072B2',   # 蓝
    'Naive Bayes':    '#009E73',   # 青绿
    'Random Forest':  '#D55E00',   # 橙红
    'SVM (linear)':   '#CC79A7',   # 粉
    'XGBoost':        '#E69F00',   # 橙
    'LightGBM':       '#56B4E9',   # 浅蓝
    'GBDT':           '#F0E442',   # 黄
}
# 消融实验色板
ABLATION_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.dpi': 120,
})

ROOT = '/home/user/-3/实验3代码'
HANDLED = os.path.join(ROOT, 'chapter3_data_handled')
FIG_DIR = os.path.join(HANDLED, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# =========================== 读取数据 ===========================
print('Loading data ...')
df = pd.read_csv(os.path.join(HANDLED, 'trainafter.csv'), index_col=0)
X = df.drop(['Y'], axis=1).values
y = df['Y'].values.astype(int)
print(f'  shape: {X.shape}, positive ratio: {y.mean():.4f}')

X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)


def build_standard_pipeline(X_train, y_train, X_test,
                            do_sampling='smotetomek',
                            do_feature_selection=True):
    """统一的预处理 pipeline:
       (1) 不平衡采样, (2) 可选 L1 特征选择.
       返回 X_train_proc, y_train_proc, X_test_proc.
    """
    if do_sampling == 'smotetomek':
        X_tr, y_tr = SMOTETomek(random_state=0).fit_resample(X_train, y_train)
    elif do_sampling == 'smote':
        X_tr, y_tr = SMOTE(random_state=0).fit_resample(X_train, y_train)
    elif do_sampling == 'none':
        X_tr, y_tr = X_train, y_train
    else:
        raise ValueError(do_sampling)

    if do_feature_selection:
        sfm = SelectFromModel(
            LogisticRegression(penalty='l1', C=1.0, solver='liblinear',
                               max_iter=200, random_state=0))
        sfm.fit(X_tr, y_tr)
        X_tr_out = sfm.transform(X_tr)
        X_te_out = sfm.transform(X_test)
    else:
        X_tr_out = X_tr
        X_te_out = X_test
    return X_tr_out, y_tr, X_te_out


def eval_model(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
    fpr, tpr, _ = roc_curve(y_te, y_prob, pos_label=1)
    return auc(fpr, tpr), fpr, tpr, y_prob


# =========================== 实验 A: 梯度提升模型扩展 ===========================
print('\n[A] Gradient-boosting models ...')
X_tr_std, y_tr_std, X_te_std = build_standard_pipeline(
    X_train_raw, y_train_raw, X_test_raw,
    do_sampling='smotetomek', do_feature_selection=True)
print(f'  After pipeline: train={X_tr_std.shape}, test={X_te_std.shape}')

results_A = {}

# 已有四个基线模型
models_baseline = {
    'Logistic (L1)':  LogisticRegression(penalty='l1', C=1.0,
                                         solver='liblinear', random_state=0),
    'Naive Bayes':    GaussianNB(),
    'Random Forest':  RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                             criterion='gini', random_state=0),
    'SVM (linear)':   SVC(C=0.1, kernel='linear',
                          probability=True, random_state=0),
}
# 新增三个梯度提升模型
models_new = {
    'XGBoost':  xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric='auc',
        n_jobs=-1, random_state=0, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, max_depth=-1, num_leaves=31, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
        random_state=0, verbose=-1),
    'GBDT':     GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0),
}

all_models = {**models_baseline, **models_new}
for name, mdl in all_models.items():
    a, fpr, tpr, prob = eval_model(mdl, X_tr_std, y_tr_std, X_te_std, y_test)
    results_A[name] = {'auc': a, 'fpr': fpr, 'tpr': tpr}
    print(f'  {name:<18s}  AUC = {a:.4f}')


# ---------- 图 A1: 7 个模型的 ROC 曲线对比 ----------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
order = ['XGBoost', 'LightGBM', 'GBDT',
         'Logistic (L1)', 'SVM (linear)', 'Random Forest', 'Naive Bayes']
for name in order:
    r = results_A[name]
    ax.plot(r['fpr'], r['tpr'], color=PALETTE[name], lw=1.8,
            label=f"{name} (AUC = {r['auc']:.4f})")
ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Random (AUC = 0.5)')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves of Seven Classifiers')
ax.legend(loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='lightgray', fontsize=9)
fig.savefig(os.path.join(FIG_DIR, 'A1_roc_seven_models.png'))
plt.close(fig)

# ---------- 图 A2: 7 个模型的 AUC 柱状图 ----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
order_sorted = sorted(results_A.keys(), key=lambda k: results_A[k]['auc'])
aucs = [results_A[k]['auc'] for k in order_sorted]
colors = [PALETTE[k] for k in order_sorted]
bars = ax.barh(order_sorted, aucs, color=colors, edgecolor='black', lw=0.8)
for bar, a in zip(bars, aucs):
    ax.text(a + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{a:.4f}', va='center', ha='left', fontsize=10)
ax.set_xlim(0.70, 0.92)
ax.set_xlabel('Test-set AUC')
ax.set_title('AUC Comparison: 4 Baselines vs. 3 Gradient-Boosting Models')
ax.axvline(0.85, ls='--', lw=0.8, color='gray', alpha=0.5)
fig.savefig(os.path.join(FIG_DIR, 'A2_auc_bar_seven_models.png'))
plt.close(fig)


# =========================== 实验 B1: 不平衡采样消融 ===========================
print('\n[B1] Sampling strategy ablation ...')
# 选 3 个代表性模型对比: Logistic, RF, XGBoost
focus_models = {
    'Logistic (L1)': lambda: LogisticRegression(penalty='l1', C=1.0,
                                                solver='liblinear', random_state=0),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                    random_state=0),
    'XGBoost':       lambda: xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        eval_metric='auc', n_jobs=-1, random_state=0, verbosity=0),
}
sampling_strategies = ['none', 'smote', 'smotetomek', 'class_weight']

results_B1 = {m: {} for m in focus_models}
for sname in sampling_strategies:
    print(f'  -- sampling={sname}')
    if sname == 'class_weight':
        X_tr_b, y_tr_b = X_train_raw, y_train_raw
        X_te_b = X_test_raw
    else:
        X_tr_b, y_tr_b, X_te_b = build_standard_pipeline(
            X_train_raw, y_train_raw, X_test_raw,
            do_sampling=sname, do_feature_selection=False)
    for mname, mfn in focus_models.items():
        if sname == 'class_weight':
            if mname == 'Logistic (L1)':
                mdl = LogisticRegression(penalty='l1', C=1.0,
                                         solver='liblinear', random_state=0,
                                         class_weight='balanced')
            elif mname == 'Random Forest':
                mdl = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                             random_state=0,
                                             class_weight='balanced')
            else:  # XGBoost: scale_pos_weight
                spw = (y_tr_b == 0).sum() / (y_tr_b == 1).sum()
                mdl = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.1,
                    eval_metric='auc', n_jobs=-1, random_state=0,
                    scale_pos_weight=spw, verbosity=0)
        else:
            mdl = mfn()
        a, _, _, _ = eval_model(mdl, X_tr_b, y_tr_b, X_te_b, y_test)
        results_B1[mname][sname] = a
        print(f'      {mname:<18s}  AUC = {a:.4f}')

# ---------- 图 B1: 采样策略对 AUC 的影响 (grouped bar) ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
strategies_label = ['None', 'SMOTE', 'SMOTE+Tomek', 'class_weight\n(balanced)']
x = np.arange(len(strategies_label))
width = 0.25
for i, (mname, color) in enumerate(zip(focus_models.keys(),
                                       [PALETTE['Logistic (L1)'],
                                        PALETTE['Random Forest'],
                                        PALETTE['XGBoost']])):
    vals = [results_B1[mname][s] for s in sampling_strategies]
    bars = ax.bar(x + i * width - width, vals, width,
                  color=color, edgecolor='black', lw=0.6, label=mname)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(strategies_label)
ax.set_ylabel('Test-set AUC')
ax.set_ylim(0.74, 0.91)
ax.set_title('B1. Impact of Class-Imbalance Strategies on AUC')
ax.legend(loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='lightgray')
fig.savefig(os.path.join(FIG_DIR, 'B1_sampling_ablation.png'))
plt.close(fig)


# =========================== 实验 B2: L1 特征选择消融 ===========================
print('\n[B2] L1 feature selection ablation ...')
results_B2 = {m: {} for m in focus_models}
for fs in [True, False]:
    print(f'  -- L1 feature_selection={fs}')
    X_tr_b, y_tr_b, X_te_b = build_standard_pipeline(
        X_train_raw, y_train_raw, X_test_raw,
        do_sampling='smotetomek', do_feature_selection=fs)
    n_feat = X_tr_b.shape[1]
    print(f'      retained features: {n_feat}')
    for mname, mfn in focus_models.items():
        mdl = mfn()
        a, _, _, _ = eval_model(mdl, X_tr_b, y_tr_b, X_te_b, y_test)
        results_B2[mname]['with L1' if fs else 'without L1'] = a
        results_B2[mname]['_nfeat_with' if fs else '_nfeat_without'] = n_feat
        print(f'      {mname:<18s}  AUC = {a:.4f}')

# ---------- 图 B2: 有无 L1 特征选择对比 ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
mnames = list(focus_models.keys())
x = np.arange(len(mnames))
width = 0.35
vals_with = [results_B2[m]['with L1'] for m in mnames]
vals_without = [results_B2[m]['without L1'] for m in mnames]
b1 = ax.bar(x - width / 2, vals_with, width, label='With L1 selection',
            color='#0072B2', edgecolor='black', lw=0.7)
b2 = ax.bar(x + width / 2, vals_without, width, label='Without L1 selection',
            color='#D55E00', edgecolor='black', lw=0.7)
for bs, vs in [(b1, vals_with), (b2, vals_without)]:
    for b, v in zip(bs, vs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(mnames)
ax.set_ylabel('Test-set AUC')
ax.set_ylim(0.78, 0.92)
ax.set_title('B2. Impact of L1 Feature Selection on AUC')
ax.legend(frameon=True, framealpha=0.95, edgecolor='lightgray')
fig.savefig(os.path.join(FIG_DIR, 'B2_l1_fs_ablation.png'))
plt.close(fig)


# =========================== 实验 B3: 标准化消融 ===========================
# trainafter.csv 中 SALARY / EDU_LEVEL / LOAN_TIME 与征信表已被标准化，
# 这里"反向对照"：手动反标准化主表里的几个数值列（用原始 1~7 的 SALARY 等）
print('\n[B3] Standardization ablation ...')
df_raw = pd.read_csv(os.path.join(ROOT, 'chapter3_data', 'contest_basic_train.tsv'),
                     sep='\t')
df_raw = df_raw.set_index('REPORT_ID')
# 用 df_raw 中的 SALARY 替换 trainafter.csv 中的 SALARY，得到"未标准化"版本
df_unscaled = df.copy()
df_unscaled.index = df.index.astype(int)
# 在 trainafter 中 SALARY 已经被 z-score；我们用 raw scale 替换
common_idx = df_unscaled.index.intersection(df_raw.index)
df_unscaled.loc[common_idx, 'SALARY'] = df_raw.loc[common_idx, 'SALARY'].fillna(
    df_raw['SALARY'].mean())
# SVM 对量纲敏感, 用 SVM 做对照
X_unscaled = df_unscaled.drop(['Y'], axis=1).values
X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(
    X_unscaled, y, test_size=0.3, random_state=0, stratify=y)
X_tr_u2, y_tr_u2, X_te_u2 = build_standard_pipeline(
    X_tr_u, y_tr_u, X_te_u, do_sampling='smotetomek',
    do_feature_selection=True)

results_B3 = {}
# 1) 不标准化
mdl = LogisticRegression(penalty='l1', C=1.0, solver='liblinear',
                         random_state=0, max_iter=500)
a, _, _, _ = eval_model(mdl, X_tr_u2, y_tr_u2, X_te_u2, y_te_u)
results_B3['Logistic (unscaled SALARY)'] = a
mdl = SVC(C=0.1, kernel='linear', probability=True, random_state=0)
a, _, _, _ = eval_model(mdl, X_tr_u2, y_tr_u2, X_te_u2, y_te_u)
results_B3['SVM (unscaled SALARY)'] = a
# 2) 标准化 (即原始 pipeline)
mdl = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=0)
a, _, _, _ = eval_model(mdl, X_tr_std, y_tr_std, X_te_std, y_test)
results_B3['Logistic (standardized)'] = a
mdl = SVC(C=0.1, kernel='linear', probability=True, random_state=0)
a, _, _, _ = eval_model(mdl, X_tr_std, y_tr_std, X_te_std, y_test)
results_B3['SVM (standardized)'] = a
for k, v in results_B3.items():
    print(f'  {k:<32s}  AUC = {v:.4f}')

# ---------- 图 B3: 标准化对比 ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
groups = ['Logistic', 'SVM']
vals_scaled = [results_B3['Logistic (standardized)'],
               results_B3['SVM (standardized)']]
vals_unscaled = [results_B3['Logistic (unscaled SALARY)'],
                 results_B3['SVM (unscaled SALARY)']]
x = np.arange(len(groups))
width = 0.35
b1 = ax.bar(x - width / 2, vals_scaled, width, label='Standardized',
            color='#009E73', edgecolor='black', lw=0.7)
b2 = ax.bar(x + width / 2, vals_unscaled, width, label='Unscaled SALARY',
            color='#CC79A7', edgecolor='black', lw=0.7)
for bs, vs in [(b1, vals_scaled), (b2, vals_unscaled)]:
    for b, v in zip(bs, vs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0.78, 0.90)
ax.set_ylabel('Test-set AUC')
ax.set_title('B3. Impact of Feature Standardization on AUC')
ax.legend(frameon=True, framealpha=0.95, edgecolor='lightgray')
fig.savefig(os.path.join(FIG_DIR, 'B3_standardize_ablation.png'))
plt.close(fig)


# =========================== 汇总 (写一份 csv 方便报告读) ===========================
summary_path = os.path.join(HANDLED, 'supplementary_results.csv')
records = []
for k, v in results_A.items():
    records.append({'experiment': 'A. Model expansion', 'config': k,
                    'metric': 'AUC', 'value': round(v['auc'], 4)})
for m, d in results_B1.items():
    for s, a in d.items():
        records.append({'experiment': 'B1. Sampling',
                        'config': f'{m} / {s}',
                        'metric': 'AUC', 'value': round(a, 4)})
for m, d in results_B2.items():
    for s, a in d.items():
        if s.startswith('_'):
            continue
        records.append({'experiment': 'B2. L1 selection',
                        'config': f'{m} / {s}',
                        'metric': 'AUC', 'value': round(a, 4)})
for k, v in results_B3.items():
    records.append({'experiment': 'B3. Standardization',
                    'config': k, 'metric': 'AUC', 'value': round(v, 4)})
pd.DataFrame(records).to_csv(summary_path, index=False)
print(f'\nAll figures saved to: {FIG_DIR}')
print(f'Summary saved to:     {summary_path}')
