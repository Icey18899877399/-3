# 机器学习实验三：贷款违约行为预测

基于某贷款机构脱敏后的多表征信数据，使用 7 种机器学习模型对个人贷款违约（逾期）行为进行二分类预测，并通过消融实验验证 pipeline 各组件的贡献。

## 实验结果（测试集 AUC）

| 模型 | AUC | 说明 |
|------|------|------|
| **XGBoost** 🏆 | **0.8679** | 最优 |
| LightGBM | 0.8627 | 训练最快 |
| GBDT | 0.8623 | sklearn 实现 |
| Logistic (L1) | 0.8600 | 最强可解释基线 |
| SVM (linear) | 0.8563 | 训练最慢 |
| Random Forest | 0.8296 | L1 后特征数太少 |
| Naive Bayes | 0.7935 | 独立性假设不成立 |

## 目录结构

```
.
├── 实验3代码/
│   ├── chapter3_data/                      # 原始 13 张 tsv/csv 表
│   ├── chapter3_data_handled/              # 中间产物 + 最终图
│   │   ├── figures/                        # 顶刊风格图 (300dpi, Wong 色板)
│   │   │   ├── A1_roc_seven_models.png     # 7 模型 ROC 对比
│   │   │   ├── A2_auc_bar_seven_models.png # AUC 排序条形图
│   │   │   ├── B1_sampling_ablation.png    # 采样策略消融
│   │   │   ├── B2_l1_fs_ablation.png       # L1 特征选择消融
│   │   │   └── B3_standardize_ablation.png # 标准化消融
│   │   ├── trainafter.csv                  # 最终训练宽表 (30000 × 157)
│   │   ├── supplementary_results.csv       # 全部补充实验 AUC 汇总
│   │   ├── LR.txt / _NB.txt / RF.txt / SVM.txt   # 单模型预测输出
│   │   └── *_LG.png / *_result.png         # 单模型 ROC 图
│   ├── chapter3_codes/
│   │   ├── table1_pre_treatment(train_test).py   # 主表预处理（含训练+测试）
│   │   ├── table2_pre_treat.py ~ table11_pre_treat.py  # 10 张征信扩展表预处理
│   │   ├── merge.py                        # 多表通过 REPORT_ID 合并
│   │   ├── pre_train.py / pre_train_Test.py # 合并后再做缺失值/无效列处理
│   │   ├── LR_NEW.py / NB1.py / RF.py / SVM.py   # 4 个基线模型
│   │   └── supp_experiments.py             # 补充实验 A + B（含 7 个模型与 3 组消融）
│   ├── run_all.py                          # 一键运行 baseline pipeline 入口
│   ├── build_final_report.py               # 报告生成脚本
│   └── 机器学习实验三_贷款违约行为预测_报告.docx   # 最终报告（10 章 / 9 表 / 11 图）
├── .gitignore
└── README.md
```

## 运行方式

### 环境依赖

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib python-docx xgboost lightgbm
```

### 一键运行 baseline pipeline（4 个基础模型，约 5-10 分钟）

```bash
cd 实验3代码
python3 run_all.py
```

会依次执行：11 张表的单表预处理 → 多表合并 → 二次预处理 → 4 个基线模型（LR / NB / RF / SVM）训练与评估，最后打印 4 个模型的 AUC。

### 补充实验（A 引入 GBDT 类模型 + B 三组消融，约 30-40 分钟）

```bash
python3 chapter3_codes/supp_experiments.py
```

会跑 7 个模型 + 22 次消融实验，输出 5 张顶刊风格图到 `chapter3_data_handled/figures/` 与一份 `supplementary_results.csv` 汇总。

### 重新生成报告

```bash
python3 build_final_report.py
```

会读取 `supplementary_results.csv` 与 `figures/` 下的图，生成最终的 docx 报告。

## 数据 pipeline 设计

1. **单表预处理**（table1_pre_treatment / table2 ~ table11_pre_treat.py）
   - 主表：身份证→性别 / WORK_PROVINCE→省份 / OneHot / Z-score 标准化
   - 扩展表：标称变量字典映射 → groupby(report_id).mean() → 独热编码 + StandardScaler
2. **多表合并**（merge.py）
   - 以 REPORT_ID 为键，左连接 table_1 与 table_2 ~ table_11
3. **合并后预处理**（pre_train.py / pre_train_Test.py）
   - 剔除 LOAN_DATE 与全空列
   - 用样本均值填补剩余缺失
4. **建模与评估**
   - 7:3 stratified split (random_state=0)
   - SMOTE + Tomek 不平衡处理
   - L1-Logistic 特征选择
   - 7 个模型训练 → 测试集 AUC

## 已修复的原始仓库 bug

1. 所有 19 个脚本中硬编码的 Windows 路径 `D:\文档\机器学习\实验3代码\...` → 改为相对/绝对 Linux 路径
2. `merge.py` 第 17 行：train 表自连接 → 改为依次 left-join table_2 ~ table_11
3. `merge.py` 第 32 行：构造 test_all 时误用 train+test → 改为以 test 表为左表
4. `pre_train.py` / `pre_train_Test.py` 中 `LOAN_DATE_x/_y` 依赖 → 因 merge bug 修复后只剩单列 `LOAN_DATE`，改为直接 drop
5. 4 个模型脚本误读 `testafter.csv`（无 Y 标签）→ 改为读 `trainafter.csv`

## 报告

完整报告见 `实验3代码/机器学习实验三_贷款违约行为预测_报告.docx`，共 9 章 / 9 表 / 11 图，涵盖：

1. 实验目的
2. 任务说明
3. 数据集介绍（13 张表的字段、规模、缺失值、类别分布）
4. 数据预处理（变量类型识别、空值处理、维归约、独热编码、标准化、多表合并）
5. 模型构建与训练（7 个模型的原理与代码）
6. 实验结果与分析（ROC 对比 + AUC 表 + 模型适用性讨论）
7. 消融实验（B1 采样 / B2 L1 特征选择 / B3 标准化）
8. 综合结论
9. 附录：代码与运行指南
