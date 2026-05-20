"""一键运行整个 pipeline：单表预处理 → 合并 → 二次预处理 → 4 个基线模型。

运行方式：
    cd 实验3代码
    python3 run_all.py

如果要继续跑补充实验（A + B），追加执行：
    python3 chapter3_codes/supp_experiments.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(ROOT, 'chapter3_codes')


def run(script):
    print(f'\n{"=" * 60}\n>> Running {script}\n{"=" * 60}')
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(CODES, script)],
        cwd=CODES, capture_output=True, text=True)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f'   [FAIL in {dt:.1f}s]')
        print(result.stdout[-2000:])
        print('STDERR:', result.stderr[-2000:])
        sys.exit(result.returncode)
    print(f'   [OK in {dt:.1f}s]')
    if result.stdout.strip():
        # 仅打印最后几行
        print('   ' + '\n   '.join(result.stdout.strip().splitlines()[-3:]))


# ---------- 1. 单表预处理 ----------
single_table_scripts = [
    'table1_pre_treatment(train_test).py',
    'table2_pre_treat.py',
    'table3_pre_treat.py',
    'table4_pre_treat.py',
    'table5_pre_treat.py',
    'table6_pre_treat.py',
    'table7_pre_treat.py',
    'table8_pre_treat.py',
    'table9_pre_treat.py',
    'table10_pre_treat.py',
    'table11_pre_treat.py',
]
for s in single_table_scripts:
    run(s)

# ---------- 2. 多表合并 ----------
run('merge.py')

# ---------- 3. 合并后预处理 ----------
run('pre_train.py')
run('pre_train_Test.py')

# ---------- 4. 四个基线模型 ----------
for s in ['LR_NEW.py', 'NB1.py', 'RF.py', 'SVM.py']:
    run(s)

# ---------- 5. 汇总 AUC ----------
import numpy as np
from sklearn.metrics import roc_auc_score
print(f'\n{"=" * 60}\n>> Final AUC summary\n{"=" * 60}')
HANDLED = os.path.join(ROOT, 'chapter3_data_handled')
for name, f in [('Logistic (L1)', 'LR.txt'),
                ('Naive Bayes',   '_NB.txt'),
                ('Random Forest', 'RF.txt'),
                ('SVM (linear)',  'SVM.txt')]:
    p = os.path.join(HANDLED, f)
    if os.path.exists(p):
        d = np.loadtxt(p)
        print(f'  {name:<18s}  AUC = {roc_auc_score(d[:, 2], d[:, 1]):.4f}')

print('\nBaseline pipeline finished. To run supplementary experiments:')
print('  python3 chapter3_codes/supp_experiments.py')
print('To regenerate report:')
print('  python3 build_final_report.py')
