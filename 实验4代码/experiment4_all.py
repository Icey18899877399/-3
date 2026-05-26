"""
实验四：聚类算法与性能度量 —— 完整实验代码
任务 1-6 + 两种聚类算法比较（K-Means vs DBSCAN）

运行方式:
    python3 experiment4_all.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from itertools import combinations

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── NPG (Nature Publishing Group) 配色 — ggsci "nrc" 色板 ──
# 来源: https://nanx.me/ggsci/reference/pal_npg.html
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
       '#F39B7F', '#8491B4', '#91D1C2', '#DC0000',
       '#7E6148', '#B09C85']

# Okabe-Ito 色板 (Nature Methods 推荐, 色盲友好)
OI = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
      '#56B4E9', '#CC79A7', '#F0E442', '#000000']

PALETTE = NPG  # 主色板

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.grid': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'lines.linewidth': 1.4,
    'lines.markersize': 5,
    'patch.linewidth': 0.5,
})

# =====================================================================
# 加载 Iris 数据集
# =====================================================================
iris = load_iris()
X_raw = iris.data          # (150, 4)
y_true = iris.target       # 真实标签 0,1,2

print('=' * 70)
print('实验四：聚类算法与性能度量')
print('=' * 70)
print(f'数据集: Iris, 样本数={X_raw.shape[0]}, 特征数={X_raw.shape[1]}')
print(f'真实类别: {np.unique(y_true)}, 各类样本数: {np.bincount(y_true)}')
print()


# =====================================================================
# 任务 1: 计算 a, b, c, d
# =====================================================================
def compute_abcd(labels_pred, labels_true):
    """计算聚类评估中的 a, b, c, d 四个变量。

    a = |SS|: 在 C 中同簇且在 C* 中同簇的样本对数
    b = |SD|: 在 C 中同簇但在 C* 中不同簇的样本对数
    c = |DS|: 在 C 中不同簇但在 C* 中同簇的样本对数
    d = |DD|: 在 C 中不同簇且在 C* 中不同簇的样本对数
    """
    n = len(labels_pred)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = (labels_pred[i] == labels_pred[j])
            same_true = (labels_true[i] == labels_true[j])
            if same_pred and same_true:
                a += 1
            elif same_pred and not same_true:
                b += 1
            elif not same_pred and same_true:
                c += 1
            else:
                d += 1
    return a, b, c, d

print('=' * 70)
print('任务 1: 计算 a, b, c, d')
print('=' * 70)


# =====================================================================
# 任务 2: 计算 Jaccard 系数
# =====================================================================
def jaccard_coefficient(a, b, c):
    """JC = a / (a + b + c)"""
    return a / (a + b + c)

print('任务 2: Jaccard 系数公式: JC = a / (a + b + c)')
print()


# =====================================================================
# 任务 3: 距离公式 (闵可夫斯基距离)
# =====================================================================
def minkowski_distance(x, y, p=2):
    """闵可夫斯基距离: dist(x,y) = (sum|x_i - y_i|^p)^(1/p)
    p=1: 曼哈顿距离, p=2: 欧氏距离
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

print('=' * 70)
print('任务 3: 距离公式')
print('=' * 70)
x_demo = np.array([1.0, 2.0, 3.0])
y_demo = np.array([4.0, 5.0, 6.0])
print(f'示例向量: x={x_demo}, y={y_demo}')
print(f'  曼哈顿距离 (p=1): {minkowski_distance(x_demo, y_demo, p=1):.4f}')
print(f'  欧氏距离   (p=2): {minkowski_distance(x_demo, y_demo, p=2):.4f}')
print(f'  闵可夫斯基 (p=3): {minkowski_distance(x_demo, y_demo, p=3):.4f}')
print()


# =====================================================================
# 任务 4: DB 指数 (Davies-Bouldin Index)
# =====================================================================
def cluster_avg_dist(X_cluster):
    """簇内样本间的平均距离 avg(C)"""
    n = len(X_cluster)
    if n <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += minkowski_distance(X_cluster[i], X_cluster[j], p=2)
            count += 1
    return total / count

def cluster_diam(X_cluster):
    """簇内样本间的最远距离 diam(C)"""
    n = len(X_cluster)
    if n <= 1:
        return 0.0
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = minkowski_distance(X_cluster[i], X_cluster[j], p=2)
            if d > max_dist:
                max_dist = d
    return max_dist

def cluster_d_min(X_ci, X_cj):
    """簇 Ci 与 Cj 最近样本间的距离 d_min"""
    min_dist = float('inf')
    for xi in X_ci:
        for xj in X_cj:
            d = minkowski_distance(xi, xj, p=2)
            if d < min_dist:
                min_dist = d
    return min_dist

def cluster_d_cen(X_ci, X_cj):
    """簇 Ci 与簇 Cj 中心点间的距离 d_cen"""
    mu_i = np.mean(X_ci, axis=0)
    mu_j = np.mean(X_cj, axis=0)
    return minkowski_distance(mu_i, mu_j, p=2)

def davies_bouldin_index(X, labels):
    """DBI = (1/k) * sum_i max_{j!=i} (avg(Ci) + avg(Cj)) / d_cen(mu_i, mu_j)"""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    k = len(unique_labels)
    if k <= 1:
        return float('inf')

    clusters = {l: X[labels == l] for l in unique_labels}
    avg_dists = {l: cluster_avg_dist(clusters[l]) for l in unique_labels}

    dbi = 0.0
    for i in unique_labels:
        max_ratio = -1.0
        for j in unique_labels:
            if i == j:
                continue
            d_cen = cluster_d_cen(clusters[i], clusters[j])
            if d_cen == 0:
                ratio = float('inf')
            else:
                ratio = (avg_dists[i] + avg_dists[j]) / d_cen
            if ratio > max_ratio:
                max_ratio = ratio
        dbi += max_ratio
    return dbi / k


# =====================================================================
# 任务 5: VDM 与 MinkovDMp
# =====================================================================
def vdm_distance(x_val, y_val, attribute_idx, X_full, labels, p=2):
    """Value Difference Metric: 处理无序属性的距离度量
    VDM_p(a, b) = sum_i |m_{u,a,i}/m_{u,a} - m_{u,b,i}/m_{u,b}|^p
    """
    col = X_full[:, attribute_idx]
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)

    m_u_a = np.sum(col == x_val)
    m_u_b = np.sum(col == y_val)
    if m_u_a == 0 or m_u_b == 0:
        return 0.0

    dist = 0.0
    for ci in unique_clusters:
        mask_ci = (labels == ci)
        m_u_a_i = np.sum(col[mask_ci] == x_val)
        m_u_b_i = np.sum(col[mask_ci] == y_val)
        dist += abs(m_u_a_i / m_u_a - m_u_b_i / m_u_b) ** p
    return dist

def minkov_dm_p(xi, xj, n_continuous, X_full, labels, p=2):
    """MinkovDMp: 处理混合属性的距离度量
    对前 n_c 个连续属性用闵可夫斯基距离, 后面的用 VDM
    """
    continuous_part = np.sum(np.abs(xi[:n_continuous] - xj[:n_continuous]) ** p)

    vdm_part = 0.0
    for u in range(n_continuous, len(xi)):
        vdm_part += vdm_distance(xi[u], xj[u], u, X_full, labels, p=p)

    return (continuous_part + vdm_part) ** (1.0 / p)

print('=' * 70)
print('任务 5: VDM 与 MinkovDMp')
print('=' * 70)
demo_X = np.array([
    [1.0, 2.0, 0, 1],
    [1.5, 2.5, 1, 0],
    [3.0, 4.0, 0, 1],
    [3.5, 4.5, 1, 0],
    [5.0, 6.0, 0, 0],
    [5.5, 6.5, 1, 1],
])
demo_labels = np.array([0, 0, 1, 1, 2, 2])
d_mix = minkov_dm_p(demo_X[0], demo_X[2], n_continuous=2, X_full=demo_X,
                    labels=demo_labels, p=2)
print(f'示例混合属性距离 MinkovDM_2(x0, x2) = {d_mix:.4f}')
print()


# =====================================================================
# 任务 6: 归一化 (Min-Max Normalization)
# =====================================================================
def min_max_normalize(X):
    """Min-Max 归一化: x' = (x - x_min) / (x_max - x_min)"""
    X = np.asarray(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    return (X - X_min) / denom

print('=' * 70)
print('任务 6: Min-Max 归一化')
print('=' * 70)
X_normalized = min_max_normalize(X_raw)
print(f'归一化前 - 均值: {X_raw.mean(axis=0).round(3)}, 范围: [{X_raw.min():.1f}, {X_raw.max():.1f}]')
print(f'归一化后 - 均值: {X_normalized.mean(axis=0).round(3)}, 范围: [{X_normalized.min():.1f}, {X_normalized.max():.1f}]')
print()

# ── 可视化: 归一化前后对比 (violin + strip overlay) ──
feature_names = iris.feature_names
box_colors = [NPG[3], NPG[1], NPG[2], NPG[0]]  # 深蓝, 青, 绿, 红
short_feat = ['Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth']

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax_idx, (data, title, ylabel) in enumerate([
        (X_raw, 'Before Normalization', 'Value (raw)'),
        (X_normalized, 'After Min-Max Normalization', 'Value (normalized)')]):
    vparts = axes[ax_idx].violinplot(
        [data[:, i] for i in range(4)], positions=range(4),
        showmedians=False, showextrema=False)
    for i, body in enumerate(vparts['bodies']):
        body.set_facecolor(box_colors[i])
        body.set_edgecolor(box_colors[i])
        body.set_alpha(0.25)
    bp = axes[ax_idx].boxplot(
        [data[:, i] for i in range(4)], positions=range(4),
        widths=0.18, patch_artist=True,
        medianprops=dict(color='white', linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker='o', markersize=3, alpha=0.5,
                        markerfacecolor='#888888', markeredgecolor='none'))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(box_colors[i])
        patch.set_edgecolor(box_colors[i])
        patch.set_alpha(0.85)
    axes[ax_idx].set_xticks(range(4))
    axes[ax_idx].set_xticklabels(short_feat, fontsize=8.5)
    axes[ax_idx].set_title(title)
    axes[ax_idx].set_ylabel(ylabel)

plt.tight_layout(w_pad=3)
plt.savefig(os.path.join(FIG_DIR, 'normalization_comparison.png'))
plt.close()
print('图片已保存: normalization_comparison.png')


# =====================================================================
# 聚类算法比较: K-Means vs DBSCAN
# =====================================================================
print('\n' + '=' * 70)
print('聚类算法比较: K-Means vs DBSCAN')
print('=' * 70)

X_data = X_normalized.copy()

# ── K-Means 聚类 ──
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_data)

# ── DBSCAN 聚类 ──
dbscan = DBSCAN(eps=0.35, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_data)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = np.sum(labels_dbscan == -1)

print(f'\nK-Means: {len(np.unique(labels_kmeans))} 个簇')
print(f'DBSCAN:  {n_clusters_dbscan} 个簇, {n_noise_dbscan} 个噪声点')


# ── 计算各性能指标 ──
def evaluate_clustering(X, labels_pred, labels_true, method_name):
    """对聚类结果计算全部性能指标"""
    print(f'\n--- {method_name} 性能评估 ---')

    a, b, c, d = compute_abcd(labels_pred, labels_true)
    total_pairs = a + b + c + d
    print(f'  a (SS) = {a}')
    print(f'  b (SD) = {b}')
    print(f'  c (DS) = {c}')
    print(f'  d (DD) = {d}')
    print(f'  总样本对数 = {total_pairs}  (C(150,2) = {150*149//2})')

    jc = jaccard_coefficient(a, b, c)
    print(f'  Jaccard 系数 (JC) = {jc:.4f}')

    # FM 指数
    fmi = a / np.sqrt((a + b) * (a + c)) if (a + b) > 0 and (a + c) > 0 else 0
    print(f'  FM 指数 (FMI)     = {fmi:.4f}')

    # Rand 指数
    ri = (a + d) / total_pairs if total_pairs > 0 else 0
    print(f'  Rand 指数 (RI)    = {ri:.4f}')

    # DB 指数
    valid_labels = labels_pred[labels_pred >= 0]
    valid_X = X[labels_pred >= 0]
    dbi = davies_bouldin_index(valid_X, valid_labels)
    print(f'  DB 指数 (DBI)     = {dbi:.4f}  (越小越好)')

    return {
        'a': a, 'b': b, 'c': c, 'd': d,
        'JC': jc, 'FMI': fmi, 'RI': ri, 'DBI': dbi
    }

metrics_km = evaluate_clustering(X_data, labels_kmeans, y_true, 'K-Means')
metrics_db = evaluate_clustering(X_data, labels_dbscan, y_true, 'DBSCAN')


# =====================================================================
# 可视化 1: PCA 降维后的聚类结果对比
# =====================================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_data)

scatter_c = [NPG[0], NPG[2], NPG[3]]  # 红, 绿, 深蓝
scatter_m = ['o', 's', '^']

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
species = ['Setosa', 'Versicolor', 'Virginica']

for idx, label in enumerate(np.unique(y_true)):
    mask = y_true == label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=scatter_c[idx],
                    marker=scatter_m[idx], label=species[idx],
                    s=28, alpha=0.82, edgecolors='white', linewidths=0.4)
axes[0].set_title('(a) Ground Truth')
axes[0].set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].legend(frameon=True, fancybox=False, edgecolor='#d0d0d0',
               fontsize=7.5, handletextpad=0.3, borderpad=0.4)

for idx, label in enumerate(np.unique(labels_kmeans)):
    mask = labels_kmeans == label
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=scatter_c[idx],
                    marker=scatter_m[idx], label=f'Cluster {label}',
                    s=28, alpha=0.82, edgecolors='white', linewidths=0.4)
axes[1].set_title('(b) K-Means (k = 3)')
axes[1].set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].legend(frameon=True, fancybox=False, edgecolor='#d0d0d0',
               fontsize=7.5, handletextpad=0.3, borderpad=0.4)

unique_db = sorted(set(labels_dbscan))
db_colors = [NPG[0], NPG[2], NPG[3], NPG[1]]
for idx, label in enumerate(unique_db):
    mask = labels_dbscan == label
    if label == -1:
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], c='#AAAAAA',
                        label='Noise', s=15, alpha=0.55, marker='x', linewidths=0.8)
    else:
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=db_colors[idx % len(db_colors)],
                        marker=scatter_m[idx % 3], label=f'Cluster {label}',
                        s=28, alpha=0.82, edgecolors='white', linewidths=0.4)
axes[2].set_title(r'(c) DBSCAN ($\epsilon$=0.35)')
axes[2].set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].legend(frameon=True, fancybox=False, edgecolor='#d0d0d0',
               fontsize=7.5, handletextpad=0.3, borderpad=0.4)

plt.tight_layout(w_pad=1.5)
plt.savefig(os.path.join(FIG_DIR, 'clustering_comparison_pca.png'))
plt.close()
print('\n图片已保存: clustering_comparison_pca.png')


# =====================================================================
# 可视化 2: 性能指标对比条形图 (水平)
# =====================================================================
all_names = ['JC', 'FMI', 'RI', 'DBI']
km_vals = [metrics_km[m] for m in all_names]
db_vals = [metrics_db[m] for m in all_names]

fig, ax = plt.subplots(figsize=(6.5, 3.8))
y_pos = np.arange(len(all_names))
h = 0.32
c_km, c_db = NPG[3], NPG[0]  # 深蓝 vs 红

bars_km = ax.barh(y_pos + h/2, km_vals, h, color=c_km, label='K-Means',
                   edgecolor='white', linewidth=0.4)
bars_db = ax.barh(y_pos - h/2, db_vals, h, color=c_db, label='DBSCAN',
                   edgecolor='white', linewidth=0.4)
for bar, val in zip(bars_km, km_vals):
    ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5, color=c_km, fontweight='bold')
for bar, val in zip(bars_db, db_vals):
    ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5, color=c_db, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(all_names)
ax.set_xlabel('Score')
ax.set_xlim(0, 1.42)
ax.legend(frameon=True, fancybox=False, edgecolor='#d0d0d0',
          loc='upper right')

ax.axhline(y=2.5, color='#cccccc', linewidth=0.5, linestyle='--')
ax.set_yticks(y_pos)
labels_txt = ['JC\n(higher=better)', 'FMI\n(higher=better)',
              'RI\n(higher=better)', 'DBI\n(lower=better)']
ax.set_yticklabels(labels_txt, fontsize=8.5)

ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'metrics_comparison.png'))
plt.close()
print('图片已保存: metrics_comparison.png')


# =====================================================================
# 可视化 3: 距离函数对比 (viridis 色系)
# =====================================================================
from scipy.spatial.distance import pdist, squareform
D_euc = squareform(pdist(X_data[:30], metric='euclidean'))
D_man = squareform(pdist(X_data[:30], metric='cityblock'))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for ax, D, title in zip(axes, [D_euc, D_man],
                          ['(a) Euclidean Distance (p = 2)',
                           '(b) Manhattan Distance (p = 1)']):
    im = ax.imshow(D, cmap='viridis', aspect='equal', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.outline.set_linewidth(0.4)

plt.tight_layout(w_pad=2.5)
plt.savefig(os.path.join(FIG_DIR, 'distance_heatmaps.png'))
plt.close()
print('图片已保存: distance_heatmaps.png')


# =====================================================================
# 可视化 4: 不同 k 值下 K-Means 的肘部法则
# =====================================================================
k_range = range(2, 9)
inertias = []
dbis_k = []
jcs_k = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred = km.fit_predict(X_data)
    inertias.append(km.inertia_)
    dbis_k.append(davies_bouldin_index(X_data, pred))
    a, b, c, _ = compute_abcd(pred, y_true)
    jcs_k.append(jaccard_coefficient(a, b, c))

line_cfgs = [
    ('o-', NPG[3], '(a) Elbow Method',           'Inertia (SSE)'),
    ('s-', NPG[0], '(b) Davies-Bouldin Index',   'DBI'),
    ('D-', NPG[2], '(c) Jaccard Coefficient',    'JC'),
]
data_series = [inertias, dbis_k, jcs_k]
opt_k = 3

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (fmt, col, title, ylabel), vals in zip(axes, line_cfgs, data_series):
    ax.plot(list(k_range), vals, fmt, color=col, linewidth=1.5,
            markersize=5.5, markerfacecolor='white', markeredgewidth=1.4,
            markeredgecolor=col)
    ax.axvline(x=opt_k, color='#bbbbbb', linewidth=0.7, linestyle='--', zorder=0)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout(w_pad=2)
plt.savefig(os.path.join(FIG_DIR, 'kmeans_k_selection.png'))
plt.close()
print('图片已保存: kmeans_k_selection.png')


# =====================================================================
# 可视化 5: 特征散点矩阵 (聚类结果)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
feat_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
short_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
sc_colors = [NPG[0], NPG[2], NPG[3]]
sc_markers = ['o', 's', '^']

for idx, (fi, fj) in enumerate(feat_pairs):
    ax = axes[idx // 3][idx % 3]
    for cl in range(3):
        mask = labels_kmeans == cl
        ax.scatter(X_data[mask, fi], X_data[mask, fj],
                   c=sc_colors[cl], marker=sc_markers[cl],
                   s=20, alpha=0.75, label=f'C{cl}',
                   edgecolors='white', linewidths=0.3)
    ax.set_xlabel(short_names[fi], fontsize=8.5)
    ax.set_ylabel(short_names[fj], fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    if idx == 0:
        ax.legend(frameon=True, fancybox=False, edgecolor='#d0d0d0',
                  fontsize=7, loc='best', handletextpad=0.2)

plt.tight_layout(h_pad=2, w_pad=1.5)
plt.savefig(os.path.join(FIG_DIR, 'kmeans_scatter_matrix.png'))
plt.close()
print('图片已保存: kmeans_scatter_matrix.png')


# =====================================================================
# 可视化 6: a, b, c, d 对比 — 环形图 (donut chart)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
pie_labels = ['a (SS)', 'b (SD)', 'c (DS)', 'd (DD)']
pie_colors = [NPG[2], NPG[0], NPG[4], NPG[3]]  # 绿, 红, 淡橙, 深蓝

km_abcd = [metrics_km['a'], metrics_km['b'], metrics_km['c'], metrics_km['d']]
db_abcd = [metrics_db['a'], metrics_db['b'], metrics_db['c'], metrics_db['d']]

for ax, data, title in zip(axes, [km_abcd, db_abcd], ['(a) K-Means', '(b) DBSCAN']):
    wedges, texts, autotexts = ax.pie(
        data, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(7.5)
        t.set_color('#333333')
    for t in texts:
        t.set_fontsize(8)
    ax.set_title(title, pad=10)

plt.tight_layout(w_pad=2)
plt.savefig(os.path.join(FIG_DIR, 'abcd_pie_chart.png'))
plt.close()
print('图片已保存: abcd_pie_chart.png')


# =====================================================================
# 输出总结
# =====================================================================
print('\n' + '=' * 70)
print('实验结果总结')
print('=' * 70)
print(f'\n{"指标":<20s} {"K-Means":>10s} {"DBSCAN":>10s} {"更优":>10s}')
print('-' * 52)
for m in ['JC', 'FMI', 'RI']:
    better = 'K-Means' if metrics_km[m] >= metrics_db[m] else 'DBSCAN'
    print(f'{m:<20s} {metrics_km[m]:>10.4f} {metrics_db[m]:>10.4f} {better:>10s}')
m = 'DBI'
better = 'K-Means' if metrics_km[m] <= metrics_db[m] else 'DBSCAN'
print(f'{m:<20s} {metrics_km[m]:>10.4f} {metrics_db[m]:>10.4f} {better:>10s}')

print('\n结论: K-Means 在 Iris 数据集上的聚类效果优于 DBSCAN。')
print('原因: Iris 数据集的类别呈凸形分布, 适合 K-Means 的球形假设;')
print('      DBSCAN 基于密度, 在类别边界模糊时容易将相邻类合并或产生噪声点。')
print('\n所有图片已保存至 figures/ 目录')
print('实验完成!')
