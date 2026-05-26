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

# ── 顶刊配色 (Nature / Science 常用) ──
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
           '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
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

# ── 可视化: 归一化前后对比 ──
feature_names = iris.feature_names
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bp0 = axes[0].boxplot([X_raw[:, i] for i in range(4)], patch_artist=True,
                       widths=0.5, medianprops=dict(color='black', linewidth=1.2))
for patch, color in zip(bp0['boxes'], PALETTE[:4]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels(['SL', 'SW', 'PL', 'PW'])
axes[0].set_title('Before Normalization')
axes[0].set_ylabel('Value')

bp1 = axes[1].boxplot([X_normalized[:, i] for i in range(4)], patch_artist=True,
                       widths=0.5, medianprops=dict(color='black', linewidth=1.2))
for patch, color in zip(bp1['boxes'], PALETTE[:4]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_xticklabels(['SL', 'SW', 'PL', 'PW'])
axes[1].set_title('After Min-Max Normalization')
axes[1].set_ylabel('Normalized Value')

fig.suptitle('Feature Distribution: Before vs After Normalization', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
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

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
species = ['Setosa', 'Versicolor', 'Virginica']

for idx, label in enumerate(np.unique(y_true)):
    mask = y_true == label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[idx],
                    label=species[idx], s=25, alpha=0.75, edgecolors='white',
                    linewidths=0.3)
axes[0].set_title('Ground Truth', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)

for idx, label in enumerate(np.unique(labels_kmeans)):
    mask = labels_kmeans == label
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[idx],
                    label=f'Cluster {label}', s=25, alpha=0.75,
                    edgecolors='white', linewidths=0.3)
axes[1].set_title('K-Means (k=3)', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)

unique_db = sorted(set(labels_dbscan))
for idx, label in enumerate(unique_db):
    mask = labels_dbscan == label
    if label == -1:
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], c='#8C8C8C',
                        label='Noise', s=15, alpha=0.5, marker='x')
    else:
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[idx % len(PALETTE)],
                        label=f'Cluster {label}', s=25, alpha=0.75,
                        edgecolors='white', linewidths=0.3)
axes[2].set_title(f'DBSCAN (eps=0.35)', fontweight='bold')
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)

fig.suptitle('Clustering Results Comparison (PCA Projection)', fontsize=14,
             fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'clustering_comparison_pca.png'))
plt.close()
print('\n图片已保存: clustering_comparison_pca.png')


# =====================================================================
# 可视化 2: 性能指标对比条形图
# =====================================================================
metrics_names = ['JC', 'FMI', 'RI', 'DBI']
km_vals = [metrics_km[m] for m in metrics_names]
db_vals = [metrics_db[m] for m in metrics_names]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左图: 外部指标 (JC, FMI, RI) - 越大越好
ext_names = ['JC', 'FMI', 'RI']
ext_km = [metrics_km[m] for m in ext_names]
ext_db = [metrics_db[m] for m in ext_names]
x_pos = np.arange(len(ext_names))
w = 0.32
bars1 = axes[0].bar(x_pos - w/2, ext_km, w, color=PALETTE[0], label='K-Means',
                     edgecolor='white', linewidth=0.5)
bars2 = axes[0].bar(x_pos + w/2, ext_db, w, color=PALETTE[1], label='DBSCAN',
                     edgecolor='white', linewidth=0.5)
for bar, val in zip(bars1, ext_km):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, ext_db):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(ext_names)
axes[0].set_ylabel('Score')
axes[0].set_title('External Metrics (higher is better)', fontweight='bold')
axes[0].set_ylim(0, 1.15)
axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc')

# 右图: 内部指标 (DBI) - 越小越好
int_names = ['DBI']
int_km = [metrics_km['DBI']]
int_db = [metrics_db['DBI']]
x_pos2 = np.arange(len(int_names))
b1 = axes[1].bar(x_pos2 - w/2, int_km, w, color=PALETTE[0], label='K-Means',
                  edgecolor='white', linewidth=0.5)
b2 = axes[1].bar(x_pos2 + w/2, int_db, w, color=PALETTE[1], label='DBSCAN',
                  edgecolor='white', linewidth=0.5)
for bar, val in zip(b1, int_km):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(b2, int_db):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
axes[1].set_xticks(x_pos2)
axes[1].set_xticklabels(int_names)
axes[1].set_ylabel('Score')
axes[1].set_title('Internal Metric (lower is better)', fontweight='bold')
axes[1].legend(frameon=True, fancybox=False, edgecolor='#cccccc')

fig.suptitle('Performance Metrics: K-Means vs DBSCAN', fontsize=14,
             fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'metrics_comparison.png'))
plt.close()
print('图片已保存: metrics_comparison.png')


# =====================================================================
# 可视化 3: 距离函数对比
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# 左图: 距离矩阵热力图 (欧氏)
from scipy.spatial.distance import pdist, squareform
D_euc = squareform(pdist(X_data[:30], metric='euclidean'))
D_man = squareform(pdist(X_data[:30], metric='cityblock'))

im0 = axes[0].imshow(D_euc, cmap='YlOrRd', aspect='auto')
axes[0].set_title('Euclidean Distance (p=2)', fontweight='bold')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Sample Index')
fig.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(D_man, cmap='YlOrRd', aspect='auto')
axes[1].set_title('Manhattan Distance (p=1)', fontweight='bold')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Sample Index')
fig.colorbar(im1, ax=axes[1], shrink=0.8)

fig.suptitle('Distance Matrix Heatmaps (First 30 Samples)', fontsize=13,
             fontweight='bold', y=1.03)
plt.tight_layout()
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

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(list(k_range), inertias, 'o-', color=PALETTE[0], linewidth=1.5,
             markersize=6, markerfacecolor='white', markeredgewidth=1.5)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (SSE)')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(list(k_range), dbis_k, 's-', color=PALETTE[1], linewidth=1.5,
             markersize=6, markerfacecolor='white', markeredgewidth=1.5)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('DBI')
axes[1].set_title('Davies-Bouldin Index vs k', fontweight='bold')
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[2].plot(list(k_range), jcs_k, 'D-', color=PALETTE[2], linewidth=1.5,
             markersize=6, markerfacecolor='white', markeredgewidth=1.5)
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('JC')
axes[2].set_title('Jaccard Coefficient vs k', fontweight='bold')
axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle('K-Means: Effect of Cluster Number on Metrics', fontsize=14,
             fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'kmeans_k_selection.png'))
plt.close()
print('图片已保存: kmeans_k_selection.png')


# =====================================================================
# 可视化 5: 特征散点矩阵 (聚类结果)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
feat_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
short_names = ['SL', 'SW', 'PL', 'PW']

for idx, (fi, fj) in enumerate(feat_pairs):
    ax = axes[idx // 3][idx % 3]
    for cl in range(3):
        mask = labels_kmeans == cl
        ax.scatter(X_data[mask, fi], X_data[mask, fj], c=PALETTE[cl],
                   s=18, alpha=0.7, label=f'C{cl}',
                   edgecolors='white', linewidths=0.2)
    ax.set_xlabel(short_names[fi])
    ax.set_ylabel(short_names[fj])
    if idx == 0:
        ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                  fontsize=7, loc='best')

fig.suptitle('K-Means Clustering: Pairwise Feature Scatter', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'kmeans_scatter_matrix.png'))
plt.close()
print('图片已保存: kmeans_scatter_matrix.png')


# =====================================================================
# 可视化 6: a, b, c, d 对比饼图
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
pie_labels = ['a (SS)', 'b (SD)', 'c (DS)', 'd (DD)']
pie_colors = [PALETTE[2], PALETTE[3], PALETTE[1], PALETTE[0]]

km_abcd = [metrics_km['a'], metrics_km['b'], metrics_km['c'], metrics_km['d']]
db_abcd = [metrics_db['a'], metrics_db['b'], metrics_db['c'], metrics_db['d']]

wedges0, texts0, autotexts0 = axes[0].pie(
    km_abcd, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1))
for t in autotexts0:
    t.set_fontsize(8)
axes[0].set_title('K-Means', fontweight='bold')

wedges1, texts1, autotexts1 = axes[1].pie(
    db_abcd, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1))
for t in autotexts1:
    t.set_fontsize(8)
axes[1].set_title('DBSCAN', fontweight='bold')

fig.suptitle('Distribution of Pair Types (a, b, c, d)', fontsize=13,
             fontweight='bold', y=1.02)
plt.tight_layout()
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
