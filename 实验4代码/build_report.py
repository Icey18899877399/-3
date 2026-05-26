"""生成实验四的完整实验报告 (Word .docx) — Wine 数据集"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
OUT = os.path.join(ROOT, '机器学习实验四_聚类算法与性能度量_报告.docx')

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def H(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.color.rgb = RGBColor(0, 0, 0)


def P(text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r.font.size = Pt(11)
    if bold:
        r.bold = True


def Code(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = 'Consolas'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r.font.size = Pt(9)


def T(rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Light Grid Accent 1'
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = str(cell)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                    r.font.size = Pt(10)
                    if i == 0:
                        r.bold = True


def Pic(path, width_in=5.0):
    doc.add_picture(path, width=Inches(width_in))


# ====================== 标题 ======================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run('机器学习实验四：聚类算法与性能度量')
r.bold = True
r.font.size = Pt(16)
r.font.name = 'Times New Roman'
r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run('——基于 Wine 数据集的 K-Means 与 DBSCAN 聚类比较')
r.font.size = Pt(12)
r.font.name = 'Times New Roman'
r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


# ====================== 一、实验目的 ======================
H('一、实验目的', 1)
P('1. 理解聚类分析的基本概念，掌握外部指标（Jaccard 系数 JC、FM 指数 FMI、Rand 指数 RI）和内部指标（Davies-Bouldin 指数 DBI）的定义与计算方法；')
P('2. 掌握闵可夫斯基距离族（p=1 曼哈顿距离、p=2 欧氏距离）的编程实现，理解 VDM（Value Difference Metric）处理无序属性的原理，以及 MinkovDMp 处理混合属性的距离度量；')
P('3. 理解数据归一化的必要性，掌握 Min-Max 归一化方法的编程实现，并通过可视化对比归一化前后的特征分布；')
P('4. 熟练运用 K-Means 和 DBSCAN 两种典型聚类算法，并在 Wine 数据集上进行聚类实验，通过多种性能度量指标对比分析两种算法的优劣。')


# ====================== 二、实验内容 ======================
H('二、实验内容', 1)

H('2.1 任务概述', 2)
P('本实验共包含 6 个核心任务和 1 个综合比较任务：')
P('任务 1：编程计算聚类评估变量 a, b, c, d（SS, SD, DS, DD 四类样本对）；')
P('任务 2：基于 a, b, c 编程计算 Jaccard 系数（JC = a/(a+b+c)）；')
P('任务 3：编程实现闵可夫斯基距离公式，分别计算 p=1（曼哈顿距离）和 p=2（欧氏距离）；')
P('任务 4：编程计算 Davies-Bouldin 指数（DBI），包括簇内平均距离 avg(C)、簇内最远距离 diam(C)、簇间最近距离 d_min、簇间中心距离 d_cen；')
P('任务 5：编程实现 VDM（Value Difference Metric）处理无序属性距离，以及 MinkovDMp 处理混合属性距离；')
P('任务 6：编程实现 Min-Max 归一化，并可视化归一化前后的数据分布；')
P('综合任务：采用 K-Means 和 DBSCAN 两种聚类算法在 Wine 数据集上进行聚类，计算各性能度量指标并比较。')

H('2.2 数据集介绍', 2)
P('本实验采用 UCI Wine（葡萄酒）数据集，该数据集来源于意大利同一地区三个不同品种（cultivar）的葡萄酒化学成分分析，共 178 个样本、13 个连续特征、3 个品种类别。')
P('表 1  Wine 数据集基本信息')
T([
    ['属性', '值'],
    ['样本数', '178'],
    ['特征数', '13'],
    ['类别数', '3（品种 0 / 品种 1 / 品种 2）'],
    ['各类样本数', '59 / 71 / 48'],
    ['特征类型', '连续型（比率属性）'],
    ['来源', 'sklearn.datasets.load_wine()'],
])

P('Wine 数据集的 13 个特征及其量纲如下：')
P('表 2  Wine 数据集特征说明')
T([
    ['序号', '特征名', '典型范围', '说明'],
    ['1', 'Alcohol', '11.0 - 14.8', '酒精度'],
    ['2', 'Malic Acid', '0.7 - 5.8', '苹果酸含量'],
    ['3', 'Ash', '1.4 - 3.2', '灰分'],
    ['4', 'Alcalinity of Ash', '10.6 - 30.0', '灰分碱度'],
    ['5', 'Magnesium', '70 - 162', '镁含量'],
    ['6', 'Total Phenols', '1.0 - 3.9', '总酚'],
    ['7', 'Flavanoids', '0.3 - 5.1', '类黄酮'],
    ['8', 'Nonflavanoid Phenols', '0.1 - 0.7', '非类黄酮酚'],
    ['9', 'Proanthocyanins', '0.4 - 3.6', '原花青素'],
    ['10', 'Color Intensity', '1.3 - 13.0', '色度'],
    ['11', 'Hue', '0.5 - 1.7', '色调'],
    ['12', 'OD280/OD315', '1.3 - 4.0', '稀释葡萄酒的OD280/OD315'],
    ['13', 'Proline', '278 - 1680', '脯氨酸含量'],
])
P('注意：特征之间量纲差异极大，例如 Proline 的范围为 278-1680，而 Hue 仅为 0.5-1.7。这使得归一化在本数据集上非常关键，能够直观展示归一化的必要性。')

H('2.3 关键公式', 2)
P('(1) 聚类评估变量 a, b, c, d：')
P('  a = |SS|: 预测同簇 & 真实同簇; b = |SD|: 预测同簇 & 真实不同簇;')
P('  c = |DS|: 预测不同簇 & 真实同簇; d = |DD|: 预测不同簇 & 真实不同簇。')
P('(2) Jaccard 系数: JC = a / (a + b + c)')
P('(3) FM 指数: FMI = a / sqrt((a+b)(a+c))')
P('(4) Rand 指数: RI = (a + d) / (a + b + c + d)')
P('(5) 闵可夫斯基距离: dist(x_i, x_j) = (sum|x_{iu} - x_{ju}|^p)^{1/p}, p=1 曼哈顿, p=2 欧氏')
P('(6) DBI = (1/k) * sum_{i=1}^{k} max_{j!=i} (avg(C_i) + avg(C_j)) / d_cen(mu_i, mu_j), 越小越好')
P('(7) VDM_p(a, b) = sum_{i=1}^{k} |m_{u,a,i}/m_{u,a} - m_{u,b,i}/m_{u,b}|^p')
P('(8) MinkovDM_p: 连续属性用闵可夫斯基 + 无序属性用 VDM')
P('(9) Min-Max 归一化: x\' = (x - x_min) / (x_max - x_min)')


# ====================== 三、实验结果 ======================
H('三、实验结果与分析', 1)

H('3.1 任务 1-2：a, b, c, d 与 Jaccard 系数', 2)
P('对 Wine 数据集分别使用 K-Means (k=3) 和 DBSCAN (eps=0.48, min_samples=5) 进行聚类，与真实标签对比计算 a, b, c, d。')
P('表 3  K-Means 与 DBSCAN 的 a, b, c, d 计算结果')
T([
    ['变量', 'K-Means', 'DBSCAN', '含义'],
    ['a (SS)', '4752', '3792', '预测同簇 & 真实同簇'],
    ['b (SD)', '454', '2557', '预测同簇 & 真实不同簇'],
    ['c (DS)', '572', '1532', '预测不同簇 & 真实同簇'],
    ['d (DD)', '9975', '7872', '预测不同簇 & 真实不同簇'],
    ['总对数', '15753', '15753', 'C(178,2) = 15753'],
])

P('图 1  K-Means 与 DBSCAN 的 a, b, c, d 分布环形图')
Pic(os.path.join(FIG_DIR, 'abcd_pie_chart.png'), 5.5)
P('分析：K-Means 的 a (SS) 占比 30.2%，d (DD) 占比 63.3%，说明绝大部分样本对被正确聚合或正确分离，错误对 (b+c) 仅占 6.5%。DBSCAN 由于只找到 2 个簇且有 34 个噪声点，b (SD) 显著增大至 16.2%，表明大量不同类的样本被错误合并。')

H('3.2 任务 3：距离函数验证', 2)
P('对示例向量 x = [1, 2, 3], y = [4, 5, 6] 分别计算不同距离：')
P('表 4  距离函数计算结果')
T([
    ['距离类型', '公式', '计算结果'],
    ['曼哈顿距离 (p=1)', '|1-4|+|2-5|+|3-6|', '9.0000'],
    ['欧氏距离 (p=2)', 'sqrt((1-4)^2+(2-5)^2+(3-6)^2)', '5.1962'],
    ['闵可夫斯基 (p=3)', '(|1-4|^3+|2-5|^3+|3-6|^3)^{1/3}', '4.3267'],
])

P('图 2  Wine 数据集前 30 个样本的距离矩阵热力图')
Pic(os.path.join(FIG_DIR, 'distance_heatmaps.png'), 5.5)
P('分析：热力图清楚地展示了样本间的距离结构。对角线为 0（自身距离），同类样本（前 30 个属于品种 0）之间距离较小（暗色），与不同品种样本的距离较大（亮色）。曼哈顿距离由于对各维度差值线性求和，绝对数值更大。')

H('3.3 任务 4：DB 指数计算', 2)
P('表 5  两种算法的 DBI 对比')
T([
    ['算法', 'DBI', '评价'],
    ['K-Means (k=3)', '1.8782', '簇间分离度和簇内紧密度适中'],
    ['DBSCAN (eps=0.48)', '1.5614', 'DBI 较低, 但仅有 2 个簇'],
])
P('分析：DBSCAN 的 DBI 较低（1.56 vs 1.88），但这是因为它只发现 2 个簇（合并了两类），簇间距离自然增大。结合外部指标可以看出，DBI 较低并不意味着聚类效果更好，需综合评估。')

H('3.4 任务 5：VDM 与 MinkovDMp', 2)
P('VDM 用于处理无序属性，MinkovDMp 将连续属性闵可夫斯基距离与无序属性 VDM 统一到同一框架。')
P('使用自构造混合属性数据验证：MinkovDM_2(x_0, x_2) = 2.8284，综合了连续与无序属性的距离。')

H('3.5 任务 6：归一化', 2)
P('归一化的定义：将不同量纲、不同取值范围的特征变换到统一尺度的过程。')
P('为什么采用归一化：')
P('  (1) Wine 数据集量纲差异巨大：Proline 范围 [278, 1680]，而 Hue 仅为 [0.48, 1.71]，相差近 1000 倍。不归一化时 Proline 会完全主导距离计算；')
P('  (2) K-Means 等基于距离的算法在归一化后收敛更快、聚类质量更高；')
P('  (3) 使各特征对聚类的贡献更加公平均衡。')
P('本实验采用 Min-Max 归一化: x\' = (x - x_min) / (x_max - x_min)')

P('表 6  归一化前后的特征统计对比（选取 4 个量纲差异大的特征）')
T([
    ['特征', '归一化前均值', '归一化前范围', '归一化后均值', '归一化后范围'],
    ['Alcohol', '13.00', '[11.0, 14.8]', '0.519', '[0, 1]'],
    ['Malic Acid', '2.34', '[0.7, 5.8]', '0.315', '[0, 1]'],
    ['Flavanoids', '2.03', '[0.3, 5.1]', '0.356', '[0, 1]'],
    ['Proline', '746.89', '[278, 1680]', '0.334', '[0, 1]'],
])

P('图 3  归一化前后特征分布对比（violin + box plot）')
Pic(os.path.join(FIG_DIR, 'normalization_comparison.png'), 5.5)
P('分析：归一化前（左图）Proline 的取值范围高达 278-1680，完全压缩了 Alcohol (11-15)、Malic Acid (0.7-5.8)、Flavanoids (0.3-5.1) 的视觉空间。这意味着未归一化时距离计算几乎完全由 Proline 决定，其他特征的信息被淹没。归一化后（右图）所有特征统一到 [0, 1]，各特征的分布形态得以清晰呈现。')

H('3.6 聚类算法比较：K-Means vs DBSCAN', 2)

P('表 7  K-Means 与 DBSCAN 全部性能指标汇总')
T([
    ['指标', 'K-Means', 'DBSCAN', '更优', '说明'],
    ['簇数', '3', '2 (+34 noise)', 'K-Means', '真实类别数为 3'],
    ['JC (Jaccard)', '0.8224', '0.4812', 'K-Means', '越大越好'],
    ['FMI', '0.9026', '0.6522', 'K-Means', '越大越好'],
    ['RI (Rand)', '0.9349', '0.7404', 'K-Means', '越大越好'],
    ['DBI', '1.8782', '1.5614', 'DBSCAN', '越小越好'],
])

P('图 4  PCA 降维后的聚类结果可视化')
Pic(os.path.join(FIG_DIR, 'clustering_comparison_pca.png'), 6.0)
P('分析：')
P('  (1) K-Means 将 178 个样本划分为 3 个簇，与真实标签高度吻合（JC=0.822），仅在品种 1 和品种 2 的边界存在少量错分；')
P('  (2) DBSCAN 只发现 2 个簇（将品种 0 和品种 1 的部分样本合并），并将 34 个样本标记为噪声（灰色 x），这是因为 Wine 数据集在 13 维空间中各类的密度分布不均匀，DBSCAN 难以用统一的 eps 参数适配所有类别。')

P('图 5  性能指标对比柱状图')
Pic(os.path.join(FIG_DIR, 'metrics_comparison.png'), 5.5)

P('图 6  K-Means 在不同 k 值下的指标变化')
Pic(os.path.join(FIG_DIR, 'kmeans_k_selection.png'), 6.0)
P('分析：肘部法则（左图）在 k=3 处 Inertia 下降速率放缓形成"肘部"；DBI 在 k=3 时取得局部最小值；JC 在 k=3 时达到峰值。三个指标一致表明 k=3 是最优簇数，与真实品种数一致。')

P('图 7  K-Means 聚类的特征两两散点图')
Pic(os.path.join(FIG_DIR, 'kmeans_scatter_matrix.png'), 5.5)
P('分析：从散点图可以看出，Flavanoids vs OD280/OD315 以及 Alcohol vs Proline 这两对特征对三个品种有较强的判别力，簇间分离清晰。Color Intensity 在不同品种间也呈现明显差异。')


# ====================== 四、实验总结 ======================
H('四、实验总结', 1)

P('1. K-Means 显著优于 DBSCAN：在 Wine 数据集上，K-Means 在全部三个外部指标（JC=0.822, FMI=0.903, RI=0.935）上大幅领先 DBSCAN（JC=0.481, FMI=0.652, RI=0.740），说明 K-Means 的聚类结果与真实品种标签高度一致。')

P('2. 归一化的关键作用：Wine 数据集的特征量纲差异极大（Proline 范围 278-1680 vs Hue 范围 0.5-1.7），不归一化会导致 Proline 完全主导距离计算。Min-Max 归一化后各特征贡献均衡，是基于距离聚类算法不可或缺的预处理步骤。')

P('3. DBSCAN 在高维数据上的局限性：Wine 有 13 个特征，高维空间中"密度"概念变得模糊（维度灾难），DBSCAN 难以找到合适的 eps 参数来区分所有类别，最终只发现 2 个簇并产生大量噪声点。')

P('4. 内部指标 vs 外部指标的矛盾：DBSCAN 的 DBI (1.56) 优于 K-Means (1.88)，但外部指标却远不如 K-Means。这再次说明 DBI 等内部指标不依赖真实标签，可能在某些情况下产生误导，需结合外部指标综合评估。')

P('5. 算法选择建议：对于类似 Wine 这样特征维度适中、类别呈凸形分布且类别数已知的数据集，K-Means 是首选；DBSCAN 更适合发现任意形状簇和处理包含噪声的低维数据。')

P('6. 距离度量的多样性：本实验系统实现了闵可夫斯基距离（p=1, 2）、VDM（无序属性）和 MinkovDMp（混合属性），为不同数据类型提供了完整的距离度量工具箱。')


# ====================== 五、附录 ======================
H('五、附录：核心代码', 1)

P('1. 计算 a, b, c, d：')
Code("""def compute_abcd(labels_pred, labels_true):
    n = len(labels_pred)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = (labels_pred[i] == labels_pred[j])
            same_true = (labels_true[i] == labels_true[j])
            if same_pred and same_true:     a += 1
            elif same_pred and not same_true: b += 1
            elif not same_pred and same_true: c += 1
            else:                            d += 1
    return a, b, c, d""")

P('2. Jaccard 系数：')
Code("def jaccard_coefficient(a, b, c): return a / (a + b + c)")

P('3. 闵可夫斯基距离：')
Code("""def minkowski_distance(x, y, p=2):
    x, y = np.asarray(x), np.asarray(y)
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)""")

P('4. Davies-Bouldin 指数：')
Code("""def davies_bouldin_index(X, labels):
    unique_labels = np.unique(labels[labels >= 0])
    k = len(unique_labels)
    clusters = {l: X[labels == l] for l in unique_labels}
    avg_dists = {l: cluster_avg_dist(clusters[l]) for l in unique_labels}
    dbi = 0.0
    for i in unique_labels:
        max_ratio = max(
            (avg_dists[i] + avg_dists[j]) / cluster_d_cen(clusters[i], clusters[j])
            for j in unique_labels if j != i)
        dbi += max_ratio
    return dbi / k""")

P('5. VDM 与 MinkovDMp：')
Code("""def vdm_distance(x_val, y_val, attr_idx, X, labels, p=2):
    col = X[:, attr_idx]
    m_a, m_b = np.sum(col == x_val), np.sum(col == y_val)
    if m_a == 0 or m_b == 0: return 0.0
    return sum(abs(np.sum(col[labels==c]==x_val)/m_a
                 - np.sum(col[labels==c]==y_val)/m_b)**p
               for c in np.unique(labels))

def minkov_dm_p(xi, xj, n_c, X, labels, p=2):
    cont = np.sum(np.abs(xi[:n_c] - xj[:n_c]) ** p)
    vdm = sum(vdm_distance(xi[u], xj[u], u, X, labels, p)
              for u in range(n_c, len(xi)))
    return (cont + vdm) ** (1.0 / p)""")

P('6. Min-Max 归一化：')
Code("""def min_max_normalize(X):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    denom = X_max - X_min; denom[denom == 0] = 1.0
    return (X - X_min) / denom""")


doc.save(OUT)
print(f'实验报告已保存: {OUT}')
