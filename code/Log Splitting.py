import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import numpy as np
from sklearn.cluster import KMeans

def log_stats(title, log_obj):
    print(f"--- {title} ---")
    print(f"轨迹数量: {len(log_obj)}")
    try:
        vars = pm4py.get_variants(log_obj)
    except:
        vars = pm4py.stats.get_variants(log_obj)
    print(f"变体数量: {len(vars)}")
    print("----------------")

file_path = r"D:\pythonProject\dataset\dataset_all\BPI Challenge 2020_ International Declarations_1_all\InternationalDeclarations.xes"
print("正在读取日志...")
log = pm4py.read_xes(file_path)
variants = pm4py.get_variants(log)
print("正在执行 K-Means 聚类划分高低频...")
variant_items = list(variants.items())  # 格式: [ (variant_tuple, count), ... ]
counts = np.array([count for _, count in variant_items])
X = counts.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
centers = kmeans.cluster_centers_.flatten()
high_freq_cluster_index = np.argmax(centers)
high_freq_counts = X[kmeans.labels_ == high_freq_cluster_index]
dynamic_threshold = np.min(high_freq_counts)
print(f"K-Means 聚类中心: {centers}")
print(f"自动计算的划分阈值 (High/Low Cutoff): >= {dynamic_threshold}")
high_freq_variants = []
low_freq_variants = []
predicted_labels = kmeans.labels_

for i, (variant_tuple, count) in enumerate(variant_items):
    if predicted_labels[i] == high_freq_cluster_index:
        high_freq_variants.append(variant_tuple)
    else:
        low_freq_variants.append(variant_tuple)

print(f"高频变体数量: {len(high_freq_variants)}")
print(f"低频变体数量: {len(low_freq_variants)}")

print("正在过滤日志...")
high_log = variants_filter.apply(log, high_freq_variants)
low_log = variants_filter.apply(log, high_freq_variants, parameters={"positive": False})

high_log = remove_nan_trace_attributes(high_log)
low_log = remove_nan_trace_attributes(low_log)

print("正在导出日志...")
xes_exporter.apply(high_log, "high_log.xes")
xes_exporter.apply(low_log, "low_log.xes")

log_stats("原始日志", log)
log_stats("高频日志 (K-Means High)", high_log)
log_stats("低频日志 (K-Means Low)", low_log)
