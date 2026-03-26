import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import numpy as np
from sklearn.cluster import KMeans

# ---------------------- 辅助函数 (保持你原有的统计函数) ----------------------
def log_stats(title, log_obj):
    print(f"--- {title} ---")
    print(f"轨迹数量: {len(log_obj)}")
    # 注意: pm4py版本不同 get_variants 行为可能不同，这里假设兼容性
    try:
        vars = pm4py.get_variants(log_obj)
    except:
        vars = pm4py.stats.get_variants(log_obj)
    print(f"变体数量: {len(vars)}")
    print("----------------")

def remove_nan_trace_attributes(log):
    # 这里保留你的占位逻辑，实际需根据需求实现
    return log

# ---------------------- 主逻辑 ----------------------

# 1. 读取日志
file_path = r"D:\pythonProject\dataset\dataset_all\BPI Challenge 2020_ International Declarations_1_all\InternationalDeclarations.xes"
print("正在读取日志...")
log = pm4py.read_xes(file_path)

# 2. 获取变体及其频次
# variants 是一个字典: { (event1, event2...): count, ... }
variants = pm4py.get_variants(log)

# ---------------------- 核心修改：使用 K-Means 自动划分 ----------------------

print("正在执行 K-Means 聚类划分高低频...")

# 2.1 准备数据：提取所有变体的“出现次数”
# 我们需要保持 list 的顺序，以便后续能通过下标找回对应的变体
variant_items = list(variants.items())  # 格式: [ (variant_tuple, count), ... ]
counts = np.array([count for _, count in variant_items])

# 2.2 数据变形：sklearn 要求输入必须是二维数组 [[cnt], [cnt], ...]
X = counts.reshape(-1, 1)

# 2.3 执行 K-Means (k=2)
# random_state固定为42是为了复现性，避免每次跑结果微小差异
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# 2.4 确定哪个簇是“高频”，哪个是“低频”
# cluster_centers_ 包含两个中心点，数值大的那个中心点对应的簇就是“高频簇”
centers = kmeans.cluster_centers_.flatten()
high_freq_cluster_index = np.argmax(centers) # 获取中心点数值最大的那个簇的索引(0或1)

# 2.5 获取动态计算出的“截断阈值” (仅用于展示，逻辑上我们直接用聚类标签)
# 属于高频簇的所有点中，最小的那个数值，就是事实上的阈值
high_freq_counts = X[kmeans.labels_ == high_freq_cluster_index]
dynamic_threshold = np.min(high_freq_counts)

print(f"K-Means 聚类中心: {centers}")
print(f"自动计算的划分阈值 (High/Low Cutoff): >= {dynamic_threshold}")

# 2.6 根据聚类标签划分变体列表
high_freq_variants = []
low_freq_variants = []

# 遍历每个变体，看它的 count 被预测为哪一类
predicted_labels = kmeans.labels_

for i, (variant_tuple, count) in enumerate(variant_items):
    if predicted_labels[i] == high_freq_cluster_index:
        high_freq_variants.append(variant_tuple)
    else:
        low_freq_variants.append(variant_tuple)

print(f"高频变体数量: {len(high_freq_variants)}")
print(f"低频变体数量: {len(low_freq_variants)}")

# ---------------------- 后续处理 (保持原有逻辑) ----------------------

# 3. 过滤日志
# 注意：variants_filter.apply 需要传入变体的 list
print("正在过滤日志...")
high_log = variants_filter.apply(log, high_freq_variants)
# 对于低频日志，使用 positive=False (即排除掉高频的，剩下的就是低频)
low_log = variants_filter.apply(log, high_freq_variants, parameters={"positive": False})

# 4. 清理日志中的 NaN 属性 (调用你的自定义函数)
high_log = remove_nan_trace_attributes(high_log)
low_log = remove_nan_trace_attributes(low_log)

# 5. 导出清理后的日志
print("正在导出日志...")
xes_exporter.apply(high_log, "high_log.xes")
xes_exporter.apply(low_log, "low_log.xes")

# 6. 统计打印
log_stats("原始日志", log)
log_stats("高频日志 (K-Means High)", high_log)
log_stats("低频日志 (K-Means Low)", low_log)