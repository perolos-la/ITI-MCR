import tkinter as tk
from tkinter import filedialog
import os
import random
from copy import deepcopy
from datetime import timedelta
import pm4py
from pm4py import read_xes, write_xes
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog, Trace, Event
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Default file paths
HIGH_LOG_PATH = "D:\\pythonProject\\code\\high_log.xes"
LOW_LOG_PATH = "D:\\pythonProject\\code\\low_log.xes"
OUTPUT_DIR = "D:\\pythonProject\\code"

def compute_dependency_matrix(high_log_path, output_dir):
    log = xes_importer.apply(high_log_path)
    traces = [ [event["concept:name"] for event in trace] for trace in log ]
    activities = sorted({act for trace in traces for act in trace})
    dependency_counts = defaultdict(lambda: defaultdict(int))
    for trace in traces:
        for i in range(len(trace) - 1):
            dependency_counts[trace[i]][trace[i + 1]] += 1
    df = pd.DataFrame.from_dict({prev: dict(next_acts) for prev, next_acts in dependency_counts.items()}, orient="index").fillna(0).astype(int)
    df = df.reindex(index=activities, columns=activities).fillna(0).astype(int)
    df.to_excel(os.path.join(output_dir, "直接跟随关系矩阵.xlsx"))

def compute_causal_probability_matrix(output_dir):
    df = pd.read_excel(os.path.join(output_dir, "直接跟随关系矩阵.xlsx"), index_col=0)
    causal_matrix = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for row_act in df.index:
        for col_act in df.columns:
            if row_act != col_act:
                ab = df.loc[row_act, col_act]
                ba = df.loc[col_act, row_act]
                causal_matrix.loc[row_act, col_act] = (ab - ba) / (ab + ba + 1)
            else:
                aa = df.loc[row_act, col_act]
                causal_matrix.loc[row_act, col_act] = aa / (aa + 1)
    causal_matrix.to_excel(os.path.join(output_dir, "因果依赖概率矩阵.xlsx"))

def find_strong_activity_pairs(dep_threshold, causal_threshold, output_dir):
    dep_matrix = pd.read_excel(os.path.join(output_dir, "直接跟随关系矩阵.xlsx"), index_col=0)
    causal_matrix = pd.read_excel(os.path.join(output_dir, "因果依赖概率矩阵.xlsx"), index_col=0)
    strong_pairs = []
    for a in dep_matrix.index:
        for b in dep_matrix.columns:
            dep_count = dep_matrix.loc[a, b]
            causal_prob = causal_matrix.loc[a, b]
            if dep_count > dep_threshold and causal_prob > causal_threshold:
                strong_pairs.append((a, b, dep_count, causal_prob))
    df_strong = pd.DataFrame(strong_pairs, columns=["Source Activity", "Target Activity", "Dependency Count", "Causal Probability"])
    df_strong = df_strong.sort_values(by=["Dependency Count", "Causal Probability"], ascending=[False, False])
    df_strong.to_excel(os.path.join(output_dir, "强关联活动对.xlsx"), index=False)
    return strong_pairs

def compute_conformance_ratio(valid_pairs, low_log_path, output_dir):
    valid_pairs_set = {(pair[0], pair[1]) for pair in valid_pairs}
    log = xes_importer.apply(low_log_path)
    records = []
    for trace in log:
        trace_id = trace.attributes.get("concept:name", str(id(trace)))
        activities = [event["concept:name"] for event in trace if "concept:name" in event]
        total_pairs = len(activities) - 1
        matching_pairs = sum(1 for i in range(total_pairs) if (activities[i], activities[i + 1]) in valid_pairs_set)
        conformance_ratio = matching_pairs / total_pairs if total_pairs > 0 else None
        records.append({"trace_id": trace_id, "total_pairs": total_pairs, "matching_pairs": matching_pairs, "conformance_ratio": conformance_ratio})
    pd.DataFrame(records).to_excel(os.path.join(output_dir, "控制维度分数.xlsx"), index=False)
    return pd.DataFrame(records)

def get_resource_activity_matrix(high_log_path, output_dir):
    high_log = xes_importer.apply(high_log_path)
    activity_resource_counts = defaultdict(lambda: defaultdict(int))
    for trace in high_log:
        for event in trace:
            activity = event["concept:name"]
            resource = event.get("org:resource", "UNKNOWN")
            activity_resource_counts[activity][resource] += 1
    df = pd.DataFrame.from_dict(activity_resource_counts, orient="index").fillna(0)
    df.to_excel(os.path.join(output_dir, "资源-活动矩阵.xlsx"), index=True, header=True)

def get_resource_activity_pairs(output_dir, min_share):
    df_matrix = pd.read_excel(os.path.join(output_dir, "资源-活动矩阵.xlsx"), index_col=0)
    df_long = df_matrix.reset_index().melt(id_vars="index", var_name="resource", value_name="count")
    df_long["count"] = pd.to_numeric(df_long["count"], errors="coerce").fillna(0)
    total = df_long.groupby("index")["count"].sum().rename("total_count").reset_index()
    df2 = df_long.merge(total, on="index")
    df2["share"] = df2["count"] / df2["total_count"]
    valid_df = df2[df2["share"] >= min_share][["index", "resource"]]
    valid_pairs = set(zip(valid_df["index"], valid_df["resource"]))
    valid_df.to_excel(os.path.join(output_dir, "资源-活动对.xlsx"), index=False)
    return valid_pairs

def compute_deviation(valid_pairs, low_log_path, output_dir):
    log = xes_importer.apply(low_log_path)
    records = []
    for trace in log:
        trace_id = trace.attributes.get("concept:name", trace.attributes.get("id", ""))
        total_pairs = matching_pairs = 0
        for ev in trace:
            if "org:resource" in ev and "concept:name" in ev:
                total_pairs += 1
                if (ev["concept:name"], ev["org:resource"]) in valid_pairs:
                    matching_pairs += 1
        deviation = matching_pairs / total_pairs if total_pairs > 0 else None
        records.append({"trace_id": trace_id, "total_pairs": total_pairs, "matching_pairs": matching_pairs, "deviation": deviation})
    df_dev = pd.DataFrame(records)
    df_dev.to_excel(os.path.join(output_dir, "组织维度分数.xlsx"), index=False)
    return df_dev

def calculate_activity_pair_time_diff(high_log_path, output_dir):
    strong_pairs = pd.read_excel(os.path.join(output_dir, "强关联活动对.xlsx"))
    log = xes_importer.apply(high_log_path)
    valid_pairs = set(zip(strong_pairs['Source Activity'], strong_pairs['Target Activity']))
    time_diffs = defaultdict(list)
    for trace in log:
        events = sorted(trace, key=lambda e: e['time:timestamp'])
        for i in range(1, len(events)):
            pair = (events[i - 1]['concept:name'], events[i]['concept:name'])
            if pair in valid_pairs:
                time_diff = (events[i]['time:timestamp'] - events[i - 1]['time:timestamp']).total_seconds()
                time_diffs[pair].append(time_diff)
    results = []
    for pair, durations in time_diffs.items():
        if durations:
            mean_sec = np.mean(durations)
            std_sec = np.std(durations)
            results.append({
                'Source Activity': pair[0], 'Target Activity': pair[1],
                'mean_time_diff_sec': mean_sec, 'std_time_diff_sec': std_sec,
                'mean_time_diff_hour': mean_sec / 3600, 'std_time_diff_hour': std_sec / 3600,
                'lower_limit_sec': mean_sec - std_sec, 'upper_limit_sec': mean_sec + std_sec
            })
    pd.DataFrame(results).to_excel(os.path.join(output_dir, "活动对时间差分布表.xlsx"), index=False)

def compute_traces_time_dimension_score(output_dir, penalty_score, low_log_path):
    time_diff_df = pd.read_excel(os.path.join(output_dir, "活动对时间差分布表.xlsx"))
    time_stats = {(row["Source Activity"], row["Target Activity"]): {"mean": row["mean_time_diff_sec"], "std": row["std_time_diff_sec"]} for _, row in time_diff_df.iterrows()}
    log = xes_importer.apply(low_log_path)
    results = []
    for trace in log:
        trace_id = trace.attributes.get("concept:name", str(id(trace)))
        events = sorted(trace, key=lambda e: e["time:timestamp"])
        deviations = []
        for i in range(len(events) - 1):
            pair = (events[i]["concept:name"], events[i + 1]["concept:name"])
            delta = (events[i + 1]["time:timestamp"] - events[i]["time:timestamp"]).total_seconds()
            if pair in time_stats:
                mean = time_stats[pair]["mean"]
                std = time_stats[pair]["std"]
                deviation = abs((delta - mean) / std) if std > 0 else 0
            else:
                deviation = penalty_score
            deviations.append(deviation)
        score = np.mean(deviations) if deviations else 0.0
        results.append({"trace_id": trace_id, "time_dimension_score": score, "num_pairs": len(deviations)})
    pd.DataFrame(results).to_excel(os.path.join(output_dir, "时间维度分数.xlsx"), index=False)


# def combine_all_three_scores(w_control, w_org, w_time, output_dir, low_log_path):
#     df_control = pd.read_excel(os.path.join(output_dir, "控制维度分数.xlsx"))
#     df_time = pd.read_excel(os.path.join(output_dir, "时间维度分数.xlsx"))
#     df_org = pd.read_excel(os.path.join(output_dir, "组织维度分数.xlsx"))
#     df = df_control.merge(df_time, on="trace_id").merge(df_org, on="trace_id")
#     log = xes_importer.apply(low_log_path)
#     noise_traces = {str(trace.attributes.get('concept:name', f'Trace-{id(trace)}')) for trace in log if any('noise:added' in event and str(event['noise:added']).lower() == 'true' for event in trace)}
#     df.insert(df.columns.get_loc("trace_id") + 1, "包含噪声事件", df["trace_id"].astype(str).isin(noise_traces).map({True: '是', False: ''}))
#     scaler = MinMaxScaler()
#     df["time_score_norm"] = 1 - scaler.fit_transform(df[["time_dimension_score"]])
#     df["org_score_norm"] = df["deviation"]
#     df["control_score_norm"] = df["conformance_ratio"]
#     df["final_score"] = (w_control * df["control_score_norm"] + w_time * df["time_score_norm"] + w_org * df["org_score_norm"])
#     cols = ["trace_id", "包含噪声事件", "control_score_norm", "time_dimension_score", "time_score_norm", "deviation", "org_score_norm", "final_score"]
#     df[cols].to_excel(os.path.join(output_dir, "维度综合评分_三视角.xlsx"), index=False)

def combine_all_three_scores(w_control, w_org, w_time, output_dir, low_log_path):
    """
    合并三个维度的分数，并计算多种权重组合下的综合评分。

    参数:
    w_control (float): 控制流视角的自定义权重
    w_org (float): 组织视角的自定义权重
    w_time (float): 时间视角的自定义权重
    output_dir (str): 包含输入Excel文件并用于保存输出的目录
    low_log_path (str): 低频日志的.xes文件路径，用于识别噪声
    """

    print("开始合并分数...")
    # --- 1. 加载和合并数据 ---
    try:
        df_control = pd.read_excel(os.path.join(output_dir, "控制维度分数.xlsx"))
        df_time = pd.read_excel(os.path.join(output_dir, "时间维度分数.xlsx"))
        df_org = pd.read_excel(os.path.join(output_dir, "组织维度分数.xlsx"))
    except FileNotFoundError as e:
        print(f"错误：无法找到输入文件。请确保 '控制维度分数.xlsx', '时间维度分数.xlsx', '组织维度分数.xlsx' 都在 '{output_dir}' 目录中。")
        print(e)
        return

    df = df_control.merge(df_time, on="trace_id").merge(df_org, on="trace_id")

    # --- 2. 加载日志并识别噪声轨迹 ---
    print(f"正在加载日志以识别噪声: {low_log_path}")
    try:
        log = xes_importer.apply(low_log_path)
        noise_traces = {str(trace.attributes.get('concept:name', f'Trace-{id(trace)}'))
                        for trace in log
                        if
                        any('noise:added' in event and str(event['noise:added']).lower() == 'true' for event in trace)}
        print(f"识别到 {len(noise_traces)} 条噪声轨迹。")
    except Exception as e:
        print(f"警告：加载XES日志失败。'包含噪声事件' 列将为空。错误: {e}")
        noise_traces = set()

    df.insert(df.columns.get_loc("trace_id") + 1, "包含噪声事件",
              df["trace_id"].astype(str).isin(noise_traces).map({True: '是', False: ''}))

    # --- 3. 规范化各维度分数 ---
    # 确保 'conformance_ratio' 和 'deviation' 存在
    if "conformance_ratio" not in df.columns or "deviation" not in df.columns:
        print("错误：输入的Excel文件中缺少 'conformance_ratio' 或 'deviation' 列。")
        return

    scaler = MinMaxScaler()
    # 时间维度：分数越低越好，因此用 1- 规范化
    df["time_score_norm"] = 1 - scaler.fit_transform(df[["time_dimension_score"]])
    # 组织维度：直接使用 'deviation' 作为分数
    df["org_score_norm"] = df["deviation"]

    # 控制流维度：直接使用 'conformance_ratio' 作为分数
    df["control_score_norm"] = df["conformance_ratio"]

    # --- 4. [核心修改] 计算所有评分组合 ---
    print("正在计算所有评分组合...")
    c_score = df["control_score_norm"]
    o_score = df["org_score_norm"]
    t_score = df["time_score_norm"]

    # 单视角
    df["score_control_only"] = c_score
    df["score_org_only"] = o_score
    df["score_time_only"] = t_score

    # 双视角 (w=0.5)
    df["score_control_org"] = (0.5 * c_score) + (0.5 * o_score)
    df["score_control_time"] = (0.5 * c_score) + (0.5 * t_score)
    df["score_org_time"] = (0.5 * o_score) + (0.5 * t_score)

    # 三视角 (使用传入的自定义权重)
    df["final_score"] = (w_control * c_score) + (w_org * o_score) + (w_time * t_score)

    # --- 5. [核心修改] 定义最终输出列 ---
    cols = [
        # 基础信息
        "trace_id",
        "包含噪声事件",

        # 规范化后的基础分数 (用于分析)
        "control_score_norm",
        "org_score_norm",
        "time_score_norm",

        # 原始分数 (用于参考)
        "conformance_ratio",
        "deviation",
        "time_dimension_score",

        # 所有计算出的综合评分
        "score_control_only",
        "score_org_only",
        "score_time_only",
        "score_control_org",
        "score_control_time",
        "score_org_time",
        "final_score"
    ]

    # 确保所有列都存在，以防万一
    final_cols = [col for col in cols if col in df.columns]

    # --- 6. [核心修改] 保存到新的Excel文件 ---
    output_filename = os.path.join(output_dir, "维度综合评分_三视角.xlsx")

    try:
        df[final_cols].to_excel(output_filename, index=False)
        print(f"成功！多视角对比评分已保存到: {output_filename}")
    except Exception as e:
        print(f"错误：保存Excel文件失败。{e}")


def filter_traces_by_score(threshold, low_log_path, output_dir):
    score_file = os.path.join(output_dir, "维度综合评分_三视角.xlsx")
    output_xes = os.path.join(output_dir, "low_log_filtered.xes")
    df = pd.read_excel(score_file)
    valid_traces = set(map(str, df[df['final_score'] >= threshold]['trace_id'].tolist()))
    log = xes_importer.apply(low_log_path)
    filtered_log = EventLog(attributes=log.attributes, extensions=log.extensions, omni_present=log.omni_present, classifiers=log.classifiers)
    for trace in log:
        trace_id = str(trace.attributes.get("concept:name", "")).strip()
        if trace_id and trace_id in valid_traces:
            filtered_log.append(trace)
    if len(filtered_log) > 0:
        xes_exporter.apply(filtered_log, output_xes)

def merge_and_export_logs(high_log_path, output_dir):
    high_log = xes_importer.apply(high_log_path)
    low_log = xes_importer.apply(os.path.join(output_dir, "low_log_filtered.xes"))
    merged_log = EventLog()
    for trace in high_log:
        merged_log.append(trace)
    for trace in low_log:
        merged_log.append(trace)
    pm4py.write_xes(merged_log, os.path.join(output_dir, "merged_log_filtered.xes"))

    # high_log = xes_importer.apply(high_log_path)
    # low_log = xes_importer.apply(os.path.join(output_dir, "low_log_noisy.xes"))
    # merged_log = EventLog()
    # for trace in high_log:
    #     merged_log.append(trace)
    # for trace in low_log:
    #     merged_log.append(trace)
    # pm4py.write_xes(merged_log, os.path.join(output_dir, "merged_log_noisy.xes"))

def main():
    root = tk.Tk()
    root.title("多维度？!")

    # Parameter Inputs
    params = [
        ("dep_threshold:", "50", "Direct Follow Threshold"),
        ("causal_threshold:", "0.9", "Causal Probability Threshold"),
        ("MIN_SHARE:", "0.03", "Min Resource-Activity Share"),
        ("PENALTY_SCORE:", "5.0", "Time Penalty Score"),
        ("w_control:", "0.45", "Control Flow Weight"),
        ("w_org:", "0.45", "Organization Weight"),
        ("w_time:", "0.1", "Time Weight"),
        ("threshold:", "0.847", "Trace Filter Threshold"),
    ]
    entries = {}
    for i, (label, default, _) in enumerate(params):
        tk.Label(root, text=label).grid(row=i, column=0, padx=5, pady=5)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.insert(0, default)
        entries[label.strip(":")] = entry

    # File Selection
    tk.Label(root, text="高频日志路径:").grid(row=8, column=0, padx=5, pady=5)
    high_log_entry = tk.Entry(root, width=50)
    high_log_entry.grid(row=8, column=1, padx=5, pady=5)
    high_log_entry.insert(0, HIGH_LOG_PATH)
    tk.Button(root, text="Browse", command=lambda: select_file(high_log_entry)).grid(row=8, column=2)

    tk.Label(root, text="低频日志路径:").grid(row=9, column=0, padx=5, pady=5)
    low_log_entry = tk.Entry(root, width=50)
    low_log_entry.grid(row=9, column=1, padx=5, pady=5)
    low_log_entry.insert(0, LOW_LOG_PATH)
    tk.Button(root, text="Browse", command=lambda: select_file(low_log_entry)).grid(row=9, column=2)

    tk.Label(root, text="输出路径:").grid(row=10, column=0, padx=5, pady=5)
    output_dir_entry = tk.Entry(root, width=50)
    output_dir_entry.grid(row=10, column=1, padx=5, pady=5)
    output_dir_entry.insert(0, OUTPUT_DIR)
    tk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry)).grid(row=10, column=2)

    # Buttons
    tk.Button(root, text="Run！！！", command=lambda: run_analysis(
        int(entries["dep_threshold"].get()), float(entries["causal_threshold"].get()),
        float(entries["MIN_SHARE"].get()), float(entries["PENALTY_SCORE"].get()),
        float(entries["w_control"].get()), float(entries["w_org"].get()), float(entries["w_time"].get()),
        float(entries["threshold"].get()), high_log_entry.get(), low_log_entry.get(), output_dir_entry.get()
    )).grid(row=11, column=0, columnspan=3, pady=10)
    tk.Button(root, text="退出", command=root.quit).grid(row=12, column=0, columnspan=3, pady=10)

    root.mainloop()

def select_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("XES files", "*.xes")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def select_directory(entry):
    dir_path = filedialog.askdirectory()
    if dir_path:
        entry.delete(0, tk.END)
        entry.insert(0, dir_path)

def run_analysis(dep_threshold, causal_threshold, min_share, penalty_score, w_control, w_org, w_time, threshold, high_log_path, low_log_path, output_dir):
    compute_dependency_matrix(high_log_path, output_dir)
    compute_causal_probability_matrix(output_dir)
    strong_activity_pairs = find_strong_activity_pairs(dep_threshold, causal_threshold, output_dir)
    compute_conformance_ratio(strong_activity_pairs, low_log_path, output_dir)
    get_resource_activity_matrix(high_log_path, output_dir)
    valid_pairs = get_resource_activity_pairs(output_dir, min_share)
    compute_deviation(valid_pairs, low_log_path, output_dir)
    calculate_activity_pair_time_diff(high_log_path, output_dir)
    compute_traces_time_dimension_score(output_dir, penalty_score, low_log_path)
    combine_all_three_scores(w_control, w_org, w_time, output_dir, low_log_path)
    filter_traces_by_score(threshold, low_log_path, output_dir)
    merge_and_export_logs(high_log_path, output_dir)
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()