import os
import random
import tools.parameters as pr


random.seed(42)
# 输入文件夹路径
gray_label_dir = "dataset_labels/labels_gray_all"
bright_label_dir = "dataset_labels/labels_bright_all"

# 输出文件夹路径
output_label_dir = "dataset_random/labels_mix_random"
os.makedirs(output_label_dir, exist_ok=True)

# 抽样参数
num_combinations = pr.num_combinations
gray_count = pr.gray_count
bright_count = pr.bright_count
min_manhattan_distance = pr.min_manhattan_distance

# ----------- 工具函数 -----------

# 替换 gray 的 class 为 1（防止混合后无法区分来源）
def read_gray_lines(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    for line in lines:
        line[0] = '1'
    return lines

# 读取 bright 标签，保持原样（class 0）
def read_bright_lines(filepath):
    with open(filepath, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

# 判断组合中任意两个框之间是否都满足最小曼哈顿距离
def check_min_manhattan(labels, min_dist):
    for i in range(len(labels)):
        xi, yi = float(labels[i][1]), float(labels[i][2])
        for j in range(i + 1, len(labels)):
            xj, yj = float(labels[j][1]), float(labels[j][2])
            dist = abs(xi - xj) + abs(yi - yj)
            if dist < min_dist:
                return False
    return True

# 从 gray 和 bright 中抽样组合，直到满足最小距离要求
def sample_valid_combination(gray_labels, bright_labels, gray_k, bright_k, min_dist, max_attempts=50000):
    for _ in range(max_attempts):
        gray_sample = random.sample(gray_labels, gray_k)
        bright_sample = random.sample(bright_labels, bright_k)
        merged = gray_sample + bright_sample
        if check_min_manhattan(merged, min_dist):
            return merged
    return None

# ----------- 主执行流程 -----------

# 获取共有的标签文件（按名称交集）
gray_label_files = set(f for f in os.listdir(gray_label_dir) if f.endswith(".txt"))
bright_label_files = set(f for f in os.listdir(bright_label_dir) if f.endswith(".txt"))
common_files = sorted(list(gray_label_files & bright_label_files))

# 主循环处理每个图像对应的标签文件
for file in common_files:
    gray_path = os.path.join(gray_label_dir, file)
    bright_path = os.path.join(bright_label_dir, file)

    gray_lines = read_gray_lines(gray_path)
    bright_lines = read_bright_lines(bright_path)

    if len(gray_lines) < gray_count or len(bright_lines) < bright_count:
        print(f"[Skip] 标签不足: {file}")
        continue

    for i in range(num_combinations):
        sample = sample_valid_combination(
            gray_lines, bright_lines, gray_count, bright_count, min_manhattan_distance
        )
        if sample is None:
            print(f"[Warn] 找不到合法组合: {file}, 第 {i+1} 组")
            continue

        out_file = os.path.join(output_label_dir, f"{os.path.splitext(file)[0]}_{i}.txt")
        with open(out_file, 'w') as f:
            for item in sample:
                f.write(" ".join(item) + "\n")
        print(f"[OK] 写入完成: {out_file}")

from tools.modify_w_h import modify_wh_by_direction
import tools.parameters as pr
w_l = pr.w_l
h_l = pr.h_l
modify_wh_by_direction(output_label_dir, w_l, h_l)