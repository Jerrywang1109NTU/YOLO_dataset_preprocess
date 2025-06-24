import os
import shutil

# 合并标签（加后缀）
def merge_labels_with_suffix(gray_folder, bright_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(gray_folder):
        if fname == "classes.txt":
            continue
        if fname.endswith(".txt"):
            base, ext = os.path.splitext(fname)
            dst_name = base + "_g" + ext
            src = os.path.join(gray_folder, fname)
            dst = os.path.join(output_folder, dst_name)
            shutil.copyfile(src, dst)
            print(f"[✓] Gray: {src} → {dst}")

    for fname in os.listdir(bright_folder):
        if fname == "classes.txt":
            continue
        if fname.endswith(".txt"):
            base, ext = os.path.splitext(fname)
            dst_name = base + "_b" + ext
            src = os.path.join(bright_folder, fname)
            dst = os.path.join(output_folder, dst_name)
            shutil.copyfile(src, dst)
            print(f"[✓] Bright: {src} → {dst}")

    print("✅ 合并完成！")

# 判断方向
def judge_direction(x, y):
    if (y > x and y > -x + 1):
        return 0  # 上
    elif (y < x and y > -x + 1):
        return 1  # 右
    elif (y < x and y < -x + 1):
        return 0  # 下
    else:
        return 1  # 左

# 按方向决定 bright 的 w,h
def classify_point(x, y):
    if judge_direction(x, y) == 0:  # 上
        return w_l, h_l
    else:  # 左/右
        return h_l, w_l

# 修改并拆分 label
def process_and_split_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt") or fname == "classes.txt":
            continue

        input_path = os.path.join(input_dir, fname)
        with open(input_path, 'r') as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            cls, x, y = int(parts[0]), float(parts[1]), float(parts[2])

            if fname.endswith("_b.txt"):
                w, h = classify_point(x, y)
                new_line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

            elif fname.endswith("_g.txt"):
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                else:
                    w, h = w_l, w_h  # 默认尺寸

                new_line = f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

            else:
                new_line = line.strip()

            new_lines.append(new_line)

        # 拆分成奇偶行
        base_name = os.path.splitext(fname)[0]
        even_lines = [l for i, l in enumerate(new_lines) if i % 2 == 0]
        odd_lines  = [l for i, l in enumerate(new_lines) if i % 2 == 1]

        out_even = os.path.join(output_dir, base_name + "_0.txt")
        out_odd  = os.path.join(output_dir, base_name + "_1.txt")

        with open(out_even, 'w') as f:
            f.write("\n".join(even_lines) + "\n")
        with open(out_odd, 'w') as f:
            f.write("\n".join(odd_lines) + "\n")

        print(f"[✓] 已处理并拆分：{fname}")

# === 用法 ===
gray_label_dir   = "dataset_labels/labels_gray_layer"
bright_label_dir = "dataset_labels/labels_bright_layer"
mix_label_dir    = "dataset_parity/labels_mix_all"
output_label_dir = "dataset_parity/labels_mix_parity"

merge_labels_with_suffix(gray_label_dir, bright_label_dir, mix_label_dir)
process_and_split_labels(mix_label_dir, output_label_dir)

from tools.modify_w_h import modify_wh_by_direction
import tools.parameters as pr
w_l = pr.w_l
h_l = pr.h_l
modify_wh_by_direction(output_label_dir, w_l, h_l)