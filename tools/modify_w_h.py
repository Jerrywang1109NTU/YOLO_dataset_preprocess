import os
import argparse

def judge_direction(x, y):
    dx = x - 0.5
    dy = y - 0.5
    if abs(dy) > abs(dx):
        return 0 if y < 0.5 else 2  # up or down
    else:
        return 3 if x < 0.5 else 1  # left or right

def classify_point(x, y, target_w, target_h):
    direction = judge_direction(x, y)
    if direction == 0 or direction == 2:  # up or down
        return target_w, target_h  # 竖 patch
    else:
        return target_h, target_w  # 横 patch

def modify_wh_by_direction(label_dir, w_l, h_l):
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt") or fname == "classes.txt":
            continue
        file_path = os.path.join(label_dir, fname)
        new_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls, x_str, y_str = parts[0], parts[1], parts[2]
            try:
                x, y = float(x_str), float(y_str)
            except ValueError:
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1):
                print(f"[⚠] 跳过越界行: {fname} -> {line.strip()}")
                continue
            w, h = classify_point(x, y, target_w, target_h)
            new_line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            new_lines.append(new_line)
        with open(file_path, 'w') as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("")
        print(f"[✓] 已处理: {fname}（保留 {len(new_lines)} 行）")