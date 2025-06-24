# %%
import os
import cv2
import numpy as np
import tools.parameters as pr

# ⬆️➡️⬇️⬅️ 方向编号函数
def judge_direction(x, y):
    dx = x - 0.5
    dy = y - 0.5
    if abs(dy) > abs(dx):
        return 0 if y < 0.5 else 2  # 上 or 下
    else:
        return 3 if x < 0.5 else 1  # 左 or 右

# 目录设置
image_dir = "./dataset_unmod/data_unenh_pruned"
label_dir = "dataset_parity/labels_mix_parity"


out_dir = "dataset_parity/images_mix_parity_enh"
tmpdir = "dataset_parity/tmp"

in_dir = image_dir
output_dir = out_dir
os.makedirs(output_dir, exist_ok=True)

# patch size setting
w_b = pr.w_b
h_b = pr.h_b
w_g = pr.w_g
h_g = pr.h_g

# 加载所有方向的 patch（bright & gray）从新目录
patch_map = {}
patch_root = "patches_texture"
for light in ['g']:  # bright, gray
    for dir_id in range(4):  # 0:上, 1:右, 2:下, 3:左
        patch_path = os.path.join(patch_root, f"patch_texture_{light}_{dir_id}.png")
        patch = cv2.imread(patch_path)
        if patch is None:
            raise FileNotFoundError(f"找不到 patch 文件: {patch_path}")
        if dir_id == 0 or dir_id == 2:  # 上下方向
            patch = cv2.resize(patch, (w_g, h_g))
        else:  # 左右方向
            patch = cv2.resize(patch, (h_g, w_g))
        patch_map[(light, dir_id)] = patch

replacement_right = pr.replacement_right
replacement_left = pr.replacement_left

# 扩展为 BGR 三通道（将灰度图转换为 BGR）
replacement_right = np.stack([replacement_right]*3, axis=-1)
replacement_left = np.stack([replacement_left]*3, axis=-1)

for dir_id in range(2):  # 0:上, 1:右, 2:下, 3:左
    light = 'b'
    if dir_id == 0:  # 上下方向
        patch = cv2.resize(replacement_left, (w_b, h_b))
    else:  # 左右方向
        patch = cv2.resize(replacement_right, (h_b, w_b))
    patch_map[(light, dir_id)] = patch


# 支持图像扩展名
image_exts = (".png", ".jpg", ".jpeg")

# 处理标签
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

cnt = 0

for label_file in label_files:
    img_id = label_file.split("_")[0]  # 例如 9_14 → 9
    base = os.path.splitext(label_file)[0]  # 9_14
    image_path = None

    for ext in image_exts:
        candidate = os.path.join(image_dir, img_id + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"[Skip] 未找到图像: {img_id}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 图像读取失败: {image_path}")
        continue

    h, w = img.shape[:2]
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls = int(parts[0])  # 0=bright, 1=gray
        if cls == 0:
            continue  # 只处理 gray 标签
        x, y = float(parts[1]), float(parts[2])
        cx = int(x * w)
        cy = int(y * h)

        direction = judge_direction(x, y)
        light = 'b' if cls == 0 else 'g'
        patch = patch_map[(light, direction)]

        ph, pw = patch.shape[:2]
        top = max(0, cy - ph // 2)
        left = max(0, cx - pw // 2)
        bottom = min(h, top + ph)
        right = min(w, left + pw)
        patch_cropped = patch[:bottom-top, :right-left]

        if (bottom - top) <= 0 or (right - left) <= 0:
            print(f"[Skip Patch] patch 超出边界: {base}, cx={cx}, cy={cy}, w={w}, h={h}")
            continue

        img[top:bottom, left:right] = patch_cropped

    # 保存图像
    save_name = base + ".png"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, img)
    print(f"[OK] 已保存: {save_name}")
    # if cnt == 0:
        # break

from tools.Img_Enhance import Enhance_all

in_dir = out_dir
output_dir = tmpdir

Enhance_all(in_dir, output_dir)
image_dir = tmpdir
output_dir = out_dir

for label_file in label_files:
    img_id = label_file.split("_")[0]  # 例如 9_14 → 9
    base = os.path.splitext(label_file)[0]  # 9_14
    image_path = None

    for ext in image_exts:
        candidate = os.path.join(image_dir, base + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"[Skip] 未找到图像: {img_id}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 图像读取失败: {image_path}")
        continue

    h, w = img.shape[:2]
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls = int(parts[0])  # 0=bright, 1=gray
        if cls == 1:
            continue  # 只处理 bright 标签
        x, y = float(parts[1]), float(parts[2])
        cx = int(x * w)
        cy = int(y * h)

        direction = 0 if x < 0.5 else 1
        light = 'b' if cls == 0 else 'g'
        patch = patch_map[(light, direction)]

        ph, pw = patch.shape[:2]
        top = max(0, cy - ph // 2)
        left = max(0, cx - pw // 2)
        bottom = min(h, top + ph)
        right = min(w, left + pw)
        patch_cropped = patch[:bottom-top, :right-left]

        if img.ndim == 2:
            img[top:bottom, left:right] = patch_cropped
        else:
            if patch_cropped.ndim == 2:
                patch_cropped = cv2.cvtColor(patch_cropped, cv2.COLOR_GRAY2BGR)
            img[top:bottom, left:right] = patch_cropped

    # 保存图像
    save_name = base + ".png"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, img)
    print(f"[OK] 已保存: {save_name}")
    # if cnt == 0:
        # break