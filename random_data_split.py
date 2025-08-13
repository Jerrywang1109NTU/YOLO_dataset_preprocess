import os
import shutil
import argparse
import cv2
from collections import defaultdict
import tools.parameters as pr
import numpy as np

# 固定编号组
train_ids = {5, 16, 13, 32, 7, 23, 29, 18, 22, 40, 27, 14, 33, 20, 25, 39, 36, 34, 42, 1, 10, 37, 3, 6, 9, 28}
test_ids = {17, 8, 30, 43, 24, 4, 26, 38}
valid_ids = {2, 31, 12, 11, 15, 19, 21, 41, 35, 44}

# 数据集分组函数
def assign_group(base_name):
    num = int(base_name)
    if num in train_ids:
        return "train"
    elif num in test_ids:
        return "test"
    elif num in valid_ids:
        return "valid"
    else:
        return None

# 图像旋转函数
def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img

# 文件复制主逻辑
def copy_and_split(image_dir, label_dir, output_image_dir, output_label_dir, background_dir=None):
    groups = defaultdict(list)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in image_files:
        base_name = file.split('_')[0] if "_" in file else os.path.splitext(file)[0]
        groups[base_name].append(file)

    # 创建输出目录
    for subset in ["train", "test", "valid"]:
        os.makedirs(os.path.join(output_image_dir, subset), exist_ok=True)
        os.makedirs(os.path.join(output_label_dir, subset), exist_ok=True)

    # 复制图像和标签
    for base_name, file_list in groups.items():
        subset = assign_group(base_name)
        if subset is None:
            print(f"⚠️ 警告: 未知编号 {base_name}, 跳过。")
            continue

        for image_file in file_list:
            image_src_path = os.path.join(image_dir, image_file)
            image_dst_path = os.path.join(output_image_dir, subset, image_file)

            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_src_path = os.path.join(label_dir, label_file)
            label_dst_path = os.path.join(output_label_dir, subset, label_file)

            shutil.copy(image_src_path, image_dst_path)
            if os.path.exists(label_src_path):
                shutil.copy(label_src_path, label_dst_path)

    # 背景图处理
    if background_dir:
        crop_w, crop_h = 256, 256         # 裁剪尺寸
        crops_per_angle = 0               # 每个角度裁剪的数量

        bg_files = [f for f in os.listdir(background_dir) if f.endswith(".png")]
        for bg_file in bg_files:
            base_id = os.path.splitext(bg_file)[0]
            subset = assign_group(base_id)
            if subset is None:
                print(f"⚠️ 背景图未知编号: {base_id}, 跳过。")
                continue

            bg_path = os.path.join(background_dir, bg_file)
            img = cv2.imread(bg_path)
            if img is None:
                print(f"⚠️ 读取失败: {bg_path}")
                continue

            for angle in [0, 90, 180, 270]:
                rotated = rotate_image(img, angle)
                h, w = rotated.shape[:2]

                if w < crop_w or h < crop_h:
                    print(f"⚠️ 图像尺寸太小，无法裁剪: {bg_file} 旋转 {angle}°")
                    continue

                for idx in range(crops_per_angle):
                    x = np.random.randint(0, w - crop_w + 1)
                    y = np.random.randint(0, h - crop_h + 1)
                    cropped = rotated[y:y + crop_h, x:x + crop_w]

                    suffix = f"_r{angle}_c{idx}"
                    filename = f"{base_id}_bk{suffix}.png"
                    img_out_path = os.path.join(output_image_dir, subset, filename)
                    label_out_path = os.path.join(output_label_dir, subset, filename.replace(".png", ".txt"))

                    cv2.imwrite(img_out_path, cropped)
                    with open(label_out_path, 'w') as f:
                        pass  # 写空标签

            print(f"✅ 背景图 {bg_file} 完成旋转 + 裁剪增强写入")


    # 统计数量
    train_count = sum(len(groups[str(i)]) for i in train_ids if str(i) in groups)
    test_count = sum(len(groups[str(i)]) for i in test_ids if str(i) in groups)
    valid_count = sum(len(groups[str(i)]) for i in valid_ids if str(i) in groups)

    print("✅ 数据集划分完成！")
    print(f"Train: {len(train_ids)} groups, {train_count} images")
    print(f"Test: {len(test_ids)} groups, {test_count} images")
    print(f"Valid: {len(valid_ids)} groups, {valid_count} images")

# ========== 主程序入口 ==========

parser = argparse.ArgumentParser(description="按编号划分图像和标签为 train/test/valid，并添加背景图像")
parser.add_argument('--image_dir', type=str, default="dataset_random/images_mix_random_enh", help='图像源路径')
parser.add_argument('--label_dir', type=str, default="dataset_random/labels_mix_random", help='标签源路径')
parser.add_argument('--output_image_dir', type=str, default=pr.YOLO_save_dir_image, help='图像输出路径')
parser.add_argument('--output_label_dir', type=str, default=pr.YOLO_save_dir_label, help='标签输出路径')
parser.add_argument('--background_dir', type=str, default="dataset_mod", help='背景图路径')
args = parser.parse_args()

copy_and_split(
    args.image_dir,
    args.label_dir,
    args.output_image_dir,
    args.output_label_dir,
    args.background_dir
)
