import os
import shutil
import argparse
from collections import defaultdict
import tools.parameters as pr

# 固定的编号分组
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

# 文件复制主逻辑
def copy_and_split(image_dir, label_dir, output_image_dir, output_label_dir):
    groups = defaultdict(list)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in image_files:
        if "_" in file:
            base_name = file.split('_')[0]
        else:
            base_name = os.path.splitext(file)[0]
        groups[base_name].append(file)

    # 创建输出目录
    for subset in ["train", "test", "valid"]:
        os.makedirs(os.path.join(output_image_dir, subset), exist_ok=True)
        os.makedirs(os.path.join(output_label_dir, subset), exist_ok=True)

    # 执行复制
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

    # 统计
    train_count = sum(len(groups[str(i)]) for i in train_ids if str(i) in groups)
    test_count = sum(len(groups[str(i)]) for i in test_ids if str(i) in groups)
    valid_count = sum(len(groups[str(i)]) for i in valid_ids if str(i) in groups)

    print("✅ 数据集划分完成！")
    print(f"Train: {len(train_ids)} groups, {train_count} images")
    print(f"Test: {len(test_ids)} groups, {test_count} images")
    print(f"Valid: {len(valid_ids)} groups, {valid_count} images")


YOLO_save_dir_image = pr.YOLO_save_dir_image
YOLO_save_dir_label = pr.YOLO_save_dir_label

parser = argparse.ArgumentParser(description="按编号划分图像和标签为 train/test/valid")
parser.add_argument('--image_dir', type=str, default="dataset_parity/images_mix_parity_enh", help='图像源路径')
parser.add_argument('--label_dir', type=str, default="dataset_parity/labels_mix_parity", help='标签源路径')
parser.add_argument('--output_image_dir', type=str, default=YOLO_save_dir_image, help='图像输出路径')
parser.add_argument('--output_label_dir', type=str, default=YOLO_save_dir_label, help='标签输出路径')
args = parser.parse_args()

copy_and_split(args.image_dir, args.label_dir, args.output_image_dir, args.output_label_dir)