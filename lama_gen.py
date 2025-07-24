import os
import cv2
import numpy as np
from tqdm import tqdm

# 判定方向函数
def judge_direction(x, y):
    dx = x - 0.5
    dy = y - 0.5
    if abs(dy) > abs(dx):
        return 0 if y < 0.5 else 2  # 上 or 下
    else:
        return 3 if x < 0.5 else 1  # 左 or 右

# 生成 mask 主函数
def generate_directional_masks(image_root, w_b=15, h_b=15, w_g=15, h_g=40):
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        image_dir = os.path.join(image_root, 'images', subset)
        label_dir = os.path.join(image_root, 'labels', subset)
        mask_dir = os.path.join(image_root, 'masks', subset)
        os.makedirs(mask_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files, desc=f'Processing {subset}'):
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
            mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '_mask001.png')

            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 跳过无效图像: {img_path}")
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            if not os.path.exists(label_path):
                cv2.imwrite(mask_path, mask)
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, cx, cy, _, _ = parts
                cx = float(cx)
                cy = float(cy)
                direction = judge_direction(cx, cy)

                if '0' in cls.lower():
                    box_w = w_b
                    box_h = h_b
                elif '1' in cls.lower():
                    box_w = w_g
                    box_h = h_g
                else:
                    continue

                if direction in [1, 3]:  # 左右
                    box_w, box_h = box_h, box_w

                x1 = int(cx * w - box_w / 2)
                y1 = int(cy * h - box_h / 2)
                x2 = int(cx * w + box_w / 2)
                y2 = int(cy * h + box_h / 2)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

            cv2.imwrite(mask_path, mask)

# ========== 用法 ==========
if __name__ == "__main__":
    dataset_root = "dataset_0_bk/LAMA"  # 替换为你的路径
    output_root = "dataset_0_bk/LAMA/masks"  # 替换为你的路径
    generate_directional_masks(dataset_root, w_b=15, h_b=15, w_g=25, h_g=40)
