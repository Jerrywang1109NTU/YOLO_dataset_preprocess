import os
import shutil
from tqdm import tqdm

def generate_augmented_images_and_collect_masks(
    base_image_dir,         # 编号图像路径，如 'dataset_mod/'
    yolo_dataset_dir,       # YOLO 数据集根目录，如 'dataset_0_bk/YOLO_data_7_22_3_p'
    output_dir              # 输出目录，如 'final_output/'
):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 收集YOLO数据集中的所有图像后缀
    suffix_list = []
    for subset in ['train', 'valid', 'test']:
        image_dir = os.path.join(yolo_dataset_dir, 'images', subset)
        if not os.path.exists(image_dir):
            continue
        for f in os.listdir(image_dir):
            if not f.lower().endswith('.png'):
                continue
            name = os.path.splitext(f)[0]
            parts = name.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                suffix_list.append('_' + parts[1])  # 例如 _bk, _bk_r90

    suffix_list = sorted(set(suffix_list))  # 去重 + 排序

    # Step 2: 给 1~44 编号图片添加后缀，并保存新图像
    print("🔄 正在生成带后缀的新图像...")
    for i in tqdm(range(1, 45)):
        src_path = os.path.join(base_image_dir, f"{i}.png")
        if not os.path.exists(src_path):
            print(f"⚠️ 原图 {src_path} 不存在，跳过")
            continue
        for suffix in suffix_list:
            dst_filename = f"{i}{suffix}.png"
            dst_path = os.path.join(output_dir, dst_filename)
            shutil.copy(src_path, dst_path)

    # Step 3: 将 masks/train|valid|test 中的所有图像 flatten 到输出目录
    print("📥 正在收集 mask 图像...")
    for subset in ['train', 'valid', 'test']:
        mask_dir = os.path.join(yolo_dataset_dir, 'masks', subset)
        if not os.path.exists(mask_dir):
            continue
        for f in os.listdir(mask_dir):
            if f.lower().endswith('.png'):
                src = os.path.join(mask_dir, f)
                dst = os.path.join(output_dir, f)
                shutil.copy(src, dst)

    print("✅ 所有图像已成功处理并保存至:", output_dir)


# ========== 使用示例 ==========
if __name__ == "__main__":
    generate_augmented_images_and_collect_masks(
        base_image_dir="dataset_mod",  # 原始 1.png ~ 44.png 所在目录
        yolo_dataset_dir="dataset_0_bk/YOLO_data_7_22_3_p",  # YOLO数据集路径
        output_dir="final_output"  # 输出目录（统一收集）
    )
