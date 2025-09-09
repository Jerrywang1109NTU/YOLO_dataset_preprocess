import os
import cv2  # 可改用 PIL.Image 根据需要
import numpy as np
import csv
import argparse

# ===================== 参数设置 =====================
# 定义不同噪声等级的 Poisson–Gaussian 参数 (α 和 σ²)
# α: 泊松噪声强度参数（模拟光子计数的尺度因子）
# σ²: 高斯噪声方差 (σ 为标准差的值平方)
NONE_ALPHA = None      # 无噪声时不应用泊松噪声
NONE_SIGMA2 = 0.0      # 无噪声时不添加高斯噪声

MILD_ALPHA = 50.0      # 轻度噪声: 较高光子计数 (泊松噪声小)
MILD_SIGMA2 = 0.001    # 轻度噪声: 很小的高斯噪声方差

MODERATE_ALPHA = 20.0  # 中度噪声: 中等光子计数
MODERATE_SIGMA2 = 0.005# 中度噪声: 中等高斯噪声方差

SEVERE_ALPHA = 5.0     # 重度噪声: 较低光子计数 (泊松噪声显著)
SEVERE_SIGMA2 = 0.02   # 重度噪声: 较大的高斯噪声方差

# 噪声级别名称列表，方便迭代
noise_levels = [
    ("none",    NONE_ALPHA,    NONE_SIGMA2),
    ("mild",    MILD_ALPHA,    MILD_SIGMA2),
    ("moderate",MODERATE_ALPHA,MODERATE_SIGMA2),
    ("severe",  SEVERE_ALPHA,  SEVERE_SIGMA2)
]

def add_poisson_gaussian_noise(image, alpha, sigma2, lock_channels=True):
    orig_dtype = image.dtype
    max_val = 255.0 if image.dtype == np.uint8 else 65535.0
    img_norm = image.astype(np.float32) / max_val

    # pick a single-channel base for noise (if image is 3-channel grayscale)
    if lock_channels and img_norm.ndim == 3:
        base = img_norm[..., :1]  # shape (H, W, 1) —— 用第一通道作为强度基
    else:
        base = img_norm

    # Poisson part
    if alpha is not None:
        lam = np.clip(base * float(alpha), 0.0, None)
        p_sample = np.random.poisson(lam).astype(np.float32) / float(alpha)
        p_term = p_sample
    else:
        p_term = base

    # Gaussian part
    if sigma2 > 0:
        sigma = float(sigma2) ** 0.5
        g = np.random.normal(0.0, sigma, size=base.shape).astype(np.float32)
    else:
        g = 0.0

    y = p_term + g  # shape: (H, W) or (H, W, 1)

    # broadcast back to match original channels
    if img_norm.ndim == 3 and y.ndim == 3 and y.shape[2] == 1:
        y = np.repeat(y, img_norm.shape[2], axis=2)

    y = np.clip(y, 0.0, 1.0)
    y = (y * max_val).astype(orig_dtype)
    return y


def process_images(input_dir, output_dir):
    """遍历输入目录，读取图像并按四档噪声等级处理后保存。"""
    # 创建输出根目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 预先创建各噪声子目录
    for level, _, _ in noise_levels:
        # 在输出路径下创建子文件夹 (exist_ok=True 保证目录不存在时创建，存在时不报错)
        os.makedirs(os.path.join(output_dir, level), exist_ok=True)

    # 打开日志文件准备记录
    log_path = os.path.join(output_dir, "log.csv")
    with open(log_path, mode="w", newline='', encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        # 写入CSV表头
        writer.writerow(["filename", "noise_type", "alpha", "sigma^2", "output_path"])

        # 遍历输入目录的文件和子目录
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                # 检查文件扩展名，确保是图像格式
                lower_name = filename.lower()
                if lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    # 构建完整输入路径
                    file_path = os.path.join(root, filename)
                    # 读取图像（保持原始格式和通道）
                    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue  # 跳过无法读取的文件

                    # 计算相对于输入根目录的相对路径，用于构建输出路径
                    rel_path = os.path.relpath(file_path, input_dir)  # 例如 'subdir/image.png'
                    
                    # 对每个噪声级别进行处理
                    for level, alpha, sigma2 in noise_levels:
                        # 构建输出文件的完整路径：输出根目录/级别/相对路径
                        out_path = os.path.join(output_dir, level, rel_path)
                        # 确保输出文件所在目录存在
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        if level == "none":
                            # 无噪声等级：直接使用原始图像
                            noisy_image = image  
                        else:
                            # 生成添加噪声后的图像
                            noisy_image = add_poisson_gaussian_noise(image, alpha, sigma2)
                        # 保存图像到指定输出路径（保持原格式，如扩展名决定格式）
                        cv2.imwrite(out_path, noisy_image)
                        # 记录日志信息：原始文件名，噪声类型，参数，输出路径
                        writer.writerow([filename, level, alpha if alpha is not None else "None", sigma2, out_path])
    print(f"处理完成！结果保存在 {output_dir}，日志记录于 {log_path}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="为图像添加不同程度的 Poisson–Gaussian 噪声")
    parser.add_argument("--input", "-i", default="dataset_0_bk/YOLO_data_8_6_0_p/images/test", help="输入图像目录路径")
    parser.add_argument("--output", "-o", default="dataset_unmod/data_test_noisy", help="输出目录路径")
    args = parser.parse_args()

    # 调用主处理函数
    process_images(args.input, args.output)

if __name__ == "__main__":
    main()
