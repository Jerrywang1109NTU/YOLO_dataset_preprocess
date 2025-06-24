# %%
import cv2
import numpy as np
import os
import argparse

# %% 图像增强函数定义
def Gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def Histogram(image):
    return cv2.equalizeHist(image)

def Sigmoid(image):
    image_float = image.astype(np.float32)
    x = (image_float - 128) / 128
    image_sigmoid = 1 / (1 + np.exp(-x)) * 128 + 128
    return np.clip(image_sigmoid, 0, 255).astype('uint8')

def CLAHE_Enh(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def Enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return Histogram(Gamma(Sigmoid(gray), 1.5))

# %% 批量增强函数
def Enhance_all(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                image = cv2.imread(input_path)
                enhanced = Enhance(image)
                enhanced_final = CLAHE_Enh(enhanced)
                cv2.imwrite(output_path, enhanced_final)
                print(f"✅ 处理完成: {filename}")
            except Exception as e:
                print(f"❌ 处理失败: {filename}，错误: {e}")

# %% 主入口，加上 argparse 支持
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量图像增强脚本")
    parser.add_argument("--input_folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output_folder", type=str, required=True, help="增强图像输出路径")
    args = parser.parse_args()

    Enhance_all(args.input_folder, args.output_folder)

#python Img_Enhance.py --input_folder dataset_mixed/images_mix_random --output_folder dataset_mixed/images_mix_random_enh
