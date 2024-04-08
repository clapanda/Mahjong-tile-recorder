import os
import shutil
from pathlib import Path
from enhance_data_traditional import apply_augmentations as apply_traditional
from enhance_data_blink import generate_images as generate_blink


def process_images(image_dir, output_dir):
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            # 检查是否为图像文件
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)

                # 创建对应的输出目录
                relative_path = os.path.relpath(root, image_dir)
                current_output_dir = os.path.join(output_dir, relative_path)
                Path(current_output_dir).mkdir(parents=True, exist_ok=True)

                # 复制原始图像到输出目录
                shutil.copy2(image_path, os.path.join(current_output_dir, filename))

                # 应用增强
                apply_traditional(image_path, current_output_dir)
                generate_blink(image_path, current_output_dir)
                print(f"Processed and copied {filename}")


# 配置输入和输出目录
image_dir = r'D:\train_images'
output_dir = r'D:\enhanced_images'

# 开始处理图像
process_images(image_dir, output_dir)
