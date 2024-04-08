import cv2
import numpy as np
import os
from pathlib import Path


def rotate_image(image, angle):
    # 旋转图像
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def flip_image(image, flip_code):
    # 翻转图像
    return cv2.flip(image, flip_code)


def apply_augmentations(image_path, output_dir):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Failed to load the original image. Please check the path.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 应用数据增强
    augmented_images = []
    angles = [90, 180, 270]  # 旋转角度
    flip_codes = [0, 1]  # 翻转代码：0为垂直翻转，1为水平翻转

    # 旋转
    for angle in angles:
        rotated = rotate_image(original_image, angle)
        augmented_images.append(rotated)

    # 翻转
    for code in flip_codes:
        flipped = flip_image(original_image, code)
        augmented_images.append(flipped)

    # 保存增强后的图像
    for i, img in enumerate(augmented_images):
        cv2.imwrite(os.path.join(output_dir, f'augmented_image_{i + 1}.jpg'), img)


# 使用示例
apply_augmentations(r'D:\train_images\1m\1m.jpg', r'D:\enhanced_images\1m')
