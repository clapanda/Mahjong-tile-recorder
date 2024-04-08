import numpy as np
import cv2
import os
from pathlib import Path

def draw_light(image, start_point, end_point, line_width=5):
    # 此函数现在只接收并处理单个起始点和结束点
    cv2.line(image, start_point, end_point, (255, 255, 255), thickness=line_width)
    return image

def generate_images(image_path, output_dir, total_images=9):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Failed to load the original image. Please check the path.")
        return

    height, width, _ = original_image.shape
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 重新设计起始点和结束点
    # 起始点：从左上到右上，然后右上到右下
    start_top_x = np.linspace(0, width, 5, endpoint=False)  # 从左上到右上
    start_right_y = np.linspace(0, height, 5, endpoint=True)  # 从右上到右下
    start_points = [(int(x), 0) for x in start_top_x] + [(width, int(y)) for y in start_right_y[1:]]

    # 结束点：从左上到左下，然后左下到右下
    end_left_y = np.linspace(0, height, 5, endpoint=False)  # 从左上到左下
    end_bottom_x = np.linspace(0, width, 5, endpoint=True)  # 从左下到右下
    end_points = [(0, int(y)) for y in end_left_y] + [(int(x), height) for x in end_bottom_x[1:]]

    # 生成图片
    for i in range(total_images):
        img_copy = original_image.copy()
        start_point = start_points[i] if i < len(start_points) else start_points[-1]
        end_point = end_points[i] if i < len(end_points) else end_points[-1]
        draw_light(img_copy, start_point, end_point, line_width=5)
        cv2.imwrite(os.path.join(output_dir, f'enhanced_image_{i+1}.jpg'), img_copy)

generate_images(r'D:\train_images\1m\1m.jpg', r'D:\enhanced_images\1m')
