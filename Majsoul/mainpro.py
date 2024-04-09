import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pygetwindow as gw
import pyautogui
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn

# 假设你的类别标签是这样的，你需要根据实际训练的类别进行调整
CLASS_NAMES = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
               '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
               '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
               '1z', '2z', '3z', '4z', '5z', '6z', '7z']

# 定义模型架构并加载权重
# def initialize_model(num_classes=34):
#     model = models.mobilenet_v2(pretrained=False)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, num_classes)
#     return model
#
# def load_model(model_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = initialize_model(num_classes=len(CLASS_NAMES))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model, device
#
# model_path = 'train/model.pth'
# model, device = load_model(model_path)

# 图像捕获与预处理
def capture_and_show_window(title, panel):
    try:
        win = gw.getWindowsWithTitle(title)[0]
        if win:
            win.activate()
            pyautogui.sleep(1)
            x, y, width, height = win.left, win.top, win.width, win.height

            target_y = y + 450  # 调整为去除上方450像素
            target_height = height - 450 - 14  # 调整为去除下方14像素
            new_x = x + 107  # 调整为去除左侧107像素
            new_width = width - 107 - 157  # 调整为去除右侧157像素

            img = pyautogui.screenshot(region=(int(new_x), int(target_y), int(new_width), int(target_height)))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            panel.config(image=img_tk)
            panel.image = img_tk
            return img_cv
    except IndexError:
        messagebox.showerror("错误", f"未找到标题为'{title}'的窗口。")
    return None


def preprocess_image(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊去噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 调整Canny边缘检测参数
    edged = cv2.Canny(blurred, 20, 100)
    return edged


def segment_tiles(image):
    # 使用自适应二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学膨胀操作，用于增强间隙
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=2)

    # 再次计算水平投影
    horizontal_projection = np.sum(dilation, axis=0)

    # 寻找麻将牌之间的间隙
    gap_threshold = np.mean(horizontal_projection) / 2  # 设定阈值为平均值的一半
    gap_positions = np.where(horizontal_projection < gap_threshold)[0]

    # 使用间隙位置来确定麻将牌的边界
    tile_boundaries = []
    start_position = 0
    for pos in gap_positions:
        # 假定间隙宽度大于2像素为有效间隙
        if pos - start_position > 3:
            tile_boundaries.append((start_position, pos))
            start_position = pos
    tile_boundaries.append((start_position, len(horizontal_projection) - 1))  # 添加最后一张麻将牌的边界

    # 利用边界分割麻将牌
    tiles = [image[:, boundary[0]:boundary[1]] for boundary in tile_boundaries if (boundary[1] - boundary[0]) > 10]

    return tiles

def visualize_tiles(tiles):
    plt.figure(figsize=(10, 5))
    for i, tile in enumerate(tiles):
        ax = plt.subplot(1, len(tiles), i + 1)
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Tile {i+1}')
        ax.axis('off')
    plt.show()

# 示例使用



# 牌面识别
# def recognize_tiles(tiles):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     recognized_tiles = []
#     for tile in tiles:
#         tile_image_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
#         tile_image_tensor = transform(tile_image_pil).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = model(tile_image_tensor)
#             _, preds = torch.max(outputs, 1)
#             recognized_tiles.append(CLASS_NAMES[preds.item()])
#     return recognized_tiles
#
# # GUI设计
def auto_recognize():
    captured_image = capture_and_show_window("雀魂麻將", panel)
    if captured_image is not None:
        # 直接调用 segment_tiles 函数对捕获的图像进行分割
        tiles = segment_tiles(captured_image)
        # 使用 visualize_tiles 函数展示分割后的麻将牌
        visualize_tiles(tiles)
    else:
        messagebox.showerror("错误", "未能捕获游戏窗口图像。")


root = tk.Tk()
root.title("麻将牌识别")
panel = tk.Label(root)
panel.pack(padx=10, pady=10)
btn_auto_recognize = tk.Button(root, text="自动识别", command=auto_recognize)
btn_auto_recognize.pack(pady=20)
root.mainloop()
