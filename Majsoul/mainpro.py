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
def initialize_model(num_classes=34):
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model_path = 'train/model.pth'
model, device = load_model(model_path)

# 图像捕获与预处理
def capture_window_by_title(title, panel):
    try:
        win = gw.getWindowsWithTitle(title)[0]
        if win:
            win.activate()
            pyautogui.sleep(1)
            x, y, width, height = win.left, win.top, win.width, win.height

            # 根据提供的测量信息计算捕获区域的具体尺寸
            target_y = y + 450  # 去除上方450像素
            target_height = height - 450 - 14  # 去除下方14像素
            new_x = x + 107  # 去除左侧107像素
            new_width = width - 107 - 157  # 去除右侧157像素

            img = pyautogui.screenshot(region=(int(new_x), int(target_y), int(new_width), int(target_height)))

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
            img_tk = ImageTk.PhotoImage(image=img_pil)  # 转换为Tkinter图像
            panel.config(image=img_tk)  # 显示图像
            panel.image = img_tk  # 防止图像被垃圾收集器回收
            return img_cv
    except IndexError:
        print(f"Window with title '{title}' not found.")
    return None

def preprocess_image(image):
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

# 图像分割
import cv2
from matplotlib import pyplot as plt


def segment_tiles(preprocessed_image):
    tile_width = 40  # 每张牌的宽度是40像素
    gap_between_tiles = 2  # 牌与牌之间的间隔是2像素
    large_gap = 12  # 第13张和第14张牌之间的间隔是12像素
    tiles = []

    # 分割前13张牌
    for i in range(13):
        start_x = i * (tile_width + gap_between_tiles)
        tile = preprocessed_image[:, start_x:start_x + tile_width]
        tiles.append(tile)

    # 分割第14张牌
    start_x_14th = 13 * (tile_width + gap_between_tiles) + large_gap
    tile_14th = preprocessed_image[:, start_x_14th:start_x_14th + tile_width]
    tiles.append(tile_14th)

    # 显示每张分割后的牌
    for i, tile in enumerate(tiles):
        plt.figure(figsize=(2, 2))
        plt.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        plt.title(f'Tile {i+1}')
        plt.axis('off')
        plt.show()

    return tiles


# 牌面识别
def recognize_tiles(tiles):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    recognized_tiles = []
    for tile in tiles:
        tile_image_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        tile_image_tensor = transform(tile_image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tile_image_tensor)
            _, preds = torch.max(outputs, 1)
            recognized_tiles.append(CLASS_NAMES[preds.item()])
    return recognized_tiles

# GUI设计
def auto_recognize():
    captured_image = capture_window_by_title("雀魂麻將", panel)
    if captured_image is not None:
        preprocessed_image = preprocess_image(captured_image)
        tiles = segment_tiles(preprocessed_image)
        recognized_tiles = recognize_tiles(tiles)
        combined_str = ' '.join(recognized_tiles)
        messagebox.showinfo("识别结果", combined_str)
    else:
        messagebox.showerror("错误", "未能捕获游戏窗口图像")

root = tk.Tk()
root.title("麻将牌识别")
panel = tk.Label(root)
panel.pack(padx=10, pady=10)
btn_auto_recognize = tk.Button(root, text="自动识别", command=auto_recognize)
btn_auto_recognize.pack(pady=20)
root.mainloop()
