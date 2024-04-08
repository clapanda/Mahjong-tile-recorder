import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pygetwindow as gw
import pyautogui
import cv2
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

            target_height = height // 6
            target_y = y + height - target_height

            adjust_left = width * 0.1  # 根据新逻辑调整宽度
            adjust_right = width * 0.27
            new_width = width - (adjust_left + adjust_right)
            new_x = x + adjust_left

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
def segment_tiles(preprocessed_image, pixels_per_cm=50):
    tile_width_pixels, gap_width_pixels = int(0.71 * pixels_per_cm), int(0.2 * pixels_per_cm)
    tiles = []
    for i in range(13):  # 前13张牌紧密相连
        start_x = i * tile_width_pixels
        tiles.append(preprocessed_image[:, start_x:start_x + tile_width_pixels])
    start_x_14th = 13 * tile_width_pixels + gap_width_pixels  # 第14张牌与前面有间隔
    tiles.append(preprocessed_image[:, start_x_14th:start_x_14th + tile_width_pixels])
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
