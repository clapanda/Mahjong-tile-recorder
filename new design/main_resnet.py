import os

import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageTk
import pygetwindow as gw
import pyautogui
from tkinter import Tk, Label, Button, messagebox
import torch
from torchvision import models, transforms
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt

# 定义麻将牌的类别标签
CLASS_NAMES = [
    "1m", "1p", "1s", "1z", "2m", "2p", "2s", "2z",
    "3m", "3p", "3s", "3z", "4m", "4p", "4s", "4z",
    "5m", "5p", "5s", "5z", "6m", "6p", "6s", "6z",
    "7m", "7p", "7s", "7z", "8m", "8p", "8s", "9m",
    "9p", "9s"
]

# 加载模型
# 加载模型
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用ResNet50模型
    model = models.resnet50(pretrained=False)
    # 获取fc层的输入特征数
    num_ftrs = model.fc.in_features
    # 替换fc层，适应你的输出类别数
    model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    # 加载你的预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


model, device = load_model('resnet_model.pth')

# 捕获手牌区域
def capture_hand_area(title, panel):
    try:
        win = gw.getWindowsWithTitle(title)[0]
        if win:
            win.activate()
            pyautogui.sleep(1)

            # 获取窗口位置和大小
            x, y, width, height = win.left, win.top, win.width, win.height

            # 调整手牌区域的截取范围
            target_y = y + 450  # 调整为去除上方450像素
            target_height = height - 450 - 14  # 调整为去除下方14像素
            new_x = x + 107  # 调整为去除左侧107像素
            new_width = width - 107 - 160  # 调整为去除右侧160像素

            # 截取手牌区域的图像
            img = pyautogui.screenshot(region=(int(new_x), int(target_y), int(new_width), int(target_height)))

            # 将图像转换为 OpenCV 格式
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # 在窗口中显示图像
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            panel.config(image=img_tk)
            panel.image = img_tk

            print("Hand area captured.")
            return img_cv
        else:
            print("Game window not found.")
            messagebox.showerror("Error", "Game window not found.")
            return None
    except IndexError:
        messagebox.showerror("错误", f"未找到标题为'{title}'的窗口。")
        return None

def find_tiles(image, expected_tile_size=(39, 60)):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义白色的HSV范围
    lower_white = np.array([0, 0, 221], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)

    # 根据白色的阈值找到白色区域
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 检查找到的轮廓是否接近预期的麻将牌大小
        if abs(w - expected_tile_size[0]) < 5 and abs(h - expected_tile_size[1]) < 5:
            # 裁剪麻将牌区域
            tile = image[y:y + h, x:x + w]

            # 如果需要，调整tile大小到期望的大小
            if tile.shape[:2] != expected_tile_size:
                tile = cv2.resize(tile, expected_tile_size)

            tiles.append(tile)

    return tiles


# 识别麻将牌
def recognize_tiles(tiles, model, device):
    predictions = []

    for i, tile in enumerate(tiles):
        # 将numpy.ndarray转换为PIL图像
        original_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))


        # 应用预处理转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tile_tensor = transform(original_image).unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            outputs = model(tile_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = CLASS_NAMES[preds.item()]
            predictions.append(pred_class)

    return predictions


def format_prediction_string(predictions):
    # 初始化存储每种类型牌的字典
    sorted_tiles = {'m': [], 'p': [], 's': [], 'z': []}

    # 将预测分组并存储在相应的列表中
    for pred in predictions:
        number = pred[:-1]
        tile = pred[-1]
        sorted_tiles[tile].extend(number)  # 将数字添加到相应的列表

    # 对每种类型牌的数字进行排序
    for tile in sorted_tiles:
        sorted_tiles[tile] = ''.join(sorted(sorted_tiles[tile]))

    # 按照mpsz的顺序合并结果字符串
    result_string = ''.join([sorted_tiles[tile] + tile for tile in 'mpsz' if sorted_tiles[tile]])
    return result_string
# GUI界面设置
def setup_gui():
    root = Tk()
    root.title("Mahjong Tile Recognizer")
    panel = Label(root)  # 用于显示手牌区域的图像
    panel.pack(padx=10, pady=10)

    def capture_and_recognize():
        img_cv = capture_hand_area("雀魂麻將", panel)
        if img_cv is not None:
            tiles = find_tiles(img_cv)  # 从手牌区域图像中识别并分割出所有麻将牌
            if tiles:  # 如果成功找到了tiles
                predictions = recognize_tiles(tiles, model, device)  # 对每个tile进行识别
                formatted_predictions = format_prediction_string(
                    predictions)  # 使用format_prediction_string格式化predictions
                messagebox.showinfo("Prediction", formatted_predictions)  # 使用格式化后的字符串展示识别结果
            else:
                messagebox.showerror("Error", "No tiles found.")  # 如果未找到任何tiles，显示错误信息
        else:
            messagebox.showerror("Error", "Failed to capture hand area.")  # 如果未能成功捕获手牌区域，显示错误信息

    btn_capture = Button(root, text="识别", command=capture_and_recognize)
    btn_capture.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()