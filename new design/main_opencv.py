import cv2
import numpy as np
import os
from tkinter import Tk, Label, Button, messagebox
from PIL import ImageGrab, Image, ImageTk
import pygetwindow as gw
import pyautogui

# 1. 加载模板图像
def load_templates(template_path):
    templates = {}
    for dirpath, dirnames, filenames in os.walk(template_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # 获取所在文件夹的名称
                folder_name = os.path.basename(dirpath)
                tile_name = os.path.splitext(filename)[0]
                tile_image = cv2.imread(os.path.join(dirpath, filename), cv2.IMREAD_GRAYSCALE)
                # 保存模板图像和其所在文件夹名称
                templates[tile_name] = (tile_image, folder_name)
    return templates


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
def recognize_tiles(tiles, templates):
    predictions = []
    for tile in tiles:
        max_score = 0
        best_match = None
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        for name, (template, folder_name) in templates.items():
            res = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_score:
                max_score = max_val
                best_match = folder_name  # 使用文件夹名称而不是文件名
        if best_match:
            predictions.append(best_match)
    return predictions


def format_prediction_string(predictions):
    # 初始化存储每种类型牌的字典
    sorted_tiles = {'m': [], 'p': [], 's': [], 'z': []}

    for pred in predictions:
        if len(pred) < 2 or pred[-1] not in sorted_tiles:
            print(f"Invalid prediction format: {pred}")
            continue  # 跳过无效格式的预测
        number = pred[:-1]
        tile = pred[-1]
        sorted_tiles[tile].append(number)  # 注意改为append，因为这里number是一个字符串，不是列表

    # 对每种类型牌的数字进行排序，并转换为字符串
    for tile in sorted_tiles:
        sorted_tiles[tile] = ''.join(sorted(sorted_tiles[tile], key=lambda x: int(x)))

    # 按照mpsz的顺序合并结果字符串
    result_string = ''.join([f"{''.join(sorted_tiles[tile])}{tile}" for tile in 'mpsz' if sorted_tiles[tile]])
    return result_string

# GUI界面设置
def setup_gui():
    root = Tk()
    root.title("Mahjong Tile Recognizer")
    panel = Label(root)  # 用于显示手牌区域的图像
    panel.pack(padx=10, pady=10)

    templates = load_templates('D:/mahjong/train')

    def capture_and_recognize():
        img_cv = capture_hand_area("雀魂麻將", panel)
        if img_cv is not None:
            tiles = find_tiles(img_cv)  # 从手牌区域图像中识别并分割出所有麻将牌
            if tiles:  # 如果成功找到了tiles
                predictions = recognize_tiles(tiles, templates)  # 注意这里传入的是模板图像
                formatted_predictions = format_prediction_string(predictions)  # 格式化预测结果
                messagebox.showinfo("Prediction", formatted_predictions)  # 展示格式化后的预测结果
            else:
                messagebox.showerror("Error", "No tiles found.")  # 如果未找到任何tiles，显示错误信息
        else:
            messagebox.showerror("Error", "Failed to capture hand area.")  # 如果未能成功捕获手牌区域，显示错误信息

    btn_capture = Button(root, text="识别", command=capture_and_recognize)
    btn_capture.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()