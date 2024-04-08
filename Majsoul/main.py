import pygetwindow as gw
import pyautogui
import cv2
import numpy as np


def capture_window_by_title(title):
    try:
        win = gw.getWindowsWithTitle(title)[0]
    except IndexError:
        print(f"Window with title '{title}' not found.")
        return None

    if win:
        win.activate()
        pyautogui.sleep(1)
        x, y, width, height = win.left, win.top, win.width, win.height

        # 调整以只捕获下方六分之一的区域
        target_height = height // 6
        target_y = y + height - target_height

        # 根据要求调整x坐标和宽度，去除左边10%和右边27%
        adjust_left = int(width * 0.125)  # 左边需要去除的宽度
        adjust_right = int(width * 0.185)  # 右边需要去除的宽度
        new_width = int(width - (adjust_left + adjust_right))  # 新的宽度
        new_x = int(x + adjust_left)  # 新的x坐标
        target_y = int(target_y)  # 确保target_y也是整数
        target_height = int(target_height)  # 确保target_height也是整数

        img = pyautogui.screenshot(region=(new_x, target_y, new_width, target_height))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return img_cv
    else:
        return None


def preprocess_image(image):
    # 轻量级降噪（使用高斯模糊）
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised


# 使用窗口的标题作为参数
window_title = "雀魂麻將"  # 需要根据实际情况修改
captured_image = capture_window_by_title(window_title)

if captured_image is not None:
    # 显示捕获的图像
    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 对捕获的图像进行预处理
    preprocessed_image = preprocess_image(captured_image)

    # 显示预处理后的图像
    cv2.imshow("Preprocessed Image", preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
