import tkinter as tk
from tkinter import filedialog
import pygetwindow as gw
import threading
from PIL import Image
import os
import time
from mss import mss


class ScreenCaptureApp:
    def __init__(self, master):
        self.master = master
        self.path = None
        self.is_capturing = False
        self.selected_window_title = None
        self.start_button = None  # 添加开始按钮的引用

        self.setup_ui()

    def setup_ui(self):
        self.master.title("屏幕捕获工具")
        tk.Button(self.master, text="选择目标", command=self.select_target).pack()
        self.start_button = tk.Button(self.master, text="开始", command=self.start_capture, state=tk.DISABLED)  # 初始化时禁用开始按钮
        self.start_button.pack()
        tk.Button(self.master, text="停止", command=self.stop_capture).pack()
        tk.Button(self.master, text="选择储存位置", command=self.select_path).pack()
        tk.Button(self.master, text="打开储存位置", command=self.open_path).pack()

    def select_target(self):
        # 实现选择目标窗口的功能
        top = tk.Toplevel(self.master)
        top.title("选择目标窗口")

        listbox = tk.Listbox(top)
        listbox.pack(side="left", fill="both", expand=True)

        windows = gw.getAllWindows()
        self.windows_titles = {w.title: w for w in windows if w.title}
        for title in self.windows_titles.keys():
            listbox.insert(tk.END, title)

        listbox.bind('<<ListboxSelect>>', self.on_select_window)
        tk.Button(top, text="确定", command=lambda: [self.check_ready(), top.destroy()]).pack(side="bottom")  # 使用lambda来调用check_ready


    def on_select_window(self, event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.selected_window_title = value

    def select_path(self):
        self.path = filedialog.askdirectory()
        if self.path:  # 确保用户选择了一个路径
            self.check_ready()  # 检查是否满足启用开始按钮的条件

    def check_ready(self):
        if self.selected_window_title and self.path:
            self.start_button['state'] = tk.NORMAL  # 启用开始按钮
        else:
            self.start_button['state'] = tk.DISABLED  # 禁用开始按钮

    def start_capture(self):
        if not self.selected_window_title or not self.path:
            print("Please select a target window and a save path first.")
            return

        self.is_capturing = True
        threading.Thread(target=self.capture_screen).start()

    def capture_screen(self):
        with mss() as sct:
            window = self.windows_titles.get(self.selected_window_title)
            if not window:
                print("Window not found.")
                return

            window_rect = window.box  # 获取窗口的位置和大小

            while self.is_capturing:
                sct_img = sct.grab(window_rect)

                # 将mss截图对象转换为Pillow图像并保存
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                output_filename = f"{self.path}/{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                img.save(output_filename)

                time.sleep(1)  # 每隔1秒捕获一次

    def stop_capture(self):
        self.is_capturing = False

    def open_path(self):
        if self.path:
            os.startfile(self.path)  # Windows特有的方式
        else:
            print("No path selected.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenCaptureApp(root)
    root.mainloop()
