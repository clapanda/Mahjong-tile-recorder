import pygetwindow as gw

def list_window_titles():
    # 获取所有窗口
    all_windows = gw.getAllWindows()
    # 打印每个窗口的标题
    for window in all_windows:
        print(window.title)

# 调用函数
list_window_titles()
