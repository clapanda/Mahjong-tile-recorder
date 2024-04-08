import os

# 定义目录路径
base_dir = r'D:\train_images'

# 定义麻将牌的种类
tiles = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
         '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
         '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
         '1z', '2z', '3z', '4z', '5z', '6z', '7z']  # 东南西北白发中分别用1z到7z表示

# 创建目录
for tile in tiles:
    # 生成每种牌面对应的目录路径
    tile_dir = os.path.join(base_dir, tile)

    # 如果目录不存在，则创建目录
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
        print(f"Directory {tile_dir} created.")
    else:
        print(f"Directory {tile_dir} already exists.")
