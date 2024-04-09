import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk

def predict_images_in_folders(root_folder, model_path):
    # 加载模型
    model = models.mobilenet_v2(pretrained=False)  # 加载模型结构
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 34)  # 修改模型的输出层为34个类别
    model.load_state_dict(torch.load(model_path))  # 加载模型参数
    model.eval()  # 将模型设置为评估模式

    # 定义预处理操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 检测CUDA设备可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 遍历每个类别的文件夹
    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue  # 跳过非文件夹的项

        print(f"Predicting images in folder: {class_path}")

        # 遍历当前类别文件夹中的图像
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # 加载并预处理待预测的图像
            image = Image.open(image_path)
            input_tensor = transform(image).unsqueeze(0)  # 添加批次维度

            # 将输入数据移动到对应设备上
            input_tensor = input_tensor.to(device)
            model = model.to(device)

            # 使用模型进行预测
            with torch.no_grad():
                output = model(input_tensor)

            # 获取预测结果索引
            _, predicted_indices = torch.max(output, 1)
            predicted_index = predicted_indices.item()  # 获取单个预测结果的索引

            print(f"Predicted class index for {image_name}: {predicted_index}")


# 指定待预测的图像根文件夹路径和模型路径
root_folder = "D:/train_images"  # 替换为你的图像根文件夹路径
model_path = "model.pth"  # 替换为你的模型文件的路径

# 进行预测
predict_images_in_folders(root_folder, model_path)
