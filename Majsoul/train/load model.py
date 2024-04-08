import torch
from torchvision import models
import torch.nn as nn

# 定义模型架构，确保与训练时完全一致
model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 34)  # 34为您的类别数

# 加载训练好的模型权重
model.load_state_dict(torch.load('majsoul_model.pth'))
model.eval()  # 设置为评估模式
