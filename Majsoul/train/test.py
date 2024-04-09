import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel, messagebox
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "1m", "1p", "1s", "1z", "2m", "2p", "2s", "2z",
    "3m", "3p", "3s", "3z", "4m", "4p", "4s", "4z",
    "5m", "5p", "5s", "5z", "6m", "6p", "6s", "6z",
    "7m", "7p", "7s", "7z", "8m", "8p", "8s", "9m",
    "9p", "9s"
]


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def evaluate_image(image_path, model, device):
    original_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.topk(outputs, 4)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        top_preds = [(CLASS_NAMES[pred], probabilities[pred].item()) for pred in preds[0]]

    # 可视化处理步骤
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    # 将Tensor转换回图像进行展示
    np_image = image_tensor.squeeze().cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = np_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    np_image = np.clip(np_image, 0, 1)

    plt.subplot(1, 2, 2)
    plt.imshow(np_image)
    plt.title("Processed Image")
    plt.axis('off')
    plt.show()

    return top_preds, original_image


def select_image(model, device):
    file_path = filedialog.askopenfilename()
    if file_path:
        top_preds, original_image = evaluate_image(file_path, model, device)  # 确保此处original_image是PIL图像对象
        display_results(top_preds, original_image)  # 传递PIL图像对象，而非路径

def display_results(predictions, original_image):
    window = Toplevel()
    window.title("Model Predictions")

    # 确保original_image已经是一个PIL Image对象
    img = ImageTk.PhotoImage(original_image)
    panel = Label(window, image=img)
    panel.image = img  # 保留对图像的引用，防止被垃圾回收
    panel.pack()

    for pred in predictions:
        label = Label(window, text=f"{pred[0]}: {pred[1]:.2f}%")
        label.pack()

    window.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model Performance Checker")
    model_path = 'model.pth'
    model, device = load_model(model_path)

    btn_load = Button(root, text="Load Image", command=lambda: select_image(model, device))
    btn_load.pack(pady=20)

    root.mainloop()