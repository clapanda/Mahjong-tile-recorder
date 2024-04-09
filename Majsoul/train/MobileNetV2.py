import torch
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=16):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader if phase == 'train' else val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            epoch_acc = running_corrects / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 检查CUDA设备可用性
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        use_cuda = input("Do you want to continue with GPU? (yes/no): ")
        if use_cuda.lower() != 'yes':
            print("Training stopped.")
            return
        device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training on CPU.")
        device = torch.device("cpu")

    data_dir = r'D:\mahjong\enhanced_images'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 34)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=9, gamma=0.1)
    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device)
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()


def model():
    return None