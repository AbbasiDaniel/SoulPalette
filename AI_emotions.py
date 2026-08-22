import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
# ==========================================
# 1. Hyperparameters
# ==========================================
BATCH_SIZE = 64          
LEARNING_RATE = 0.003
EPOCHS = 20
NUM_CLASSES = 7
IMAGE_SIZE = 48
TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
MODEL_SAVE_PATH = 'best_emotion_model.pth'

# ==========================================
# 2. Data Preparation
# ==========================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

# ==========================================
# 3. Fast CNN Architecture
# ==========================================
class FastEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FastEmotionCNN, self).__init__()
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(
            conv_block(1, 32),    # 24x24
            conv_block(32, 64),   # 12x12
            conv_block(64, 128),  # 6x6
            conv_block(128, 256)  # 3x3
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ==========================================
# 4. Training Loop (Clean & Warning-Free)
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')
    
    print(f"Using device: {device}", flush=True)
    if not use_cuda:
        print("cpu in use", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_cuda)

    model = FastEmotionCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    best_acc = 0.0

    print("\n--- Starting Training ---", flush=True)
    
    for epoch in range(EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total

        # ----------------- EVALUATION -----------------
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        epoch_test_loss = test_loss / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_marker = " (★ Saved)"
        else:
            best_marker = ""

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} - Test Acc: {test_acc:.2f}%{best_marker}", flush=True)

    print(f"\n--- Training Completed ---")
    print(f"Highest Test Accuracy: {best_acc:.2f}%\n")

_MODEL = None
_DEVICE = None

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def tell_emotion(face):
    global _MODEL, _DEVICE
    
    if _MODEL is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _MODEL = FastEmotionCNN(num_classes=7)
        _MODEL.load_state_dict(torch.load('best_emotion_model.pth', map_location=_DEVICE, weights_only=True))
        _MODEL.to(_DEVICE)
        _MODEL.eval()

    tensor = torch.from_numpy(face).float()
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.mean(dim=0, keepdim=True) 
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(48, 48), mode='bilinear', align_corners=False)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    tensor = tensor.to(_DEVICE)

    with torch.no_grad():
        outputs = _MODEL(tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return EMOTION_CLASSES[predicted_idx.item()]
