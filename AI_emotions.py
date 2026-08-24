import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# ۱. محدود کردن نخ‌های PyTorch برای جلوگیری از مصرف رم
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ==========================================
# 1. Fast CNN Architecture
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
# 2. Global Initialization (Lazy)
# ==========================================
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# مدل رو اینجا فقط تعریف می‌کنیم اما بارگذاری نمی‌کنیم (None می‌ذاریم)
_MODEL = None

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),  # مصرف رم را ۲۲ برابر کم می‌کند
    transforms.ToTensor()
])

def get_model():
    """بارگذاری تنبل: مدل فقط زمانی لود می‌شود که برای اولین بار فراخوانی شود"""
    global _MODEL
    if _MODEL is None:
        print("Lazy Loading: Initializing and loading model weights into RAM...", flush=True)
        _MODEL = FastEmotionCNN(num_classes=7)
        try:
            _MODEL.load_state_dict(torch.load('best_emotion_model.pth', map_location=_DEVICE, weights_only=True))
            print("PyTorch model loaded successfully.", flush=True)
        except FileNotFoundError:
            print("Warning: 'best_emotion_model.pth' not found. Running with uninitialized weights.", flush=True)
        
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    
    return _MODEL

# ==========================================
# 3. Fast Batch Inference Function
# ==========================================
def tell_emotions_batch(faces, batch_size=4):  
    if faces is None or len(faces) == 0:
        return np.array([])
    
    all_emotions = []
    print("qh1: Starting emotion batching...", flush=True)
    
    # دریافت مدل (اگر بار اول باشد لود می‌شود، دفعات بعد از رم می‌خواند)
    model = get_model()
    
    with torch.no_grad():
        for i in range(0, len(faces), batch_size):
            batch_faces = faces[i:i + batch_size]
            
            tensors = [_TRANSFORM(face) for face in batch_faces]
            batch_tensor = torch.stack(tensors).to(_DEVICE)
            
            print(f"qh2: Running batch {i//batch_size + 1} with tensor shape {batch_tensor.shape}...", flush=True)
            
            # استفاده از مدل
            outputs = model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1).tolist()
            
            all_emotions.extend([EMOTION_CLASSES[idx] for idx in predictions])
            
    print("qh3: Emotion detection completed successfully!", flush=True)
    return np.array(all_emotions)

# ==========================================
# 3. Training Script (Isolated for Direct Execution Only)
# ==========================================
if __name__ == '__main__':
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    BATCH_SIZE = 64          
    LEARNING_RATE = 0.003
    EPOCHS = 20
    NUM_CLASSES = 7
    IMAGE_SIZE = 48
    TRAIN_DIR = './dataset/train'
    TEST_DIR = './dataset/test'
    MODEL_SAVE_PATH = 'best_emotion_model.pth'

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')
    
    print(f"Using device: {device}", flush=True)

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
