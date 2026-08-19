import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 7
TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
MODEL_SAVE_PATH = 'best_emotion_model.pth'  # مسیر ذخیره بهترین مدل

# ==========================================
# 2. Data Preparation (Train & Test)
# ==========================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Neural Network Architecture (CNN)
# ==========================================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

# ==========================================
# 4. Training and Evaluation Loop
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0  # متغیر برای نگه داشتن بالاترین دقت

    print("\n--- Starting Training ---", flush=True)
    
    for epoch in range(EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ----------------- EVALUATION ON TEST SET -----------------
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = 100 * test_correct / test_total

        # ----------------- SAVE BEST MODEL -----------------
        # اگر دقت این اپوک بهتر از تمام اپوک‌های قبلی باشد، مدل ذخیره می‌شود
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_marker = " (★ Best Model Saved)"
        else:
            best_marker = ""

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} - Acc: {epoch_test_acc:.2f}%{best_marker}", flush=True)

    print(f"\n--- Training Completed ---")
    print(f"Highest Test Accuracy: {best_acc:.2f}%\n")

    # ==========================================
    # 5. Load Best Model & Print Predictions
    # ==========================================
    print("--- Loading Best Saved Model for Inference ---", flush=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    class_names = test_dataset.classes
    print("\n--- Sample Individual Predictions (First 20 Images) ---", flush=True)
    sample_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                true_emotion = class_names[labels[i].item()]
                pred_emotion = class_names[predicted[i].item()]
                status = "✔ CORRECT" if true_emotion == pred_emotion else "✘ WRONG"
                
                print(f"Image {sample_count+1:02d} | True: {true_emotion:<10} | Predicted: {pred_emotion:<10} | {status}", flush=True)
                
                sample_count += 1
                if sample_count >= 20:
                    break
            if sample_count >= 20:
                break