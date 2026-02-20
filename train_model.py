import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import os

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# DATA PATH  (MAKE SURE THIS FOLDER EXISTS)
# ===============================
data_dir = "dataset"   # <-- IMPORTANT: Your folder must be named "dataset"

# ===============================
# TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===============================
# DATASETS
# ===============================
train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "val")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training folder not found: {train_path}")

if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation folder not found: {val_path}")

train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("Classes found:", train_dataset.classes)

# ===============================
# MODEL
# ===============================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ===============================
# TRAINING LOOP
# ===============================
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("weights", exist_ok=True)
model_path = "weights/resnet18_pneumonia_classifier.pth"
torch.save(model.state_dict(), model_path)

print(f"✅ Model saved successfully at {model_path}")