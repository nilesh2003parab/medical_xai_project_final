import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATASET PATH
# Kaggle Brain Tumor MRI dataset structure:
#   brain_tumor/Training/glioma/
#   brain_tumor/Training/meningioma/
#   brain_tumor/Training/notumor/
#   brain_tumor/Training/pituitary/
#   brain_tumor/Testing/glioma/  ... etc
# ============================================================
data_dir    = "brain_tumor"
train_path  = os.path.join(data_dir, "Training")
test_path   = os.path.join(data_dir, "Testing")

for p in [train_path, test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Folder not found: {p}\n"
            "Extract the Kaggle dataset so you have:\n"
            "  brain_tumor/Training/glioma/\n"
            "  brain_tumor/Training/meningioma/\n"
            "  brain_tumor/Training/notumor/\n"
            "  brain_tumor/Training/pituitary/\n"
            "  brain_tumor/Testing/  (same 4 classes)"
        )

# ============================================================
# TRANSFORMS
# Brain MRI images are grayscale but stored as RGB JPG
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# DATASETS
# ImageFolder sorts alphabetically:
#   glioma=0, meningioma=1, notumor=2, pituitary=3
# ============================================================
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_path,  transform=test_transform)

# Use 10% of training data as validation
val_size   = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Classes : {train_dataset.dataset.classes}")
print(f"Train   : {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

# ============================================================
# HANDLE CLASS IMBALANCE with WeightedRandomSampler
# ============================================================
targets       = np.array(train_dataset.dataset.targets)[train_dataset.indices]
class_counts  = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=torch.FloatTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,   num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False,   num_workers=2)

# ============================================================
# MODEL — ResNet18 pretrained, 4-class output
# ============================================================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)   # 4 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ============================================================
# TRAINING
# ============================================================
epochs       = 10
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_loss = train_correct = train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Validation
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    val_acc   = 100 * val_correct   / val_total
    avg_loss  = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), "weights/resnet18_brain_tumor.pth")
        print(f"  ✅ Best model saved (Val: {val_acc:.2f}%)")

    scheduler.step()

# ============================================================
# TEST EVALUATION
# ============================================================
print("\n--- Test Set Evaluation ---")
model.load_state_dict(torch.load("weights/resnet18_brain_tumor.pth", map_location=device))
model.eval()

class_names   = train_dataset.dataset.classes
num_classes   = len(class_names)
class_correct = [0] * num_classes
class_total   = [0] * num_classes
test_correct  = test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total   += labels.size(0)
        for p, l in zip(predicted, labels):
            class_correct[l.item()] += (p == l).item()
            class_total[l.item()]   += 1

print(f"Overall Test Accuracy: {100*test_correct/test_total:.2f}%\n")
for i, cls in enumerate(class_names):
    acc = 100 * class_correct[i] / (class_total[i] + 1e-8)
    print(f"  {cls:12s}: {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")

print(f"\nModel saved to: weights/resnet18_brain_tumor.pth")
