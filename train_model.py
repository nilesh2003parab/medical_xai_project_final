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
# Kaggle dataset extracts to: chest_xray/train, chest_xray/val, chest_xray/test
# ============================================================
data_dir = "chest_xray"

train_path = os.path.join(data_dir, "train")
val_path   = os.path.join(data_dir, "val")
test_path  = os.path.join(data_dir, "test")

for p in [train_path, val_path, test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Folder not found: {p}\n"
            "Make sure you extracted the Kaggle dataset so you have:\n"
            "  chest_xray/train/NORMAL/\n"
            "  chest_xray/train/PNEUMONIA/\n"
            "  chest_xray/val/NORMAL/\n"
            "  chest_xray/val/PNEUMONIA/"
        )

# ============================================================
# TRANSFORMS
# Kaggle chest x-ray images vary in size — resize + augment for training
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# DATASETS
# ImageFolder maps folders alphabetically:
#   NORMAL    -> class index 0
#   PNEUMONIA -> class index 1
# ============================================================
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_path,   transform=val_transform)
test_dataset  = datasets.ImageFolder(test_path,  transform=val_transform)

print(f"Classes: {train_dataset.classes}")   # ['NORMAL', 'PNEUMONIA']
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ============================================================
# HANDLE CLASS IMBALANCE
# Kaggle dataset has ~3x more PNEUMONIA than NORMAL in train set
# Use WeightedRandomSampler to balance batches
# ============================================================
targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=torch.FloatTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,    num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False,    num_workers=2)

# ============================================================
# MODEL — ResNet18 with pretrained ImageNet weights
# ============================================================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)   # 2 classes: NORMAL, PNEUMONIA
model = model.to(device)

# ============================================================
# LOSS & OPTIMIZER
# Weight loss to further handle imbalance
# ============================================================
class_weight_tensor = torch.FloatTensor(class_weights / class_weights.sum() * 2).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ============================================================
# TRAINING LOOP
# ============================================================
epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    # -- Train --
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # -- Validate --
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    val_acc   = 100 * val_correct   / val_total
    avg_loss  = train_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), "weights/resnet18_pneumonia_classifier.pth")
        print(f"  ✅ Best model saved (Val Acc: {val_acc:.2f}%)")

    scheduler.step()

# ============================================================
# TEST SET EVALUATION
# ============================================================
print("\n--- Test Set Evaluation ---")
checkpoint = torch.load("weights/resnet18_pneumonia_classifier.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

test_correct, test_total = 0, 0
tp = tn = fp = fn = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total   += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        for p, l in zip(predicted, labels):
            if p == 1 and l == 1: tp += 1
            elif p == 0 and l == 0: tn += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 1: fn += 1

test_acc   = 100 * test_correct / test_total
precision  = tp / (tp + fp + 1e-8)
recall     = tp / (tp + fn + 1e-8)
f1         = 2 * precision * recall / (precision + recall + 1e-8)
specificity = tn / (tn + fp + 1e-8)

print(f"Test Accuracy  : {test_acc:.2f}%")
print(f"Precision      : {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity    : {specificity:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"\nBest model saved to: weights/resnet18_pneumonia_classifier.pth")
