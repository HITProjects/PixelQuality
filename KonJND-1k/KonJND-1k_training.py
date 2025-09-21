import json
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

json_path = "KonJND-1k/patches/metadata.json"
patches_dir = "KonJND-1k/patches"


train_path = "KonJND-1k/patches/train.json"
test_path = "KonJND-1k/patches/test.json"

with open(json_path, "r") as f:
    data = json.load(f)

print(f"Total samples in JSON: {len(data)}")

missing_files = []
for sample in data:
    clean_path = os.path.join(patches_dir, sample["clean_image"])
    distorted_path = os.path.join(patches_dir, sample["distorted_image"])
    if not os.path.exists(clean_path):
        missing_files.append(sample["clean_image"])
    if not os.path.exists(distorted_path):
        missing_files.append(sample["distorted_image"])

if missing_files:
    print(f"Missing files ({len(missing_files)}):")
    for f in missing_files:
        print(" ", f)
else:
    print("✅ All images found!")

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, shuffle=True
)

with open(train_path, "w") as f:
    json.dump(train_data, f, indent=2)

with open(test_path, "w") as f:
    json.dump(test_data, f, indent=2)

print(
    f"Split {len(data)} samples into {len(train_data)} train and {len(test_data)} test samples."
)
print("Saved train.json and test.json in patches/")


class PairImageDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root = Path(root_dir)
        with open(self.root / metadata_file, "r") as f:
            self.meta = json.load(f)
        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        sample = self.meta[idx]
        img1_path = self.root / sample["clean_image"]
        img2_path = self.root / sample["distorted_image"]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        t1 = self.transform(img1)
        t2 = self.transform(img2)

        pair = torch.cat([t1, t2], dim=0)
        score = torch.tensor(sample["score"], dtype=torch.float32)

        return pair, score


dataset = PairImageDataset("KonJND-1k/patches", "train.json")
print(f"Dataset length: {len(dataset)}")
pair_tensor, score = dataset[0]
print(f"Pair shape: {pair_tensor.shape}, Score: {score}")


class SimplePairCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(20 * 20 * 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(1)


model = SimplePairCNN()
dummy_input = torch.randn(1, 6, 20, 20)
output = model(dummy_input)
print(output.shape)
print(output.item())


batch_size = 32
epochs = 10
learning_rate = 1e-3

train_dataset = PairImageDataset("KonJND-1k/patches", "train.json")
test_dataset = PairImageDataset("KonJND-1k/patches", "test.json")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SimplePairCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for pairs, scores in train_loader:
        pairs, scores = pairs.to(device), scores.to(device)
        optimizer.zero_grad()
        outputs = model(pairs).squeeze()

        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * pairs.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Train MSE Loss: {train_loss:.4f}")


def digitize_scores(scores):
    bins = [0.3, 0.6]
    return np.digitize(scores, bins)


model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for pairs, scores in test_loader:
        pairs = pairs.to(device)
        outputs = model(pairs).cpu().numpy()
        all_preds.extend(outputs)
        all_targets.extend(scores.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)


pred_classes = digitize_scores(all_preds)
true_classes = digitize_scores(all_targets)

cm = confusion_matrix(true_classes, pred_classes)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, digits=4))
