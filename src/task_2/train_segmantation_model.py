import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from segmentation_models_pytorch import Unet
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomRotate90
from sklearn.metrics import jaccard_score

# Custom Dataset for SCD with LabelMe-style JSON Annotations
class SCDDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List only .jpg files in image_dir
        self.images = []
        self.masks = []
        for img_file in sorted(os.listdir(image_dir)):
            if not img_file.endswith('.jpg'):
                continue
            json_file = img_file.replace('.jpg', '.json')
            json_path = os.path.join(mask_dir, json_file)
            if not os.path.exists(json_path):
                print(f"Warning: No JSON for {img_file}")
                continue
            # Check if JSON has valid Carton polygons
            with open(json_path, 'r') as f:
                ann = json.load(f)
            if 'shapes' not in ann or not ann['shapes']:
                print(f"Warning: No shapes in {json_file}")
                continue
            has_carton = any(s.get('shape_type') == 'polygon' and s.get('label') == 'Carton' for s in ann['shapes'])
            if not has_carton:
                print(f"Warning: No Carton polygons in {json_file}")
                continue
            self.images.append(img_file)
            self.masks.append(json_file)
        # Ensure equal number of images and masks
        if len(self.images) != len(self.masks):
            raise ValueError(f"Mismatch: {len(self.images)} images, {len(self.masks)} JSONs")
        if len(self.images) == 0:
            raise ValueError("No valid images with Carton annotations found")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load JSON annotation
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        with open(mask_path, 'r') as f:
            ann = json.load(f)

        # Create binary mask from shapes (LabelMe format)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in ann.get('shapes', []):
            if shape.get('shape_type') == 'polygon' and shape.get('label') == 'Carton':
                points = np.array(shape['points'], dtype=np.float32)
                points = points.reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask

# Training Setup
train_transforms = Compose([
    Resize(256, 256),
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = Compose([
    Resize(256, 256),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Paths
train_dataset = SCDDataset(
    image_dir="/home/thang/oneclass_carton/train/images",
    mask_dir="/home/thang/oneclass_carton/train/masks",
    transform=train_transforms
)
val_dataset = SCDDataset(
    image_dir="/home/thang/oneclass_carton/val/images",
    mask_dir="/home/thang/oneclass_carton/val/masks",
    transform=val_transforms
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Batch size for CPU
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model Architecture
model = Unet(encoder_name="resnet34", classes=1, activation="sigmoid")  # CPU
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training Loop
num_epochs = 20
train_losses, val_losses, val_ious = [], [], []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            preds = (outputs > 0.5).float()
            for p, m in zip(preds.numpy(), masks.numpy()):
                val_iou += jaccard_score(m.flatten(), p.flatten())
    val_loss /= len(val_loader)
    val_iou /= len(val_dataset)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

# Save Model
torch.save(model.state_dict(), "model/box_segmentation.pth")

# Visualize Validation Results
model.eval()
val_images, val_masks, val_preds = [], [], []
with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        if i >= 2: break  # Visualize 2 samples
        preds = (model(images) > 0.5).float()
        val_images.extend(images[:2])
        val_masks.extend(masks[:2])
        val_preds.extend(preds[:2])

# Plot and Save Visualizations
for i in range(2):
    img = val_images[i].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    mask = val_masks[i].squeeze().numpy()
    pred = val_preds[i].squeeze().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(img)
    plt.imshow(pred, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"pictures/task2_validation_result_{i+1}.png")
    plt.close()

# Save Training Logs
with open("log/training_logs.txt", "w") as f:
    for epoch in range(num_epochs):
        f.write(f"Epoch {epoch+1}, Train Loss: {train_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, Val IoU: {val_ious[epoch]:.4f}\n")