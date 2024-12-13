import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from model import UNet, BCEWithPowerJaccardLoss, SegmentationDataset
from drunet import load_zipped_pickle, save_checkpoint, load_checkpoint
from sklearn.model_selection import train_test_split
import os
import numpy as np

def calculate_iou(predictions, targets, threshold=0.7):
    """
    Calculate the IoU for a batch of predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted masks (after sigmoid activation).
        targets (torch.Tensor): Ground truth masks.
        threshold (float): Threshold to binarize predictions.

    Returns:
        list: IoU for each sample in the batch.
    """
    predictions = (predictions > threshold).float()  # Binarize predictions
    intersection = (predictions * targets).sum(dim=[1, 2, 3])  # Sum intersection over H, W
    union = (predictions + targets).sum(dim=[1, 2, 3]) - intersection  # Sum union
    iou = (intersection + 1e-7) / (union + 1e-7)  # Avoid division by zero
    return iou.cpu().numpy().tolist()  # Convert to list for easier processing


# Load data
train_frames, train_masks = load_zipped_pickle('final_train_data.pkl')
train_frames, val_frames, train_masks, val_masks = train_test_split(
    train_frames, train_masks, test_size=0.2, random_state=42
)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    A.pytorch.ToTensorV2()
])

train_dataset = SegmentationDataset(train_frames, train_masks, transform=transform)
val_dataset = SegmentationDataset(val_frames, val_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Select if you want to resume training and the checkpoint path
resume_training = True
checkpoint_path = "checkpoints/unet_epoch_4.pth"

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = BCEWithPowerJaccardLoss(bce_weight=0.7, jaccard_weight=0.3, jaccard_exponent=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize variables for checkpointing
start_epoch = 0
best_val_loss = float('inf')  # Set initial best loss to a very large value

if resume_training and os.path.exists(checkpoint_path):
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

# Training loop
num_epochs = 50
for epoch in range(start_epoch,num_epochs):
    model.train()
    train_loss = 0
    train_ious = []
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Compute IoU
        train_ious += calculate_iou(torch.sigmoid(outputs), masks)
        
    train_median_iou = np.median(train_ious)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Median IoU: {train_median_iou:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        eval_ious = []
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Compute IoU
            eval_ious += calculate_iou(torch.sigmoid(outputs), masks)

    eval_median_iou = np.median(eval_ious)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Median IoU: {eval_median_iou:.4f}")

    # Checkpoint: Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir="checkpoints", name="unet")
