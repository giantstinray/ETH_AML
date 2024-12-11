import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import gzip
import pickle

# Load and preprocess functions
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# DRUNet Model
class DRUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(DRUNet, self).__init__()
        # Encoding layers
        self.enc1 = self._conv_block(in_channels, num_features)
        self.enc2 = self._conv_block(num_features, num_features * 2)
        self.enc3 = self._conv_block(num_features * 2, num_features * 4)

        # Bottleneck
        self.bottleneck = self._conv_block(num_features * 4, num_features * 8)

        # Decoding layers
        self.dec3 = self._conv_block(num_features * 4, num_features * 4)
        self.dec2 = self._conv_block(num_features * 2, num_features * 2)
        self.dec1 = self._conv_block(num_features, num_features)

        # Output layer
        self.output = nn.Conv2d(num_features, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoding path
        upsampled3 = self.upsample3(bottleneck)
        if upsampled3.size() != enc3.size():
            upsampled3 = F.interpolate(upsampled3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(upsampled3 + enc3)

        upsampled2 = self.upsample2(dec3)
        if upsampled2.size() != enc2.size():
            upsampled2 = F.interpolate(upsampled2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(upsampled2 + enc2)

        upsampled1 = self.upsample1(dec2)
        if upsampled1.size() != enc1.size():
            upsampled1 = F.interpolate(upsampled1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(upsampled1 + enc1)

        # Output
        return self.output(dec1)

# Dataset Class
class Noise2NoiseDataset(Dataset):
    def __init__(self, frames, masks=None, noise_level=0.1):
        self.frames = frames
        self.masks = masks
        self.noise_level = noise_level

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = torch.tensor(self.frames[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        noisy1 = frame + self.noise_level * torch.randn_like(frame)
        noisy2 = frame + self.noise_level * torch.randn_like(frame)
        if self.masks is not None:
            mask = torch.tensor(self.masks[idx], dtype=torch.float32).unsqueeze(0)
            return noisy1, noisy2, mask
        return noisy1, noisy2

# Loss Function
def hybrid_loss(outputs, noisy_targets, clean_targets=None, masks=None):
    mse_loss = nn.MSELoss()(outputs, noisy_targets)
    if masks is not None:
        smooth = 1e-7
        dice_loss = 1 - (2 * (outputs * masks).sum() + smooth) / (outputs.sum() + masks.sum() + smooth)
        return mse_loss + dice_loss
    return mse_loss

# Save/Load Checkpoint Functions
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints", name="drunet"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, name+f"_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load a saved checkpoint to resume training.
    :param checkpoint_path: Path to the saved checkpoint file.
    :param model: The model instance.
    :param optimizer: The optimizer instance.
    :return: model, optimizer, start_epoch, and last_loss
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    print(f"Resumed from checkpoint: {checkpoint_path}, Epoch: {start_epoch}, Loss: {last_loss}")
    return model, optimizer, start_epoch, last_loss

if __name__ == "__main__":
    # Load Data
    train_frames, train_masks = load_zipped_pickle('combined_train_data.pkl')
    train_frames, val_frames, train_masks, val_masks = train_test_split(
        train_frames, train_masks, test_size=0.2, random_state=42
    )

    # Create Datasets and DataLoaders
    train_dataset = Noise2NoiseDataset(train_frames, train_masks)
    val_dataset = Noise2NoiseDataset(val_frames, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model, optimizer, scheduler, and optionally load from checkpoint
    resume_training = True
    checkpoint_path = "checkpoints/drunet_epoch_6.pth"

    # Model, optimizer, scheduler
    model = DRUNet().cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = hybrid_loss

    # Start epoch and best validation loss
    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if enabled
    if resume_training and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

    # TensorBoard for Logging
    writer = SummaryWriter()

    # Training Loop
    for epoch in range(start_epoch, 50):  # Start from checkpoint epoch
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:  # Frames, Noisy Targets, Masks
                noisy1, noisy2, masks = batch
                noisy1, noisy2, masks = noisy1.cuda(), noisy2.cuda(), masks.cuda()
                outputs = model(noisy1)
                loss = criterion(outputs, noisy2, masks=masks)
            else:
                noisy1, noisy2 = batch
                noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()
                outputs = model(noisy1)
                loss = criterion(outputs, noisy2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    noisy1, noisy2, masks = batch
                    noisy1, noisy2, masks = noisy1.cuda(), noisy2.cuda(), masks.cuda()
                    outputs = model(noisy1)
                    loss = criterion(outputs, noisy2, masks=masks)
                else:
                    noisy1, noisy2 = batch
                    noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()
                    outputs = model(noisy1)
                    loss = criterion(outputs, noisy2)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss)

    writer.close()