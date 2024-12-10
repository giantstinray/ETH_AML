import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataextractor import preprocess_train_data, preprocess_test_data
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

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
        self.dec3 = self._conv_block(num_features * 8, num_features * 4)
        self.dec2 = self._conv_block(num_features * 4, num_features * 2)
        self.dec1 = self._conv_block(num_features * 2, num_features)
        
        # Output layer
        self.output = nn.Conv2d(num_features, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(num_features * 4, num_features * 4, kernel_size=2, stride=2)

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
        dec3 = self.dec3(self.upsample(bottleneck) + enc3)
        dec2 = self.dec2(self.upsample(dec3) + enc2)
        dec1 = self.dec1(self.upsample(dec2) + enc1)
        
        # Output
        return self.output(dec1)

class FrameDataset(Dataset):
    def __init__(self, frames, masks=None, apply_noise=False):
        self.frames = frames
        self.masks = masks
        self.apply_noise = apply_noise

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.apply_noise:
            noisy_frame = frame + 0.1 * torch.randn_like(frame)  # Add Gaussian noise
        else:
            noisy_frame = frame

        if self.masks:
            return noisy_frame, frame  # Return noisy and clean frame pairs
        return noisy_frame

def augment_noise(frame, noise_level=0.1):
    """
    Adds Gaussian noise to a frame. Automatically handles NumPy arrays or PyTorch tensors.
    """
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)  # Convert to Tensor if input is NumPy array
    return frame + noise_level * torch.randn_like(frame)

class Noise2NoiseDataset(Dataset):
    
    def __init__(self, frames):
        """
        Initializes the Noise2Noise dataset.
        :param frames: List of frames (N, H, W, C)
        :param noise_level: Level of Gaussian noise to add
        """
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        noisy1 = augment_noise(frame)
        noisy2 = augment_noise(frame)
        return noisy1, noisy2

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"drunet_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f"Resumed from checkpoint: {checkpoint_path}, Epoch: {start_epoch}, Loss: {loss}")
    return model, optimizer, start_epoch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        """
        Early stopping to stop training when validation loss doesn't improve.
        :param patience: Number of epochs to wait for improvement before stopping.
        :param verbose: If True, prints messages when early stopping is triggered.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

    
train_frames = preprocess_train_data(select='all')[1]
train_frames, val_frames = train_test_split(train_frames, test_size=0.2, random_state=42)
# Create datasets
train_dataset = Noise2NoiseDataset(train_frames)
val_dataset = Noise2NoiseDataset(val_frames)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, loss, and optimizer
model = DRUNet().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Resuming from a checkpoint (Optional)
resume_training = False
start_epoch = 0
if resume_training:
    model, optimizer, start_epoch = load_checkpoint("checkpoints/drunet_epoch_5.pth", model, optimizer)

# Training loop with early stopping and checkpoints
best_val_loss = float('inf')
early_stopping = EarlyStopping(patience=5, verbose=True)

for epoch in range(50):  # Maximum number of epochs
    # Training phase
    model.train()
    train_loss = 0.0
    for noisy1, noisy2 in train_loader:
        noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()
        outputs = model(noisy1)
        loss = criterion(outputs, noisy2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy1, noisy2 in val_loader:
            noisy1, noisy2 = noisy1.cuda(), noisy2.cuda()
            outputs = model(noisy1)
            loss = criterion(outputs, noisy2)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss)

    # Check early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break