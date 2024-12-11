import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from model import UNet, BCEWithPowerJaccardLoss, SegmentationDataset
from drunet import load_zipped_pickle, save_checkpoint
from sklearn.model_selection import train_test_split

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = BCEWithPowerJaccardLoss(bce_weight=0.7, jaccard_weight=0.3, jaccard_exponent=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

# Initialize variables for checkpointing
best_val_loss = float('inf')  # Set initial best loss to a very large value
checkpoint_path = "unet_checkpoint.pth"  # Path to save the checkpoint

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Checkpoint: Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir="checkpoints", name="unet")
