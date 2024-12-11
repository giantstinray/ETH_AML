import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = UNet._block(in_channels, features)
        self.encoder2 = UNet._block(features, features * 2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.encoder4 = UNet._block(features * 4, features * 8)

        self.bottleneck = UNet._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        dec4 = self.upconv4(bottleneck)

        # Ensure dec4 matches enc4 dimensions
        if dec4.size(2) != enc4.size(2) or dec4.size(3) != enc4.size(3):
            dec4 = F.pad(dec4, [0, enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2)])

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.size(2) != enc3.size(2) or dec3.size(3) != enc3.size(3):
            dec3 = F.pad(dec3, [0, enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2)])

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.size(2) != enc2.size(2) or dec2.size(3) != enc2.size(3):
            dec2 = F.pad(dec2, [0, enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2)])

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.size(2) != enc1.size(2) or dec1.size(3) != enc1.size(3):
            dec1 = F.pad(dec1, [0, enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2)])

        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))


    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class BCEWithPowerJaccardLoss(nn.Module):
    def __init__(self, bce_weight=0.7, jaccard_weight=0.3, jaccard_exponent=2):
        super(BCEWithPowerJaccardLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight
        self.jaccard_exponent = jaccard_exponent

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)

        # Power Jaccard Loss
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        union = (inputs.pow(self.jaccard_exponent) + targets.pow(self.jaccard_exponent)).sum() - intersection
        jaccard_loss = 1 - (intersection + 1e-7) / (union + 1e-7)

        # Combined Loss
        combined_loss = self.bce_weight * bce_loss + self.jaccard_weight * jaccard_loss
        return combined_loss


# Training pipeline

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask


