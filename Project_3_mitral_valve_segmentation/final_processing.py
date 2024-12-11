import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import pickle
# Load the DRUNet model definition
from drunet import DRUNet, load_zipped_pickle, load_checkpoint  # Assume this file contains your DRUNet definition
from reshape import save_zipped_pickle

def denoise_image(image, model, device='cuda'):
    """Denoise a single image using the DRUNet model."""
    model = model.to(device)
    image = image.astype(np.float32) / 255.0  # Normalize image to [0, 1]
    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    denoised_image = denoised_tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    denoised_image = (denoised_image * 255).clip(0, 255).astype(np.uint8)  # Scale back to [0, 255]
    return denoised_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)
    return enhanced_image

def smooth_mask(mask, kernel_size=5, method='gaussian'):
    """Smooth the outline of the mask and keep it binary."""
    # Ensure mask is in np.float32 for compatibility with OpenCV
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    # Apply smoothing based on the selected method
    if method == 'gaussian':
        smoothed_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    elif method == 'morphological':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif method == 'median':
        smoothed_mask = cv2.medianBlur(mask.astype(np.uint8), kernel_size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    # Threshold to ensure binary mask (0 or 1)
    _, binary_mask = cv2.threshold(smoothed_mask, 0.5, 1, cv2.THRESH_BINARY)

    # Convert back to the original dtype if needed
    return binary_mask.astype(mask.dtype)


def process_combined_data(pkl_file_path, output_pkl_path, checkpoint_path):
    """Process train frames and masks: Denoise and enhance contrast."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the DRUNet model
    drunet = DRUNet()
    optimizer = Adam(drunet.parameters(), lr=1e-4)
    drunet = load_checkpoint(checkpoint_path=checkpoint_path, model=drunet, optimizer=optimizer)[0]

    # Load from combined_train_data.pkl
    train_frames, train_masks = load_zipped_pickle(pkl_file_path)

    processed_frames = []
    processed_masks = []

    # Process each frame
    for frame, mask in tqdm(zip(train_frames, train_masks), total=len(train_frames)):
        # Denoise and apply CLAHE to the frame
        denoised_frame = denoise_image(frame, drunet, device)
        enhanced_frame = apply_clahe(denoised_frame)

        # Smooth the mask outline
        smoothed_mask = smooth_mask(mask, kernel_size=5, method='gaussian')

        # Append processed frame and mask
        processed_frames.append(enhanced_frame)
        processed_masks.append(smoothed_mask)  # Masks remain unchanged


    save_zipped_pickle((processed_frames, processed_masks), output_pkl_path)

    print(f"Processed data saved to {output_pkl_path}")

if __name__ == '__main__':
    # Paths to input .pkl file and output .pkl file
    INPUT_PKL_FILE = 'combined_train_data.pkl'
    OUTPUT_PKL_FILE = 'final_train_data.pkl'

    # Path to the DRUNet checkpoint
    CHECKPOINT_PATH = 'checkpoints/drunet_epoch_6.pth'

    # Process the images
    process_combined_data(INPUT_PKL_FILE, OUTPUT_PKL_FILE, CHECKPOINT_PATH)