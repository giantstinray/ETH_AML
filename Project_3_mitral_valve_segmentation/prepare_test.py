from dataextractor import load_zipped_pickle, save_zipped_pickle, preprocess_test_data
from final_processing import denoise_image, apply_clahe
from drunet import DRUNet, load_checkpoint
from torch.optim import Adam
import torch
from tqdm import tqdm
import numpy as np

def prepare_test_data(test_frames, output_pkl_path, checkpoint_path):
    """Process test frames: Denoise and enhance contrast."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the DRUNet model
    drunet = DRUNet()
    optimizer = Adam(drunet.parameters(), lr=1e-4)
    drunet = load_checkpoint(checkpoint_path=checkpoint_path, model=drunet, optimizer=optimizer)[0]

    processed_frames = []

    # Process each frame
    for frame in tqdm(test_frames, total=len(test_frames)):
        # Denoise and apply CLAHE to the frame
        denoised_frame = denoise_image(frame, drunet, device)
        enhanced_frame = apply_clahe(denoised_frame)

        # Append processed frame and mask
        processed_frames.append(enhanced_frame)

    save_zipped_pickle((test_names, processed_frames), output_pkl_path)

    print(f"Processed data saved to {output_pkl_path}")

test_names, test_frames = preprocess_test_data()
# Remove the channel dimension from each frame
test_frames = [np.squeeze(frame, axis=-1) for frame in test_frames]
prepare_test_data(test_frames, "final_test_data.pkl", "checkpoints/drunet_epoch_6.pth")

