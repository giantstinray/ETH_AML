# Load unet_epoch_31.pth and final_train_data.pkl and process the data to create a submission file.
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from dataextractor import load_zipped_pickle
from model import UNet

def load_unet_model(checkpoint_path, device='cuda'):
    """Load the pre-trained UNet model."""
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def predict_on_test_frames(model, test_frames, original_shapes, device='cuda'):
    """Predict binary masks for the test frames using the UNet model."""
    predictions = []
    for frame, original_shape in tqdm(zip(test_frames, original_shapes), total=len(test_frames)):
        # Preprocess the frame
        frame_normalized = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5  # Normalize to match training
        frame_tensor = torch.tensor(frame_normalized).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

        # Predict the mask
        with torch.no_grad():
            pred_mask = model(frame_tensor)  # Raw model output
            pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()  # Apply sigmoid and remove batch/channel dims

        # Threshold to create a binary mask
        binary_mask = (pred_mask > 0.5001).astype(np.uint8)  # Threshold at 0.5

        # Resize to the original frame shape if necessary
        resized_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        predictions.append(resized_mask)
    
    return predictions

def get_sequences(arr):
    """Extract start indices and lengths of sequences of consecutive 1s."""
    first_indices, lengths = [], []
    n, i = len(arr), 0
    arr = np.concatenate(([0], arr, [0]))  # Add padding to detect sequence ends
    for index in range(1, len(arr) - 1):
        if arr[index - 1] == 0 and arr[index] == 1:  # Start of a sequence
            first_indices.append(index - 1)  # Adjust for padding
        if arr[index] == 1 and arr[index + 1] == 0:  # End of a sequence
            lengths.append(index - first_indices[-1] + 1)  # Compute length
    return first_indices, lengths

def generate_submission(test_names, binary_masks, output_csv):
    """Generate a submission file from binary masks."""
    ids, values = [], []

    for name, mask in tqdm(zip(test_names, binary_masks), total=len(test_names)):
        flattened_mask = mask.flatten()  # Flatten the binary mask
        start_indices, lengths = get_sequences(flattened_mask)  # Extract sequences
        
        for i, (start, length) in enumerate(zip(start_indices, lengths)):
            ids.append(f"{name}_{i}")  # Format ID as name_i
            values.append(f"[{start}, {length}]")  # Format value as [flattenedIdx, len]
    
    # Create DataFrame and save as CSV
    submission_df = pd.DataFrame({"id": ids, "value": values})
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission file saved to {output_csv}")


if __name__ == '__main__':
    # Load processed test data
    test_names, test_frames = load_zipped_pickle('final_test_data.pkl')

    # Save the original shapes of the test frames
    original_shapes = [frame.shape for frame in test_frames]

    # Load the UNet model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet_model = load_unet_model('checkpoints/unet_epoch_46.pth', device=device)

    # Predict binary masks
    binary_masks = predict_on_test_frames(unet_model, test_frames, original_shapes, device=device)


output_csv = "submission.csv"
generate_submission(test_names, binary_masks, output_csv)