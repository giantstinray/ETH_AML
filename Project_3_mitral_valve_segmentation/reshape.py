import pickle
import gzip
import numpy as np
from dataextractor import preprocess_train_data, resize_with_mask, pad_or_crop_with_mask

# Reshape the training data. 
# For the amateur data, resize the frames and masks to 512x512 and pad to 700x700.
# For the expert data, pad or crop the frames and masks to 700x700.

_, train_frames_amateur, train_masks_amateur = preprocess_train_data(select='amateur')
_, train_frames_expert, train_masks_expert = preprocess_train_data(select='expert')
train_frames_expert = [frame[:, :, 0] for frame in train_frames_expert]
train_masks_expert = [mask[:, :, 0] for mask in train_masks_expert]
train_frames_amateur, train_masks_amateur = resize_with_mask(train_frames_amateur, train_masks_amateur, target_size=(512, 512))
train_frames_amateur, train_masks_amateur = pad_or_crop_with_mask(train_frames_amateur, train_masks_amateur, target_size=(700, 700))
train_frames_expert, train_masks_expert = pad_or_crop_with_mask(train_frames_expert, train_masks_expert, target_size=(700, 700))

# Combine amateur and expert data
train_frames = train_frames_amateur + train_frames_expert
train_masks = train_masks_amateur + train_masks_expert

# Save combined data
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

save_zipped_pickle((train_frames, train_masks), "combined_train_data.pkl")