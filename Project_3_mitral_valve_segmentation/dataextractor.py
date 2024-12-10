import pickle
import gzip
import numpy as np
import os
from tqdm import tqdm
import cv2

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def preprocess_item(item):
    '''
    This function preprocesses an item. It returns the names of the item, the video frames and the mask frames.
    '''
    item_video_frames = []
    item_mask_frames = []
    item_names = []
    video = item['video']
    name = item['name']
    height, width, n_frames = video.shape
    mask = np.zeros((height, width, n_frames), dtype=bool)
    for frame in item['frames']:
        mask[:, :, frame] = item['label'][:, :, frame]
        video_frame = video[:, :, frame]
        mask_frame = mask[:, :, frame]
        video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
        mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)
        item_video_frames.append(video_frame)
        item_mask_frames.append(mask_frame)
        item_names.append(name)
    return item_names, item_video_frames, item_mask_frames

def preprocess_train_data(select='all'):
    '''
    This function preprocesses the training data. It returns the names of the items, the video frames and the mask frames. 
    The select parameter can be 'all', 'expert' or 'amateur'. 
    '''
    video_frames = []
    mask_frames = []
    names = []
    train_data = load_zipped_pickle("train.pkl")
    for item in tqdm(train_data):
        if select == 'all':
            item_names, item_video_frames, item_mask_frames = preprocess_item(item)
        elif select == 'expert':
            if item['dataset'] == 'expert':
                item_names, item_video_frames, item_mask_frames = preprocess_item(item)
            else: continue
        elif select == 'amateur':
            if item['dataset'] == 'amateur':
                item_names, item_video_frames, item_mask_frames = preprocess_item(item)
            else: continue
        else: raise ValueError('Invalid select')
        video_frames += item_video_frames
        mask_frames += item_mask_frames
        names += item_names

    return names, video_frames, mask_frames

def preprocess_test_data():
    video_frames = []
    names = []
    test_data = load_zipped_pickle("test.pkl")
    for item in tqdm(test_data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        video_frames += list(video)
        names += [item['name'] for _ in video]
    return names, video_frames

def resize_with_mask(video_frames, mask_frames, target_size=(512, 512)):
    """
    Resize a frame and its corresponding mask to the target size.
    Uses bicubic interpolation for frames and nearest-neighbor for masks.
    This is for resizing the amateur videos which are significantly smaller 
    than the expert videos ONLY.
    """
    resized_frames = []
    resized_masks = []
    for frame, mask in zip(video_frames, mask_frames):
        resized_frames.append(cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC))
        resized_masks.append(cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST))
    return resized_frames, resized_masks

def pad_or_crop_with_mask(video_frames, mask_frames, target_size=(700, 700)):
    """
    Resize a frame and its mask to fit the target size, either by padding or cropping.
    Handles frames smaller than the target size by padding and larger ones by cropping.
    This is for resizing the expert videos and also the processed amateur videos to the
    final size of 700x700.
    """
    target_h, target_w = target_size
    final_frames = []
    final_masks = []

    for frame, mask in zip(video_frames, mask_frames):
        h, w = frame.shape[:2]
        # Adjust height
        if h < target_h:  # Pad height
            pad_h = (target_h - h) // 2
            pad_h_remain = target_h - h - pad_h
            frame = np.pad(frame, ((pad_h, pad_h_remain), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((pad_h, pad_h_remain), (0, 0)), mode='constant', constant_values=0)
        elif h > target_h:  # Crop height
            crop_h = (h - target_h) // 2
            frame = frame[crop_h:crop_h + target_h, :]
            mask = mask[crop_h:crop_h + target_h, :]

        # Adjust width
        if w < target_w:  # Pad width
            pad_w = (target_w - w) // 2
            pad_w_remain = target_w - w - pad_w
            frame = np.pad(frame, ((0, 0), (pad_w, pad_w_remain)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, 0), (pad_w, pad_w_remain)), mode='constant', constant_values=0)
        elif w > target_w:  # Crop width
            crop_w = (w - target_w) // 2
            frame = frame[:, crop_w:crop_w + target_w]
            mask = mask[:, crop_w:crop_w + target_w]

        final_frames.append(frame)
        final_masks.append(mask)
    
    return final_frames, final_masks