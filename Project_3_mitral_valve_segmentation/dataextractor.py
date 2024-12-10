import pickle
import gzip
import numpy as np
import os
from tqdm import tqdm

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