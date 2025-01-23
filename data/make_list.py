import os
import random
from tqdm import tqdm

import cv2
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cv2.setRNGSeed(SEED)

DATA_PATH = '../../data'
TASK = 'abdomen'                # [ven, abdomen]
DATASET = 'sabs_sammed'         # [bhx_sammed, sabs_sammed]
NUM_CLASS = 8                   # [8, 4]
SCALE = 0.01                    # [0.1, 0.05, 0.01]

data_path = os.path.join(DATA_PATH, TASK, DATASET, 'train', 'masks')
save_path = os.path.join(DATA_PATH, TASK, DATASET, 'lists')
os.makedirs(save_path, exist_ok=True)

masks_list = sorted(os.listdir(data_path))
masks_len = len(masks_list) // NUM_CLASS
target_len = int(len(masks_list) * SCALE) + 1
target_list = []
target_mask = {}

for im_name in tqdm(masks_list):
    mask = cv2.imread(os.path.join(data_path, im_name))
    unique_classes = np.unique(mask)
    
    for cls in unique_classes:
        if cls == 0:
            continue

        class_mask = np.array(mask == cls, dtype=np.uint8)
        area = np.sum(class_mask)
        if cls not in target_mask:
            target_mask[cls] = []

        target_mask[cls].append((im_name, area))

for cls, masks in tqdm(target_mask.items()):
    masks_sorted = sorted(masks, key=lambda x: x[1], reverse=True)[ : masks_len]
    for im_name, _ in masks_sorted:
        if im_name not in target_list:
            target_list.append(im_name)

if len(target_list) < target_len:
    remaining_masks = [m for m in masks_list if m not in target_list]
    target_list.extend(random.sample(remaining_masks, target_len - len(target_list)))

elif len(target_list) > target_len:
    target_list = random.sample(target_list, target_len)

with open(os.path.join(save_path, f'{DATASET}_{str(int(SCALE * 100))}.txt'), 'w') as f:
    for item in target_list:
        f.write(f"{item}\n")
    
    



