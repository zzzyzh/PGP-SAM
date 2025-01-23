import os 
import os.path as osp 
import logging
from tqdm import tqdm

import cv2 
import numpy as np 
import matplotlib.pyplot as plt


# bhx
BHX_TEST_VOLUME = [str(i).zfill(4) for i in range(701, 801)]
# sabs
SABS_TEST_VOLUME = ["0018", "0002", "0000", "0003", "0014", "0024"]


def read_gt_masks(data_root_dir="../../data/ven/bhx_sammed", mode="val", mask_size=512, volume=False):   
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_eval_masks = dict()
    
    gt_eval_masks_path = osp.join(data_root_dir, mode, "masks")
    if not volume:
        for mask_name in sorted(os.listdir(gt_eval_masks_path)):
            mask = cv2.imread(osp.join(gt_eval_masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            gt_eval_masks[mask_name] = mask
    else:
        if 'bhx' in data_root_dir:
            test_volume = BHX_TEST_VOLUME
            test_len = 40
        elif 'sabs' in data_root_dir:
            test_volume = SABS_TEST_VOLUME
            test_len = 200
        
        for id in tqdm(test_volume):
            mask_as_png = np.zeros([test_len, mask_size, mask_size], dtype='uint8')
            for mask_name in sorted(os.listdir(gt_eval_masks_path)):
                if f'{id}_' in mask_name:
                    mask = cv2.imread(osp.join(gt_eval_masks_path, mask_name), 0)
                    mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
                    i = int(mask_name.split('.')[0].split('_')[-1])
                    mask_as_png[i, :, :] = mask

            gt_eval_masks[id] = mask_as_png
                
    return gt_eval_masks
  

def create_masks(data_root_dir, val_masks, mask_size, volume=False):
    if not volume:
        eval_masks = val_masks
    else:
        eval_masks = dict()
        if 'bhx' in data_root_dir:
            test_volume = BHX_TEST_VOLUME
            test_len = 40
        elif 'sabs' in data_root_dir:
            test_volume = SABS_TEST_VOLUME
            test_len = 200

        for id in tqdm(test_volume):
            mask_as_png = np.zeros([test_len, mask_size, mask_size], dtype='uint8')
            for mask_name, mask in val_masks.items():
                if id in mask_name:
                    i = int(mask_name.split('.')[0].split('_')[-1])
                    mask_as_png[i, :, :] = mask.astype(np.uint8)
            eval_masks[id] = mask_as_png    
             
    return eval_masks

        
def get_logger(filename, resume=False):
    write_mode = "a" if resume else "w"    
    verbosity = 1
    name = None
    
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, write_mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_pred(pred_dict, save_dir):
    save_dir = os.path.join(save_dir, 'masks')
    os.makedirs(save_dir, exist_ok=True)

    for key, value in pred_dict.items():
        cv2.imwrite(os.path.join(save_dir, key), value.astype(np.uint8))


CLASS2COLOR = {
    0: (0, 0, 0),
    1: (11, 158, 150),   
    2: (0, 0, 255),     
    3: (255, 0, 255),     
    4: (241, 156, 118),    
    5: (27, 255, 255),    
    6: (227, 0, 127),    
    7: (255, 255, 0),    
    8: (0, 255, 0)     
}


def vis_pred(pred_dict, gt_dict, save_dir, dataset_name):
    save_dir = os.path.join(save_dir, 'vis')
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == 'bhx_sammed':
        color_mapping = {i: CLASS2COLOR[i] for i in range(1, 5)}
    elif dataset_name == 'sabs_sammed' or 'sabs_sammed_roi':
        color_mapping = {i: CLASS2COLOR[i] for i in range(1, 9)}

    for _, mask_name in enumerate(tqdm(list(pred_dict.keys()))):
        pred = np.array(pred_dict[mask_name])  # [256, 256]
        gt = np.array(gt_dict[mask_name])
        
        pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        
        for cls, color in color_mapping.items():
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
            
            pred_vis[pred_mask] = color
            gt_vis[gt_mask] = color

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_vis)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_vis)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, mask_name))
        plt.close()


if __name__ == '__main__':
    read_gt_masks()

