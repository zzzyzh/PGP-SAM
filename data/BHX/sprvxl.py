import os
from tqdm import tqdm

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
import skimage.segmentation
from skimage.measure import label


def superpixel(img, method='fezlen', **kwargs):
    """
    loop through the entire volume
    assuming image with axis z, x, y
    """
    if method == 'fezlen':
        seg_func = skimage.segmentation.felzenszwalb
    else:
        raise NotImplementedError

    seg = seg_func(img, min_size=300, sigma=1)
    return seg


# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh):
    mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = binary_fill_holes(post_mask)
    return fill_mask


# remove superpixels within the empty regions
def superpixel_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = np.max(lbvs)
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append(max_lb)
    raw_seg2d = raw_seg2d * mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0:
            continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            lb_new += 1

    return out_seg2d


def pre_sprvxl(target_path):
    fg_thresh = 1e-4
    
    for task in ['train']:    
        images_path = os.path.join(target_path, task, 'images')
        pseudo_path = os.path.join(target_path, task, 'pseudo')
        os.makedirs(pseudo_path, exist_ok=True)
        
        for _, im_name in enumerate(tqdm(sorted(os.listdir(images_path)))):
            image = cv2.imread(os.path.join(images_path, im_name), 0)
            raw_seg = superpixel(image)
            _fgm = fg_mask2d(image, fg_thresh)
            _out_seg = superpixel_masking(raw_seg, _fgm)
            cv2.imwrite(os.path.join(pseudo_path, im_name), np.uint8(_out_seg))
        

if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)    # set random seed for reproductivity
    np.set_printoptions(suppress=True)
    
    target_path = '/home/yanzhonghao/data/ven/bhx_sammed'
    pre_sprvxl(target_path)    