import os 
import os.path as osp
import re
import random 
import math
import cv2 
from tqdm import tqdm
from einops import repeat

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, map_coordinates


class TestingDataset(Dataset):
    def __init__(self, 
                 data_root_dir = "../../data/ven/bhx_sammed", 
                 mode = "test", 
                 image_size = 512,
                 ):
        
        self.image_size = image_size       

        # directory
        self.dataset = data_root_dir.split('/')[-1]
        self.image_dir = osp.join(data_root_dir, mode, "images")
        self.mask_dir = osp.join(data_root_dir, mode, "masks")
        self.mask_list = sorted(os.listdir(self.mask_dir))
                
        # normalization
        # self.pixel_mean=[123.675, 116.28, 103.53]
        # self.pixel_std=[58.395, 57.12, 57.375]
        self.pixel_mean=[0, 0, 0]
        self.pixel_std=[1, 1, 1]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        mask_path = osp.join(self.mask_dir, mask_name)
        image_path = osp.join(self.image_dir, mask_name)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = repeat(image[np.newaxis, :, :], 'c h w -> (repeat c) h w', repeat=3)
        
        mask = torch.tensor(mask).unsqueeze(0).to(torch.float32)
        image = torch.tensor(image).to(torch.float32)
        image = image / 255.0
        # image = (image - torch.tensor(self.pixel_mean).view(-1, 1, 1)) / torch.tensor(self.pixel_std).view(-1, 1, 1)

        batch_input = {
            'images': image,
            'masks': mask,
            'mask_names': mask_name
        }

        return batch_input


class TrainingDataset(Dataset):
    def __init__(self, 
                data_root_dir = "../../data/ven/bhx_sammed",
                image_size = 512,
                scale = 0.1,
                ):
        
        self.image_size = image_size       

        # directory
        self.dataset = data_root_dir.split('/')[-1]
        self.image_dir = osp.join(data_root_dir, 'train', "images")
        self.mask_dir = osp.join(data_root_dir, 'train', "masks")
        
        self.support = self.get_support(data_root_dir, scale)
        
        # normalization
        # self.pixel_mean=[123.675, 116.28, 103.53]
        # self.pixel_std=[58.395, 57.12, 57.375]
        self.pixel_mean=[0, 0, 0]
        self.pixel_std=[1, 1, 1]
        
    def __len__(self):
        return len(self.support)

    def __getitem__(self, index):
        im_name = self.support[index]
        
        mask_path = osp.join(self.mask_dir, im_name)
        image_path = osp.join(self.image_dir, im_name)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image, mask = self.augmentation(image, mask, index)
        
        mask = torch.tensor(mask).unsqueeze(0).to(torch.float32)
        image = torch.tensor(image).to(torch.float32)
        image = image / 255.0
        # image = (image - torch.tensor(self.pixel_mean).view(-1, 1, 1)) / torch.tensor(self.pixel_std).view(-1, 1, 1)
        
        batch_input = {
            'im_names': im_name,
            'images': image,
            'masks': mask,
        }
        
        return batch_input

    def get_support(self, data_root_dir, scale):
        
        dataset = data_root_dir.split('/')[-1]
        with open(osp.join(data_root_dir, 'lists', f'{dataset}_{str(int(scale * 100))}.txt'), 'r') as f:
            slice_idx = f.readlines()

        slice_idx = [line.strip() for line in slice_idx]
        return slice_idx
        
    def augmentation(
            self, 
            image, 
            mask,
            index,
            ):
        """Generate augmentation to image and masks
        image - original image
        mask - binary mask for the classes present in the image

        Returns:
            image - image after the augmentation
            mask - mask after the augmentation 
        """
        
        random_prob = random.random()
        if random_prob > 0.7:
            image, mask = random_rotate(image, mask)
        elif random_prob > 0.4:
            image, mask = random_elastic(image, mask, 
                                         image.shape[1] * 2,
                                         image.shape[1] * 0.08,
                                         image.shape[1] * 0.08)
        elif random_prob > 0.2:
            image, mask = random_scale(image, mask)

        random_prob = random.random()
        if random_prob > 0.7:
            image = add_gaussian_noise(image)
        elif random_prob > 0.5:
            image = add_poisson_noise(image)
        elif random_prob > 0.4:
            image = add_pepper_noise(image)

        x, y = image.shape
        if x != self.image_size or y != self.image_size:
            image = zoom(image, (self.image_size / x, self.image_size / y), order=3)
            mask = zoom(mask, (self.image_size / x, self.image_size / y), order=0)

        image = repeat(image[np.newaxis, :, :], 'c h w -> (repeat c) h w', repeat=3)
    
        return image, mask


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_elastic(image, label, alpha, sigma, alpha_affine, mode='reflect'):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]

    # 1. 仿射变换
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])

    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    image_affine = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    label_affine = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # 2. 生成随机位移场
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 3. 使用弹性变形应用于图像和标签
    image_deformed = map_coordinates(image_affine, indices, order=1, mode=mode).reshape(shape)
    label_deformed = map_coordinates(label_affine, indices, order=1, mode=mode).reshape(shape)

    return image_deformed, label_deformed


def random_scale(image, label, min_ratio=0.4, max_ratio=0.9):
    h, w = image.shape

    scale = random.uniform(min_ratio, max_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)

    new_h = min(new_h, h)
    new_w = min(new_w, w)

    y = np.random.randint(0, h - new_h + 1)
    x = np.random.randint(0, w - new_w + 1)

    image = image[y:y+new_h, x:x+new_w]
    label = label[y:y+new_h, x:x+new_w]

    return image, label


def random_cutmix(image, label, cut_image, cut_label, beta=0.5):
    def random_roi(shape, lam):
        W = shape[0]
        H = shape[1]
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
        bby1 = int(np.clip(cy - cut_h // 2, 0, H))
        bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
        bby2 = int(np.clip(cy + cut_h // 2, 0, H))

        return [bbx1, bby1, bbx2, bby2]

    lam = np.random.beta(beta, beta)
    roi = random_roi(image.shape, lam)
    bbx1, bby1, bbx2, bby2 = roi
    
    cut_image_part = cut_image[bbx1:bbx2, bby1:bby2]
    cut_label_part = cut_label[bbx1:bbx2, bby1:bby2]

    image[bbx1:bbx2, bby1:bby2] = cut_image_part
    label[bbx1:bbx2, bby1:bby2] = cut_label_part

    return image, label


def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)


def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy_image, 0, 255)


def add_pepper_noise(image, snr=0.7):
    noisy_image = np.copy(image)
    h, w = image.shape

    signal_pct = snr
    noise_pct = 1 - snr

    noise_mask = np.random.choice([0, 1, 2], size=(h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
    noisy_image[noise_mask == 1] = 255  # 盐噪声
    noisy_image[noise_mask == 2] = 0    # 椒噪声
        
    return np.clip(noisy_image, 0, 255)


if __name__ == '__main__':
    random.seed(2024)
    pixel_mean=[123.675, 116.28, 103.53]
    pixel_std=[58.395, 57.12, 57.375]
    
    data_root_dir = '../../data/abdomen/sabs_sammed'
    
    train_dataset = TrainingDataset(
                                data_root_dir = data_root_dir,
                                scale = 0.01,
                                )
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=16)
    
    for epoch in range(1):
        tbar = tqdm((train_dataloader), total = len(train_dataloader), leave=False)
        for batch_input in tbar:
            im_names = batch_input['im_names']

