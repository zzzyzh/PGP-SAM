import os
import glob
import shutil
import json
from tqdm import tqdm

import cv2
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

 
def position_priori(target_path, source_path):
    for task in ['train', 'val', 'test']:    
        source_task_path = os.path.join(source_path, task)
        target_task_path = os.path.join(target_path, task)
        masks_stat = analysis(source_task_path, "masks", ismask=True, find_bbox=True)
        
        patient_ID = sorted(set(masks_stat["volume_name"]))  # 0701 - 0800
        patient_bbox = dict()
        for ID in patient_ID:
            patient_stat = masks_stat[masks_stat["volume_name"]==ID]
            x0, y0, x1, y1 = patient_stat["x0"].min(), patient_stat["y0"].min(), patient_stat["x1"].max(), patient_stat["y1"].max()
            assert (256 >= x1 > x0 >= 0) and (256 >= y1 > y0 >= 0), (x0, y0, x1, y1)
            patient_bbox[str(ID)] = [x0, y0, x1, y1]
        print(f"\033[31mpropotion per class: \033[0m{np.around((masks_stat['pixel_count'].sum() / masks_stat['pixel_count'].sum().sum()) * 100, 4)} (%)")

        crop(source_task_path, target_task_path, "images", patient_bbox, ismask=False)
        crop(source_task_path, target_task_path, "masks", patient_bbox, ismask=True)
  
        masks_crop_stat = analysis(target_task_path, "masks", ismask=True, find_bbox=False)
        print(f"\033[31mpropotion per class: \033[0m{np.around((masks_crop_stat['pixel_count'].sum() / masks_crop_stat['pixel_count'].sum().sum()) * 100, 4)} (%)")
        
        target_labels_path = os.path.join(target_task_path, 'labels')
        os.makedirs(target_labels_path, exist_ok=True)

        ven = ['right', 'left', 'third', 'fourth']
        for im_name in sorted(os.listdir(os.path.join(target_task_path, 'masks'))):
            im_name = im_name.split('.')[0]
            
            source_mask = cv2.imread(os.path.join(target_task_path, 'masks', f'{im_name}.png'))
            source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)

            unique_pixel = np.unique(source_mask)[1:]
            if 5 in unique_pixel:
                unique_pixel = unique_pixel[:-1]
                
            for uni in unique_pixel:
                mask = np.zeros_like(source_mask)
                mask[source_mask == uni] = 255
                target_name = f'{im_name}_{ven[uni-1]}.png'
                cv2.imwrite(os.path.join(target_labels_path, target_name), mask)
                
    
def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask) 
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1, :4]

    
def analysis(source_path, data_type, ismask=False, find_bbox=False):
    data_path = os.path.join(source_path, data_type)

    statistics = list()
    for path in tqdm(sorted(glob.glob(os.path.join(data_path, "*.png"))), desc=f"Analysis {data_type}"):
        volume_name = path.split("_")[3]
        slice_image = np.asarray(Image.open(path).convert("L"))
        if ismask:
            classes = set(slice_image.flatten())
            uni_mask = (slice_image > 0).astype(np.uint8)*255
            stats = mask_find_bboxs(uni_mask)

            if find_bbox:
                # find bbox
                padding = np.random.randint(40, 50)     # random padding
                x, y, w, h = np.hsplit(stats, 4)
                x0, y0 = max(x.min() - padding, 0), max(y.min() - padding, 0)
                x1, y1 = min((x+w).max() + padding, 256), min((y+h).max() + padding, 256)

                # visualize bbox
                # bbox = cv2.rectangle(cv2.cvtColor(uni_mask, cv2.COLOR_GRAY2BGR), (x0, y0), (x1, y1), (255,0,0), 1)
                # plt.imshow(bbox+cv2.cvtColor(uni_mask, cv2.COLOR_GRAY2RGB), vmin=0, vmax=255, cmap="gray")
                # plt.show()

            # count pixels
            pixel_count = np.asarray([(slice_image==cls_id).sum() for cls_id in range(5)])

            if find_bbox:
                statistics.append(pd.Series([volume_name, slice_image.min(), slice_image.max(), slice_image.shape, 
                                            classes, x0, y0, x1, y1, pixel_count]))
            else:
                statistics.append(pd.Series([volume_name, slice_image.min(), slice_image.max(), slice_image.shape, 
                                            classes, pixel_count]))
        else:
            statistics.append(pd.Series([volume_name, slice_image.min(), slice_image.max(), slice_image.shape]))
    statistics = pd.concat(statistics, axis=1).T
    if ismask and find_bbox:
        statistics.columns = ["volume_name", "min", "max", "shape", "classes", "x0", "y0", "x1", "y1", "pixel_count"]
    elif ismask and not find_bbox:
        statistics.columns = ["volume_name", "min", "max", "shape", "classes", "pixel_count"]
    else:
        statistics.columns = ["volume_name", "min", "max", "shape"]
    return statistics


def crop(source_path, target_path, data_type, patient_bboxs, ismask=False):
    data_path = os.path.join(source_path, data_type)
    save_path = os.path.join(target_path, data_type)
    os.makedirs(save_path, exist_ok=True)

    for path in tqdm(sorted(glob.glob(os.path.join(data_path, "*.png"))), desc=f"Crop {data_type}"):
        volume_name = path.split("_")[3]
        file_name = path.split("/")[-1]
        slice_image = np.asarray(Image.open(path).convert("L"))

        x0, y0, x1, y1 = patient_bboxs[volume_name]
        slice_image_crop = slice_image[y0:y1, x0:x1]
        if ismask:
            slice_image_croped = Image.fromarray(slice_image_crop).resize((256, 256), Image.Resampling.NEAREST)
        else:
            slice_image_croped = Image.fromarray(slice_image_crop).resize((256, 256), Image.Resampling.BILINEAR).convert('RGB')

        slice_image_croped.save(os.path.join(save_path, file_name))
     
        
# def convert2nii(source_path, target_path, data_type, stat):
#     data_path = os.path.join(ROOT_PATH, data_type)
#     save_path = os.path.join(ROOT_PATH, data_type+"-niigz")
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)

#     patient_IDs = sorted(set(stat["volume_name"]))
#     patient_lens = [(stat["volume_name"]==cls_id).sum() for cls_id in patient_IDs]

#     for idx, ID in enumerate(patient_IDs):
#         volume = np.zeros((256, 256, patient_lens[idx]))

#         pbar = tqdm(sorted(glob.glob(os.path.join(data_path, f"*_{ID}_*.png"))), desc=f"Convert {data_type}")
#         for path in pbar:
#             slice_image = np.asarray(Image.open(path).convert("L"))
#             volume[:, :, pbar.n] = slice_image

#             pbar.update()
        
#         ni_img = nib.Nifti1Image(volume, affine=np.eye(4))
#         nib.save(ni_img, os.path.join(save_path, f"{ID}.nii.gz"))
      
      
def pre_process(target_path, source_path, image_size=512):
    source_images_path = os.path.join(source_path, 'images')
    source_masks_path = os.path.join(source_path, 'labels')

    ven = ['right', 'left', 'third', 'fourth']    
    
    for task in ['train', 'val', 'test', 'support']:    
        target_images_path = os.path.join(target_path, task, 'images')
        target_masks_path = os.path.join(target_path, task, 'masks')
        target_labels_path = os.path.join(target_path, task, 'labels')
        os.makedirs(target_images_path, exist_ok=True)
        os.makedirs(target_masks_path, exist_ok=True)
        os.makedirs(target_labels_path, exist_ok=True)

        mask2image = {}
        label2image = {}
        image2label = {}
        crop_list = {}

        for index in tqdm(sorted(os.listdir(source_masks_path))):
            
            train = False
            if task == 'train' and int(index) <= 600:
                train = True
            if task == 'val' and 600 < int(index) <= 700:
                train = True
            if task == 'test' and 700 < int(index) <= 800:
                train = True
            if task == 'support' and 800 < int(index) <= 850:
                train = True

            if not train:
                continue
            
            m_x, m_y, m_w, m_h = 512, 512, 0, 0
            
            for im_name in sorted(os.listdir(os.path.join(source_masks_path, index))):
                im_name = im_name.split('.')[0]

                source_mask = cv2.imread(os.path.join(source_masks_path, index, f'{im_name}.png'), 0)
                if np.max(source_mask) == 0:
                    continue  # 不包含脑室
                
                source_dcm = pydicom.dcmread(os.path.join(source_images_path, index, f'{im_name}.dcm'))
                # 将像素值转换为 Hounsfield units (HU)
                data = source_dcm.pixel_array
                slope = source_dcm.RescaleSlope
                intercept = source_dcm.RescaleIntercept
                hu_data = slope * data + intercept
                
                # 设置窗宽和窗位
                window_width = 80  # 窗宽
                window_level = 50  # 窗位
                
                CT_min = window_level - window_width / 2
                CT_max = window_level + window_width / 2
                data = np.clip(hu_data, CT_min, CT_max) # [512, 512]
                
                unique_pixel = np.unique(source_mask)[1:]
                if 5 in unique_pixel:
                    unique_pixel = unique_pixel[:-1]

                data_min = data.min()
                data_max = data.max()
                source_image = np.uint8(((data - data_min) / (data_max - data_min + 1e-5)) * 255)
                _, binary_image = cv2.threshold(source_image, 1, 255, cv2.THRESH_BINARY)
                x, y, w, h = cv2.boundingRect(binary_image)

                m_x = x if x < m_x else m_x
                m_y = y if y < m_y else m_y
                m_w = w if w > m_w else m_w
                m_h = h if h > m_h else m_h
                
                save = False
                for uni in unique_pixel:
                    mask = np.zeros_like(source_mask)
                    mask[source_mask == uni] = 255
                
                    if np.sum(mask == 255) < 100:
                        continue
                    
                    save = True
                    target_name = f'bhx_{index}_{im_name}_{ven[uni-1]}.png'
                    label2image[os.path.join(target_labels_path, target_name)] = os.path.join(target_images_path,
                                                                                        f'bhx_{index}_{im_name}.png')
                    cv2.imwrite(os.path.join(target_labels_path, target_name), mask)
            
                    # # 寻找轮廓
                    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # # 循环遍历每个轮廓
                    # for i, contour in enumerate(contours):
                    #     target = np.zeros_like(source_mask)
                    #     cv2.drawContours(target, [contour], -1, 255, thickness=cv2.FILLED)
                    #     if np.sum(target == 255) < 100:
                    #         continue
                    #     save = True
                    #     target_name = f'bhx_{index}_{im_name}_{ven[uni-1]}_{i}.png'
                    #     label2image[os.path.join(target_labels_path, target_name)] = os.path.join(target_images_path,
                    #                                                                         f'bhx_{index}_{im_name}.png')
                    #     cv2.imwrite(os.path.join(target_labels_path, target_name), target)
                
                if save:        
                    # 保存为 png
                    cv2.imwrite(os.path.join(target_images_path, f'bhx_{index}_{im_name}.png'), source_image) 

                    # 保存 mask
                    raw_mask = cv2.imread(os.path.join(source_masks_path, index, f'{im_name}.png'), 0)
                    raw_mask[raw_mask == 5] = 0
                    cv2.imwrite(os.path.join(target_masks_path, f'bhx_{index}_{im_name}.png'), raw_mask)
                    mask2image[os.path.join(target_masks_path, target_name)] = os.path.join(target_images_path, target_name)
            
            r = np.random.randint(5,10)
            crop_list[index] = [max(m_x-r, 0), max(m_y-r, 0), min(m_w+r, image_size), min(m_h+r, image_size)]

            match_path = glob.glob(os.path.join(target_images_path, f'bhx_{index}_*'))
            crop(match_path, crop_list[index], mode='image')
            match_path = glob.glob(os.path.join(target_masks_path, f'bhx_{index}_*'))
            crop(match_path, crop_list[index], mode='mask')
            match_path = glob.glob(os.path.join(target_labels_path, f'bhx_{index}_*'))
            crop(match_path, crop_list[index], mode='mask')
            
        for label, image in label2image.items():
            if image not in image2label.keys():
                image2label[image] = [label]
            else:
                image2label[image].append(label)
                image2label[image] = sorted(image2label[image])

        with open(os.path.join(target_path, task, 'label2image_test.json'), 'w', newline='\n') as f:
            json.dump(label2image, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, task, 'image2label_train.json'), 'w', newline='\n') as f:
            json.dump(image2label, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, task, 'mask2image.json'), 'w', newline='\n') as f:
            json.dump(mask2image, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, task, 'crop_list.json'), 'w', newline='\n') as f:
            json.dump(crop_list, f, indent=2)  # 换行显示


def crop(match_path, c, mode, image_size=512):
    for target_path in match_path:
        if mode == 'image' and target_path.split('.')[-1] == 'png':
            image = cv2.imread(target_path, 0)
            image = image[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(target_path, np.uint8(image))
        elif mode == 'image' and target_path.split('.')[-1] == 'npz':
            image = np.load(target_path)['image']
            image = image[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            np.savez(target_path, image=image)
        elif mode == 'mask':
            image = cv2.imread(target_path, 0)
            image = image[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_path, np.uint8(image))


if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)    # set random seed for reproductivity
    np.set_printoptions(suppress=True)
    
    source_path = '../../data/ven/bhx'
    target_path = '../../data/ven/bhx_sammed'

    pre_process(target_path, source_path)

    # source_path = '../../data/ven/bhx_sammed_test'
    # target_path = '../../data/ven/bhx_sammed_test'

    # position_priori(target_path, source_path)    

