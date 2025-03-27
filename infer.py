import os
import os.path as osp 
import random 
import argparse
from tqdm import tqdm

import csv
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset import TestingDataset
from model import PGP_SAM
from utils.utils import create_masks, read_gt_masks, get_logger, vis_pred, save_pred
from utils.cal_metrics import eval_metrics, eval_hd95


print('======> Process Arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='pgp_sam')

# Set-up Model
parser.add_argument('--dataset', type=str, default='bhx_sammed', help='specify dataset')
parser.add_argument('--root_dir', type=str, default='/home/yanzhonghao/data', help='specify root path')
parser.add_argument('--data_dir', type=str, default='datasets', help='specify dataset path')
parser.add_argument('--save_dir', type=str, default='experiments', help='specify save path')
parser.add_argument('--num_classes', type=int, default=8, help='specify the classes of the dataset without the bg')
parser.add_argument('--num_tokens', type=int, default=8, help='the num of prompts')
parser.add_argument('--sam_mode', type=str, default='vit_b', choices=['vit_b', 'vit_l'], help='specify backbone')
parser.add_argument('--train_time', type=str, default=None, help='specify the training time')
parser.add_argument('--model_type', type=str, default='lora', help='specify the parameters involved in training')
parser.add_argument('--stage', type=int, default=2, help='specify the stage of decoders')

# Testing Strategy
parser.add_argument('--scale', type=float, default=0.1, help='percentage of training data')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--image_size', type=int, default=512, help='image_size')
parser.add_argument('--resolution', type=int, default=512, help='image_size')
parser.add_argument('--volume', type=bool, default=False, help='whether to evaluate test set in volume')
parser.add_argument('--vis', type=bool, default=False, help='whether to visualise results')
parser.add_argument('--tsne', type=bool, default=False, help='whether to visualise features with tsne')
parser.add_argument('--seed', type=int, default=2024)

parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
torch.cuda.set_device(args.gpu_ids[0])

def test(args):
    print('======> Set Parameters for Inference' )  
    run_name = args.run_name
        
    seed = args.seed  
    batch_size = args.batch_size
    num_workers = args.num_workers

    volume = args.volume
    vis = args.vis

    # set seed for reproducibility 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print('======> Load Dataset-Specific Parameters' )
    scale = args.scale
    sam_mode = args.sam_mode
    model_type = args.model_type
    dataset_name = args.dataset
    image_size = args.image_size
    resolution = args.resolution
    
    data_root_dir = osp.join(args.root_dir, args.data_dir, args.dataset)
    test_dataset = TestingDataset(
                        data_root_dir=data_root_dir,
                        mode='test',
                        image_size=image_size
                        )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print('======> Set Saving Directories and Logs')
    now = args.train_time
    task = f'{sam_mode}_{model_type}_{now}'
    settings = f'few_shot_{int(scale*100)}'
    save_dir = osp.join(args.root_dir, args.save_dir, run_name, dataset_name, settings, task) 
    save_log_dir = osp.join(save_dir, 'log')
    save_ckpt_dir = osp.join(save_dir, 'ckpt')
    save_pred_dir = osp.join(save_dir, 'pred')
    
    if not volume:
        loggers = get_logger(os.path.join(save_log_dir, f'{task}_test.log'))
    else:
        loggers = get_logger(os.path.join(save_log_dir, f'{task}_test_volume.log'))
    loggers.info(f'Args: {args}')
    
    print('======> Load Prototype-based Model for different model mode')
    sam_checkpoint = osp.join(save_ckpt_dir, f'best_ckpt.pth') 
    stage = args.stage
    num_classes = args.num_classes
    num_tokens = args.num_tokens

    model = PGP_SAM(
                sam_checkpoint=sam_checkpoint,
                sam_mode=sam_mode,   
                model_type=model_type,
                stage=stage,
                mask_size=image_size,
                num_classes=num_classes,
                num_tokens=num_tokens,
                resolution=resolution,
            ) 
          
    model.to(args.device)
    model.load_state_dict(torch.load(sam_checkpoint, map_location='cuda'))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if args.multi_gpu else model

    print('======> Start Inference')
    val_masks = dict()
    model.eval()

    with torch.no_grad():
        tbar = tqdm((test_dataloader), total = len(test_dataloader), leave=False)
            
        for batch_input in tbar:   
            mask_names = batch_input['mask_names']
            images = batch_input['images'].cuda()
            images = F.interpolate(images, (resolution, resolution), mode='bilinear', align_corners=False)
            
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs[-1], dim=1), dim=1).squeeze(0) # [b, 512, 512]

            for pred, im_name in zip(preds, mask_names):
                val_masks[im_name] = np.array(pred.detach().cpu())
        
    gt_masks = read_gt_masks(data_root_dir=data_root_dir, mode='test', mask_size=image_size, volume=volume)
    val_masks = create_masks(data_root_dir, val_masks, image_size, volume=volume)
    iou_results, dice_results, iou_csv, dice_csv = eval_metrics(val_masks, gt_masks, num_classes)
    loggers.info(f'IoU_Results: {iou_results};')
    loggers.info(f'Dice_Results: {dice_results}.')
    with open(os.path.join(save_log_dir, 'results_volume.csv' if volume else 'results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(iou_csv)
        writer.writerow(dice_csv)
    
    if vis and not volume:
        save_pred(val_masks, save_pred_dir)
        vis_pred(val_masks, gt_masks, save_pred_dir, dataset_name)


if __name__ == '__main__':
    test(args)

