import os
import os.path as osp 
import random 
import argparse
from datetime import datetime

from tqdm import tqdm

import numpy as np 
import torch 
from torch.backends import cudnn
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset import TrainingDataset, TestingDataset
from model import PGP_SAM
from utils.utils import read_gt_masks, get_logger
from utils.loss import DiceLoss, FocalLoss, LogLR, WarmupCosineLR, cal_seg_loss
from utils.cal_metrics import eval_metrics


# ======> Process Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='pgp_sam')

# Set-up Model    
parser.add_argument('--task', type=str, default='ven', help='specify task')
parser.add_argument('--dataset', type=str, default='bhx_sammed', help='specify dataset')
parser.add_argument('--data_root_dir', type=str, default='dataset', help='specify dataset root path')
parser.add_argument('--save_dir', type=str, default='experiments', help='specify save path')
parser.add_argument('--num_classes', type=int, default=4, help='specify the classes of the dataset without the bg')
parser.add_argument('--num_tokens', type=int, default=8, help='the num of prompts') 
parser.add_argument('--sam_mode', type=str, default='vit_b', choices=['vit_b', 'vit_l'], help='specify backbone')
parser.add_argument('--sam_ckpt', type=str, default='models/sam_vit_b_01ec64.pth', help='specify raw SAM ckpt path')
parser.add_argument('--model_type', type=str, default='lora', help='specify the parameters involved in training')
parser.add_argument('--stage', type=int, default=2, help='specify the stage of decoders')

# Training Strategy
parser.add_argument('--scale', type=float, default=0.1, help='percentage of training data')
parser.add_argument('--num_epochs', type=int, default=300, help='the num of epochs')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--resolution', type=int, default=512, choices=[256, 512], help='input size of the model')
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='optimizer')
parser.add_argument('--scheduler', type=str, default='WarmupCosineLR', choices=['CosWarm', 'LogLR', 'WarmupCosineLR'], help='scheduler')
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'], help='loss function')
parser.add_argument('--dice_weight', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='learning rate')
parser.add_argument('--seed', type=int, default=42)

# Multi-GPU Settings
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])


class Trainer:
    def __init__(self, args, model, train_dataloader, val_dataloader, loggers, writer):
        
        self.model = model

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.loggers = loggers
        self.writer = writer

        self.args = args
        self.save_ckpt_dir = self.args.save_ckpt_dir
        self.resolution = self.args.resolution
        self.image_size = self.args.image_size
        self.num_epochs = self.args.num_epochs
        self.num_classes = self.args.num_classes
        self.num_tokens = self.args.num_tokens
        
        self.set_gt_masks()
        self.set_loss_fn()
        self.set_optimizer()
        
        self.cal_model_params()
        
    def set_loss_fn(self):
        alpha = self.cal_class_freq(mode='train') + self.cal_class_freq(mode='val') + 0.1
        
        self.dice_weight = self.args.dice_weight
        self.dice_loss_model = DiceLoss().cuda()
        if self.args.loss == 'ce':
            self.ce_loss_model = CrossEntropyLoss().cuda()
        else:
            self.ce_loss_model = FocalLoss(alpha=alpha, num_classes=self.num_classes).cuda()
        
    def set_optimizer(self):
        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model
            
        self.args.max_iterations = self.num_epochs * len(self.train_dataloader)
        self.args.warmup_iters = min(int(self.args.max_iterations * 0.1), 250)
        
        if self.args.optimizer == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)  
        elif self.args.optimizer == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        
        if self.args.scheduler == 'CosWarm':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        elif self.args.scheduler == 'LogLR':
            self.scheduler = LogLR(self.optimizer, warmup_iters=self.args.warmup_iters, total_iters=self.args.max_iterations, lr=self.args.lr)
        elif self.args.scheduler == 'WarmupCosineLR':
            self.scheduler = WarmupCosineLR(self.optimizer, warmup_iters=self.args.warmup_iters, total_iters=self.args.max_iterations, base_lr=self.args.lr, base_lr_end=1e-6)
    
    def set_gt_masks(self):
        data_root_dir = osp.join(self.args.data_root_dir, self.args.task, self.args.dataset)
        self.gt_masks = read_gt_masks(data_root_dir=data_root_dir, mask_size=self.image_size, mode='val')

    def cal_class_freq(self, mode='train', smoothing=1e-6):
        data_root_dir = osp.join(self.args.data_root_dir, self.args.task, self.args.dataset)
        gt_masks = read_gt_masks(data_root_dir=data_root_dir, mask_size=self.image_size, mode=mode)

        flattened_data = torch.tensor(np.array(list(gt_masks.values()))).view(-1)
        class_counts = torch.bincount(flattened_data, minlength=self.num_classes)
        total_count = flattened_data.numel()
        class_frequencies = class_counts.float() / total_count
        class_frequencies = class_frequencies + smoothing
        
        inverse_frequencies = 1.0 / class_frequencies
        inverse_frequencies = torch.sqrt(inverse_frequencies + smoothing)
        inverse_frequencies = inverse_frequencies / inverse_frequencies.sum()

        return inverse_frequencies

    def cal_multi_stage_loss(self, loss, epoch=0):
        weight = 0.6**(0.990**epoch)
        
        seg_loss = (1-weight) * loss[0]['loss'] + weight * loss[1]['loss']
        seg_ce_loss = (1-weight) * loss[0]['ce_loss'] + weight * loss[1]['ce_loss']
        seg_dice_loss = (1-weight) * loss[0]['dice_loss'] + weight * loss[1]['dice_loss']
        
        loss = {'loss': seg_loss,
                'ce_loss': seg_ce_loss,
                'dice_loss': seg_dice_loss
                }

        return loss

    def cal_model_params(self):
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.loggers.info(f'Args: {self.args}')

            model_grad_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            model_total_params = sum(p.numel() for p in self.model.parameters())
            self.loggers.info('model_grad_params:' + str(model_grad_params))
            self.loggers.info('model_total_params:' + str(model_total_params))
            
            model_grad_params_mb = model_grad_params * 4 / (1024 ** 2)
            model_total_params_mb = model_total_params * 4 / (1024 ** 2)
            self.loggers.info(f'model_grad_params: {model_grad_params_mb:.2f} MB')
            self.loggers.info(f'model_total_params: {model_total_params_mb:.2f} MB')
            
    def main(self):

        best_dice_val = -100.0
        for epoch in range(self.num_epochs):
            
            if self.args.multi_gpu:
                self.train_dataloader.sampler.set_epoch(epoch)
                self.val_dataloader.sampler.set_epoch(epoch)
            
            self.train(epoch)

            if self.args.multi_gpu:
                dist.barrier()

            dice = self.val(epoch)
            
            if self.args.multi_gpu:
                dist.barrier()
                
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if dice > best_dice_val:
                    best_dice_val = dice
                    
                    state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()
                    torch.save(state_dict, osp.join(self.save_ckpt_dir, f'best_ckpt.pth'))
                    opt = {
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "best_dice_val": best_dice_val
                    }
                    torch.save(opt, osp.join(self.save_ckpt_dir, f'best_opt.pth'))
                    
                    self.loggers.info(f'Best Dice: {best_dice_val:.4f} at Epoch {epoch+1}')        
           
    def train(self, epoch):
        train_loss = 0
        train_ce_loss = 0
        train_dice_loss = 0
        self.model.train()
        
        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.train_dataloader, total=len(self.train_dataloader), leave=False)
        else:
            tbar = self.train_dataloader
        
        for batch_input in tbar:
            masks = batch_input['masks'].cuda()
            images = batch_input['images'].cuda()
            images = F.interpolate(images, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
            
            outputs = model(images)
            seg_loss = cal_seg_loss(outputs, masks, self.dice_loss_model, self.ce_loss_model, self.dice_weight)
            
            loss = self.cal_multi_stage_loss(seg_loss, epoch)
            _loss = loss['loss']
            
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            train_loss += loss['loss'].detach().cpu()
            train_ce_loss += loss['ce_loss'].detach().cpu()
            train_dice_loss += loss['dice_loss'].detach().cpu()
            
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                tbar.set_description(f'Train Epoch [{epoch+1}/{self.num_epochs}]')
                tbar.set_postfix(loss=_loss.item())
            
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.loggers.info(f'Train - Epoch: {epoch+1}/{self.num_epochs}; Average Train Loss: {train_loss/len(self.train_dataloader)}')
            self.writer.add_scalar('train/loss', train_loss/len(self.train_dataloader), epoch)
            self.writer.add_scalar('train/ce_loss', train_ce_loss/len(self.train_dataloader), epoch)
            self.writer.add_scalar('train/dice_loss', train_dice_loss/len(self.train_dataloader), epoch)

    def val(self, epoch):
        val_masks = dict()  
        val_loss = 0
        val_ce_loss = 0
        val_dice_loss = 0
        self.model.eval()
        
        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model
            self.args.rank = -1
            
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            vbar = tqdm(self.val_dataloader, total=len(self.val_dataloader), leave=False)
        else:
            vbar = self.val_dataloader
        
        with torch.no_grad():
            for batch_input in vbar:
                mask_names = batch_input['mask_names']
                masks = batch_input['masks'].cuda()
                images = batch_input['images'].cuda()
                images = F.interpolate(images, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
                
                outputs = model(images)
                preds = torch.argmax(torch.softmax(outputs[-1], dim=1), dim=1).squeeze(0)
                
                for pred, im_name in zip(preds, mask_names):
                    val_masks[im_name] = np.array(pred.detach().cpu())
                
                loss = cal_seg_loss(outputs, masks, self.dice_loss_model, self.ce_loss_model, self.dice_weight)
                loss = self.cal_multi_stage_loss(loss, epoch=epoch)
                val_loss += loss['loss'].detach().cpu()
                val_ce_loss += loss['ce_loss'].detach().cpu()
                val_dice_loss += loss['dice_loss'].detach().cpu()
                
                if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                    vbar.set_description(f'Val Epoch [{epoch+1}/{self.num_epochs}]')
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.loggers.info(f'Val - Epoch: {epoch+1}/{self.num_epochs}; Average Val Loss: {val_loss/len(self.val_dataloader)}')
            self.writer.add_scalar('val/loss', val_loss/len(self.val_dataloader), epoch)
            self.writer.add_scalar('val/ce_loss', val_ce_loss/len(self.val_dataloader), epoch)
            self.writer.add_scalar('val/dice_loss', val_dice_loss/len(self.val_dataloader), epoch)
            
        if self.args.multi_gpu:
            if self.args.rank == 0:
                gathered_val_masks = [None] * dist.get_world_size()
                dist.gather_object(val_masks, gathered_val_masks, dst=0)
                
                # Rank 0 merges gathered masks
                merged_val_masks = {}
                for masks in gathered_val_masks:
                    merged_val_masks.update(masks)
                
                iou_results, dice_results, _, _ = eval_metrics(merged_val_masks, self.gt_masks, self.num_classes)
            else:
                # Other ranks pass None as the gather list
                dist.gather_object(val_masks, None, dst=0)
        else:
            # Single-GPU or non-distributed case
            iou_results, dice_results, _, _ = eval_metrics(val_masks, self.gt_masks, self.num_classes)
        del val_masks

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.loggers.info(f'Val - Epoch: {epoch+1}/{self.num_epochs};')   
            self.loggers.info(f'IoU_Results: {iou_results};')
            self.loggers.info(f'Dice_Results: {dice_results}.')
            self.writer.add_scalar('val/iou', iou_results['IoU'], epoch)
            self.writer.add_scalar('val/dice', dice_results['Dice'], epoch)
            
            return dice_results['Dice']


def main(args):
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        init_seeds(seed=args.seed)
        
        # Set loggers and writer
        loggers, writer = set_logging_and_writer(args)
        # Load datasets
        train_dataloader, val_dataloader = set_dataloaders(args)
        # Build model
        model = set_model(args)
        model.to(device)
        # Create trainer
        trainer = Trainer(args, model, train_dataloader, val_dataloader, loggers, writer)
        # Train
        trainer.main()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    
    seed=args.seed+rank
    init_seeds(seed=seed)
    
    # Set loggers and writer
    loggers, writer = set_logging_and_writer(args)
    # Load datasets
    train_dataloader, val_dataloader = set_dataloaders(args)
    # Build model
    model = set_model(args)
    model.to(args.device)
    # Create trainer
    trainer = Trainer(args, model, train_dataloader, val_dataloader, loggers, writer)
    # Train
    trainer.main()
    
    cleanup()


def set_logging_and_writer(args):
    now = datetime.now().strftime('%Y%m%d-%H%M')
    task = f'{args.sam_mode}_{args.model_type}_{now}'
    save_dir = osp.join(args.save_dir, args.run_name, args.dataset, f'few_shot_{int(args.scale*100)}', task)
    
    writer = SummaryWriter(osp.join(save_dir, 'runs'))
    save_log_dir = osp.join(save_dir, 'log')
    args.save_ckpt_dir = osp.join(save_dir, 'ckpt')
    
    os.makedirs(save_log_dir, exist_ok=True)
    os.makedirs(args.save_ckpt_dir, exist_ok=True)

    loggers = get_logger(os.path.join(save_log_dir, f'{task}.log'))
    
    return loggers, writer


def set_dataloaders(args):
    
    # ======> Load Dataset-Specific Parameters
    scale = args.scale
    data_root_dir = osp.join(args.data_root_dir, args.task, args.dataset)
    train_dataset = TrainingDataset(
                        data_root_dir=data_root_dir,
                        scale=scale
                    )
    val_dataset = TestingDataset(
                        data_root_dir=data_root_dir,
                        mode='val',
                    ) 
    
    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  sampler=train_sampler,
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers,
                                  shuffle=shuffle,
                                  pin_memory=True,)
    val_dataloader = DataLoader(dataset=val_dataset, 
                                sampler=val_sampler,
                                batch_size=32,
                                num_workers=32,
                                shuffle=False, 
                                pin_memory=True,)
    
    return train_dataloader, val_dataloader


def set_model(args):
    
    # ======> Load Prototype-based Model
    model = PGP_SAM(
                sam_checkpoint=args.sam_ckpt,
                sam_mode=args.sam_mode,   
                model_type=args.model_type,
                stage=args.stage,
                mask_size=args.image_size,
                resolution=args.resolution,
                num_classes=args.num_classes,
                num_tokens=args.num_tokens,
            ) 
    
    # set requires_grad to False to the whole model 
    for params in model.parameters():
        params.requires_grad=False
        
    # finetune correct weights
    for name, params in model.named_parameters(): 
        if 'image_encoder' in name and 'lora_linear' in name:
            params.requires_grad = True
        if 'mask_decoder' in name:
            params.requires_grad = True
        if 'prototype_prompt_encoder' in name:
            params.requires_grad = True
        if 'pre_prompt' in name:
            params.requires_grad = True
        if 'global_prototypes' in name:
            params.requires_grad = True

    model.to(args.device)
    
    if args.multi_gpu:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
    return model


def device_config(args):
    try:
        if not args.multi_gpu:
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = torch.device(f"cuda")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def init_seeds(seed=42, cuda_deterministic=True):
    # set seed for reproducibility 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  
        # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  
        # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main(args)

