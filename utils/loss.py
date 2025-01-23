import math

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def cal_seg_loss(preds, gts, dice_loss_model, ce_loss_model, dice_weight=0.8):
    loss = {}
    for i, pred in enumerate(preds):
        _, _, h, w = pred.shape
        gt = F.interpolate(gts, (h,w), mode='nearest')
        
        dice_loss = dice_loss_model(pred, gt.squeeze(1).to(torch.long))
        ce_loss = ce_loss_model(pred, gt.squeeze(1).to(torch.long))
        
        loss[i] = {'loss': (1-dice_weight) * ce_loss + dice_weight * dice_loss,
                   'ce_loss': ce_loss,
                   'dice_loss': dice_loss
                   }
    
    return loss
    

# Segment Loss funcation
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        """
        Initialize DiceLoss.
        :param ignore_index: Class index to ignore, e.g., the background class. None means no class is ignored.
        :param smooth: Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, preds, masks):
        """
        Compute the Dice Loss.
        :param preds: Model predictions, shape [B, C, H, W]
        :param masks: Ground truth masks, shape [B, H, W]
        """
        
        preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]
        masks = F.one_hot(masks, num_classes).permute(0, 3, 1, 2).float()  # Convert to one-hot encoding
        
        intersection = torch.sum(preds * masks, dim=(0, 2, 3))
        union = torch.sum(preds * preds, dim=(0, 2, 3)) + torch.sum(masks * masks, dim=(0, 2, 3))
        dice_score = 2.0 * intersection / (union + self.smooth)
        
        loss = 1 - dice_score
        return loss.mean()  # Return the average Dice Loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=9, ignore_index=None):
        """
        Initialize FocalLoss.
        :param alpha: Balancing factor for classes. Can be a float or a list of values per class.
                      A value between [0, 1) will adjust the importance of foreground vs. background.
        :param gamma: Focusing parameter. Higher gamma reduces the loss contribution from easy examples.
        :param num_classes: The number of classes for the task.
        :param ignore_index: Class index to ignore in the loss calculation, e.g., background class.
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, float):
            assert 0 <= alpha < 1, "alpha should be in [0, 1)"
            self.alpha = torch.tensor([alpha] + [1 - alpha] * (num_classes - 1))
        elif isinstance(alpha, (list, tuple)):
            assert len(alpha) == num_classes, "Length of alpha must match num_classes"
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds, masks):
        """
        Calculate focal loss for segmentation
        :param preds: predictions from the model, shape: [B, C, H, W]
        :param masks: ground truth masks, shape: [B, H, W]
        :return: computed focal loss
        """
        if not preds.shape[1] == self.num_classes:
            raise ValueError(f"Expected input tensor to have {self.num_classes} channels, got {preds.shape[1]}")
        
        preds = preds.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        preds = preds.view(-1, self.num_classes)  # Flatten [N, C] where N is B*H*W
        masks = masks.view(-1)  # Flatten label tensor
        
        preds_logsoft = F.log_softmax(preds, dim=1)  # Log softmax on the class dimension
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, masks.unsqueeze(1)).squeeze(1)
        preds_logsoft = preds_logsoft.gather(1, masks.unsqueeze(1)).squeeze(1)
        
        self.alpha = self.alpha.to(preds.device)  # Ensure alpha is on the correct device
        alpha = self.alpha.gather(0, masks)
        
        loss = -alpha * torch.pow((1 - preds_softmax), self.gamma) * preds_logsoft
        return loss.mean()


# LR scheduler
class LogLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, lr, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.base_lr = lr
        super(LogLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: Scale up linearly from lr/warmup_iters to lr
            lr = (self.base_lr / self.warmup_iters) * (self.last_epoch + 1)
        else:
            # Post-warmup logarithmic decay
            shift_iter = self.last_epoch - self.warmup_iters
            lr = self.base_lr * (1.0 - shift_iter / self.total_iters) ** 0.9
        
        return [lr for _ in self.optimizer.param_groups]
    

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, base_lr, base_lr_end=0.0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.base_lr = base_lr
        self.base_lr_end_ratio = base_lr_end / self.base_lr
        self.warmup_factor = 1. / warmup_iters if warmup_iters > 0 else 1.
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase
            return [self.base_lr * (self.last_epoch + 1) * self.warmup_factor for _ in self.base_lrs]
        else:
            # Cosine Annealing phase
            progress = (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            return [self.cosine_annealing(self.base_lr, self.base_lr * self.base_lr_end_ratio, progress) for _ in self.base_lrs]

    def cosine_annealing(self, start_lr, end_lr, progress):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(math.pi * progress))

