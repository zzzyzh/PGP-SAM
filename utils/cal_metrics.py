from medpy import metric

import numpy as np 
import torch 


def eval_metrics(eval_masks, gt_eval_masks, num_classes=4):
    """Given the predicted masks and groundtruth annotations, predict the IoU, mean class IoU, and the IoU for each class / Dice, mean class Dice and the Dice for each class.
    Args:
        eval_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_eval_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    iou_results = dict()
    dice_results = dict()
    
    all_im_iou_acc = [] 
    all_im_dice_acc = []
    
    class_ious = {c: [] for c in range(1, num_classes)}
    class_dices = {c: [] for c in range(1, num_classes)}
    cum_I, cum_U = 0, 0
    
    for file_name, prediction in eval_masks.items():
        
        full_mask = gt_eval_masks[file_name]
        im_iou = []
        im_dice = []
        
        target = np.copy(full_mask)
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_dice_acc.append(0)
                all_im_iou_acc.append(0)
                for class_id in gt_classes:
                    class_dices[class_id].append(0)
                    class_ious[class_id].append(0)
            continue

        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask == class_id).astype(np.float64)
            assert current_pred.shape[-2:] == current_target.shape[-2:]

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                dice = compute_mask_dice(current_pred, current_target)
                im_dice.append(dice)
                i, u, iou = compute_mask_IoU(current_pred, current_target)     
                im_iou.append(iou)
                cum_I += i
                cum_U += u

                class_dices[class_id].append(dice)
                class_ious[class_id].append(iou)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.nanmean(im_iou))
        if len(im_dice) > 0:
            all_im_dice_acc.append(np.nanmean(im_dice))

    # calculate final metrics
    mean_im_iou = np.nanmean(all_im_iou_acc)
    mean_im_dice = np.nanmean(all_im_dice_acc)
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    mean_class_dice = torch.tensor([torch.tensor(values).float().mean() for c, values in class_dices.items() if len(values) > 0]).mean().item()
    mean_imm_iou = cum_I / (cum_U + 1e-15)
    
    dice_per_class = []
    cIoU_per_class = []
    for c in range(1, num_classes):
        final_class_im_dice = torch.tensor(class_dices[c]).float().mean()
        dice_per_class.append(round((final_class_im_dice*100).item(), 2))
        final_class_im_iou = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou*100).item(), 2))
        
    iou_results["IoU"] = round(mean_im_iou*100,2)
    iou_results["mcIoU"] = round(mean_class_iou*100,2)
    iou_results["mIoU"] = round(mean_imm_iou*100,2)
    iou_results["cIoU_per_class"] = cIoU_per_class
    
    dice_results["Dice"] = round(mean_im_dice*100,2)
    dice_results["mcDice"] = round(mean_class_dice*100,2)
    dice_results["Dice_per_class"] = dice_per_class
    
    iou_csv = [iou_results["IoU"], iou_results["mcIoU"], iou_results["mIoU"]] + iou_results["cIoU_per_class"]
    dice_csv = [dice_results["Dice"], dice_results["mcDice"]] + dice_results["Dice_per_class"] 
    
    return iou_results, dice_results, np.array(iou_csv), np.array(dice_csv)


def compute_mask_dice(masks, target):
    """compute dice used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    intersection = (masks * target).sum()
    union = (masks + target).sum()
    dice = (2. * intersection) / union
    return dice


def compute_mask_IoU(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, intersection / (union+1e-15)


def eval_hd95(gt_masks, val_masks, num_classes):
    metric_hd95 = []
    for i in range(1, num_classes):
        metric_cls_hd95 = []
        for key in val_masks.keys():   
            metric_cls_hd95.append(compute_hd95(val_masks[key]==i, gt_masks[key]==i))
        metric_hd95.append(np.mean(metric_cls_hd95, axis=0))
    hd95 = np.mean(metric_hd95, axis=0)

    return hd95


def compute_hd95(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    pred = np.array(pred)
    gt = np.array(gt)
    
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

