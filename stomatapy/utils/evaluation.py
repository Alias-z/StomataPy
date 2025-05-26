import numpy as np
import torch
from surface_distance import compute_surface_distances, compute_dice_coefficient, compute_robust_hausdorff

def compute_segmentation_metrics(gt_seg, pred_seg):

    max_class_no_fix = int(max(gt_seg.max(), pred_seg.max()))
    max_class_no_mov = int(max(gt_seg.max(), pred_seg.max()))
    max_class_no = max(max_class_no_fix, max_class_no_mov)
    if max_class_no == 0:
        raise ValueError("No classes found in the segmentations.")

    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.cpu().numpy()

    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy() 

    dice_metrics, hd95_metrics = {}, {}

    for i in range(1, max_class_no+1):
        if ((gt_seg==i).sum()==0) or ((pred_seg==i).sum()==0):
            dice_metrics[f"dice_{i}"] = np.nan
            hd95_metrics[f"hd95_{i}"] = np.nan
            continue
        dice_metrics[f"dice_{i}"] = compute_dice_coefficient((gt_seg==i), (pred_seg==i))
        hd95_metrics[f"hd95_{i}"] = compute_robust_hausdorff(compute_surface_distances((gt_seg==i), (pred_seg==i), np.ones(3)), 95.)
    
    dice = np.nanmean(list(dice_metrics.values()))
    hd95 = np.nanmean(list(hd95_metrics.values()))

    return {'dice': float(dice), **dice_metrics, 'hd95': float(hd95), **hd95_metrics}
