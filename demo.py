import os
from tqdm import tqdm
from stomatapy.core.core import get_paths, imread_rgb
from stomatapy.core.isat import Anything2ISAT
from stomatapy.models.sam_hq import SAMHQ
from matplotlib import pyplot as plt 
import numpy as np


# samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04,
#                  'pred_iou_thresh':0.9, "box_nms_thresh": 0.7, "ellipse_threshold": 0.7, "statistics_filter": True}  

samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04,
                 'pred_iou_thresh':0.0, "box_nms_thresh": 1.0 , "ellipse_threshold": 0.0, "statistics_filter": False} 


sample_image_dir = 'asserts/sample images'

def show_masks(image: np.ndarray, masks: dict, random_color: bool = False, save: bool = False, output_dir: str = None, image_name: str = None) -> None:
    """Visualize and/or save the autolabeled image with masks"""
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    
    if len(masks) > 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        image_rgba = np.ones((*image.shape[:2], 4))
        image_rgba[:, :, 3] = 0
        
        for mask in masks:
            if random_color:
                color_mask = np.concatenate([np.random.random(3), [0.35]])
            else:
                color_mask = np.concatenate([(1, 0, 0), [0.35]])
            image_rgba[mask] = color_mask
        
        ax.imshow(image_rgba)
        plt.axis('off')
        
        if save and output_dir and image_name:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_masked.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
        plt.show()
    else:
        plt.show()

    return None


def get_annotations(sample_image_dir: str, samhq_configs:dict, catergory: str = 'pavement cell', visualize: bool = False, random_color: bool = True) -> None:
    """
    Generate ISAT annotations JSON files for images in a specified directory using SAM-HQ configurations
    
    Args:
    - sample_image_dir (str): a directory containing sample images
    - samhq_configs (dict): the configuration for SAM-HQ including points_per_side, min_mask_ratio, and max_mask_ratio (ratio means size percentage to the entire image)
    - category (str): the category label for the annotations, defaults to 'pavement cell'
    - visualize (bool): if True, visualize the masks on images
    - random_color (bool): if True and visualizing, use random colors for masks

    Returns:
    - None: annotations are generated and potentially visualized
    """
    points_per_side, min_mask_ratio, max_mask_ratio = samhq_configs['points_per_side'], samhq_configs['min_mask_ratio'], samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
    pred_iou_thresh, box_nms_thresh, ellipse_threshold, statistics_filter \
        = samhq_configs['pred_iou_thresh'], samhq_configs["box_nms_thresh"], samhq_configs["ellipse_threshold"], samhq_configs["statistics_filter"]

    image_paths = get_paths(sample_image_dir, '.tif') + get_paths(sample_image_dir, '.jpg')  # get the image paths under the species folder
    for image_path in tqdm(image_paths, total=len(image_paths)):
        image, masks = imread_rgb(image_path), []  # load the image in RGB scale
        try:
            auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, box_nms_thresh=box_nms_thresh, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=ellipse_threshold,statistics_filter=statistics_filter)  # get the auto labelled masks
            # auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.7)  # get the auto labelled masks
            # masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
            masks = auto_masks
            if visualize:
                visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                show_masks(image, visual_masks, random_color=random_color, save=True, output_dir='output_masks_nopost_all', image_name=os.path.basename(image_path))  # visualize and/or save masks
                # show_masks(image, visual_masks, random_color=random_color, save=True, output_dir='output_masks_nopost_all', image_name='M')  # visualize and/or save masks
            # if len(masks) > 0:
            #     Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
        except ValueError:
            pass
    return None


get_annotations(sample_image_dir, samhq_configs, visualize=True)