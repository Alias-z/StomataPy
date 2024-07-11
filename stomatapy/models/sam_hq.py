"""Module providing functions autolabeling images with SAM-HQ"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args, wrong-import-position
import os  # interact with the operating system
import re  # regular expression operations
from typing import Literal  # to support type hints
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='segment_anything_hq.modeling.tiny_vit_sam')  # noqa: suppress SAM-HQ import warnings
import numpy as np  # NumPy
import torch  # PyTorch
from matplotlib import pyplot as plt  # for image visualization
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor  # import SAM-HQ functions
from ..core.core import device, imread_rgb, get_checkpoints, suppress_stdout  # import core elements
from ..core.isat import UtilsISAT  # to interact with ISAT jason files


class SAMHQ:
    """
    Automatic mask generation with SAM-HQ https://github.com/SysCV/sam-hq
    """
    def __init__(self,
                 image_path: str = None,
                 checkpoint_path: str = 'Checkpoints/SAM-HQ/sam_hq_vit_h.pth',
                 points_per_side: tuple = (23, 24),
                 pred_iou_thresh: float = 0.9,
                 box_nms_thresh: float = 0.7,
                 crop_n_layers: int = 1,
                 min_mask_ratio: float = 0.02,
                 max_mask_ratio: float = 0.035):
        self.image_path = image_path  # the path of the image to be autolabeled
        self.image = imread_rgb(self.image_path)  # load the image in RGB scale
        self.image_area = self.image.shape[0] * self.image.shape[1]  # the area of the image
        self.checkpoint_path = checkpoint_path  # SAM-HQ checkpoint path
        self.model_type = re.search(r'vit_\w+', os.path.basename(self.checkpoint_path)).group()  # SAM-HQ model type: vit_tiny, vit_b, vit_l, or vit_h
        self.points_per_side = points_per_side  # the number of points to be sampled along one side of the image, total = points_per_side**2
        self.pred_iou_thresh = pred_iou_thresh  # a filtering threshold in [0,1], using the model's predicted mask quality.
        self.box_nms_thresh = box_nms_thresh  # the box IoU cutoff used by non-maximal suppression to filter duplicate masks.
        self.crop_n_layers = crop_n_layers  # if >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
        self.min_mask_area = self.image_area * min_mask_ratio  # filter out mask whose area is smaller than this
        self.max_mask_area = self.image_area * max_mask_ratio  # filter out mask whose area is larger than this

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)  # create directory for the checkpoint
            get_checkpoints('https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth', self.checkpoint_path)  # download the checkpoint if not exist
            assert os.path.exists(self.checkpoint_path), 'Checkpoint download failed.'  # in case download failure

    @staticmethod
    def isolate_masks(masks: list) -> list:
        """When two masks' bboxes intersect, keep the most elliptical mask"""
        to_remove = set()  # to store redundant masks
        for idx, mask_1 in enumerate(masks):
            for idx_j, mask_2 in enumerate(masks):
                if idx >= idx_j:
                    continue  # avoid redundant comparisons
                if UtilsISAT.bbox_intersection(mask_1['bbox'], mask_2['bbox'], threshold=0.3):
                    if mask_1['ellipse_iou'] > mask_2['ellipse_iou']:
                        to_remove.add(idx_j)  # compare scores and mark the mask with the lower score for removal
                    else:
                        to_remove.add(idx)
        masks = [mask for idx, mask in enumerate(masks) if idx not in to_remove]  # remove noisy masks
        return masks

    _model_cache = {}  # check if the model is already loaded

    def load_model(self) -> any:
        """Initialize SAM-HQ model"""
        if self.model_type not in SAMHQ._model_cache:
            with suppress_stdout():
                SAMHQ._model_cache[self.model_type] = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(device)  # load the model only if it hasn't been loaded before
        return SAMHQ._model_cache[self.model_type]

    def auto_label(self, ellipse_threshold: float = 0.9, statistics_filter: bool = True, use_isolate_masks: bool = False) -> list:
        """Auto label with SAM-HQ"""
        filtered_masks = []  # store ellipse filtered masks
        for points_per_side in self.points_per_side:
            mask_generator = SamAutomaticMaskGenerator(
                model=self.load_model(),
                points_per_side=points_per_side,
                points_per_batch=64,  # higher numer may cause errors!
                pred_iou_thresh=self.pred_iou_thresh,
                # The stability score is the IoU between the binary masks obtained by thresholding the predicted mask logits at high and low values.
                stability_score_thresh=0.9,  # a filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
                stability_score_offset=1.0,  # the amount to shift the cutoff when calculated the stability score.
                box_nms_thresh=self.box_nms_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_nms_thresh=self.box_nms_thresh,  # the box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
                crop_overlap_ratio=512 / 1500,  # sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
                crop_n_points_downscale_factor=1,  # the number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n
                point_grids=None,  # a list over explicit grids of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side.
                min_mask_region_area=100,  # if >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
                output_mode='binary_mask'  # the form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
            )
            masks = mask_generator.generate(self.image)  # auto predict the masks
            masks = [mask for mask in masks if self.min_mask_area <= mask['area'] <= self.max_mask_area]  # filter out masks that are too large
            for mask in masks:
                mask['is_ellipse'], mask['ellipse_iou'] = UtilsISAT.ellipse_filter(mask['segmentation'], ellipse_threshold)  # check if fitting ellipse well and get the ellipse_iou score
                if mask['is_ellipse']:
                    mask['bbox'] = UtilsISAT.boolmask2bbox(mask['segmentation'])  # get the bbox
                    filtered_masks.append(mask)  # filter out masks that are not ellipse
        if use_isolate_masks:
            filtered_masks = self.isolate_masks(filtered_masks)  # remove noisy masks

        if statistics_filter:
            areas = [mask['area'] for mask in filtered_masks]  # get all mask areas for statistical filtering
            if len(areas) >= 3:
                quartile_1, quartile_3 = np.percentile(areas, [25, 75])  # get the 1st and 3rd percentile
                iqr = quartile_3 - quartile_1  # calculate the interquartile range
                lower_boundary, upper_boundary = quartile_1 - 2.0 * iqr, quartile_3 + 1.5 * iqr  # set the area bounary
                filtered_masks = [mask for mask in filtered_masks if lower_boundary <= mask['area'] <= upper_boundary]  # final masks

        return filtered_masks

    def prompt_label(self, input_point: np.ndarray = None, input_label: np.ndarray = None, input_box: list = None, mode: Literal['single', 'multiple'] = 'single') -> list:
        """Generate masks with points or bboxes as prompts using SAM-HQ"""
        sam_hq = SamPredictor(self.load_model())  # initialize the model
        sam_hq.set_image(self.image)  # set predicting target
        input_box = np.array(input_box)
        if mode == 'single':
            segmentations, scores, _ = sam_hq.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False, hq_token_only=True)  # get predictions
        elif mode == 'multiple':
            masks = []  # to store masks
            input_box = torch.tensor(input_box, device=sam_hq.device)  # bbox to tensor
            transformed_box = sam_hq.transform.apply_boxes_torch(input_box, self.image.shape[:2])  # bboxes transformation
            segmentations, scores, _ = sam_hq.predict_torch(point_coords=None, point_labels=None, boxes=transformed_box, multimask_output=False, hq_token_only=True)  # get predictions
            segmentations = segmentations.squeeze(1).cpu().numpy()  # masks to cpu
            scores = scores.squeeze(1).cpu().numpy()  # scores to cpu
            input_box = input_box.cpu().numpy()[scores > self.pred_iou_thresh]  # filter out low mask-score bboxes
            segmentations = segmentations[scores > self.pred_iou_thresh]  # filter out masks that have a score lower than threshold
            for segmentation in segmentations:
                is_ellipse, ellipse_iou = UtilsISAT.ellipse_filter(segmentation)  # ellipse score
                mask = {'segmentation': segmentation, 'area': np.sum(segmentation), 'bbox': UtilsISAT.boolmask2bbox(segmentation), 'is_ellipse': is_ellipse, 'ellipse_iou': ellipse_iou}  # get mask information
                masks.append(mask)  # collect all stomatal masks of the image
        return masks

    @staticmethod
    def show_masks(image: np.ndarray, masks: dict, random_color: bool = False) -> None:
        """Visualize the autolabled image"""
        plt.figure(figsize=(20, 20)); plt.imshow(image)  # noqa: set the figure size and show the image
        if len(masks) > 0:
            ax = plt.gca(); ax.set_autoscale_on(False)  # noqa: get current axes
            image_rgba = np.ones((*image.shape[:2], 4))  # add an alph channel
            image_rgba[:, :, 3] = 0  # initially transparent (alpha channel) set to 0
            for mask in masks:
                if random_color:
                    color_mask = np.concatenate([np.random.random(3), [0.35]])  # using random colors for each mask
                else:
                    color_mask = np.concatenate([(1, 0, 0), [0.35]])  # use fixed color for all masks
                image_rgba[mask] = color_mask  # set the mask color
            ax.imshow(image_rgba); plt.axis('off'); plt.show()  # noqa: show the visualization
        else:
            plt.show()
        return None
