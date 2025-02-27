"""module providing functions autolabeling images with openmmlab models"""

# pylint: disable=line-too-long, import-error, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args, wrong-import-position, undefined-loop-variable, unused-import, no-name-in-module
import os  # interact with the operating system
import json  # manipulate json files
import random  # suppress xformers to generate random predictions results
from typing import List  # to support type hints
import warnings; warnings.filterwarnings('ignore', message='.*in an upcoming release, it will be required to pass the indexing argument.*'); warnings.filterwarnings('ignore', message='Failed to add*'); warnings.filterwarnings('ignore', message='xFormers is available'); warnings.filterwarnings('ignore', module=r'.*dino_layers.*'); warnings.filterwarnings('ignore', message='The current default scope .* is not .*')  # noqa: supress warning messages
import numpy as np  # numpy for numerical operations
import cv2  # opencv for image processing
from PIL import Image  # pillow image processing
from scipy import ndimage as ndi  # for hole filling
from skimage.measure import label  # for using instance segmentation results as detected objects
import torch  # pytorch
from tqdm import tqdm  # progress bar
from matplotlib import pyplot as plt  # for image visualization
import matplotlib.patches as patches  # for drawing boxes and shapes on plots
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from sahi.auto_model import AutoDetectionModel  # sahi wrapper for mmdetection
from sahi.predict import get_sliced_prediction  # sahi sliced prediction
from mmseg.utils import register_all_modules as mmseg_utils_register_all_modules  # register mmseg modules
from mmseg.apis import init_model as mmseg_apis_init_model  # initialize mmseg model
from mmseg.apis import inference_model as mmseg_apis_inference_model  # mmseg inference segmentor
from ..core.core import device, Cell_Colors, imread_rgb, imread_rgb_stack, resize_and_pad_image, restore_original_dimensions  # import core elements
from ..core.isat import UtilsISAT, Anything2ISAT  # to interact with isat jason files
from ..utils.data4training import Data4Training  # the traning processing pipelines
from ..utils.focus_stack import focus_stack  # focus stacking


def set_seeds(seed: int = 42) -> None:
    """
    set the random seeds for reproducibility in python, numpy, and pytorch.

    args:
    - seed (int): the seed value to use. default is 42.
    """
    random.seed(seed)  # set python random seed
    np.random.seed(seed)  # set numpy random seed
    torch.manual_seed(seed)  # set pytorch random seed
    if torch.cuda.is_available():  # check if cuda is available
        torch.cuda.manual_seed(seed)  # set cuda random seed
        torch.cuda.manual_seed_all(seed)  # if you are using multi-gpu, set all gpu seeds
    torch.backends.cudnn.enabled = False  # disable cudnn
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # disable cudnn benchmark
    return None  # return none


class OpenMMlab(Data4Training):
    """
    automatic mask generation with mmdet (https://github.com/open-mmlab/mmdetection) and mmsegmentation (https://github.com/open-mmlab/mmsegmentation)
    """
    def __init__(self,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 detector_threshold: float = 0.2,
                 segmentor_config_path: str = None,
                 segmentor_weight_path: str = None,
                 seg_onehot_mapping: dict = {cell_color.class_encoding: cell_color.class_name for cell_color in Cell_Colors},
                 stack_input: bool = False,  # add parameter for stack input
                 **kwargs):
        super().__init__(**kwargs)  # initialize parent class
        self.detector_config_path = detector_config_path  # object detection config path
        self.detector_weight_path = detector_weight_path  # object detection weight path
        self.detector_threshold = detector_threshold  # object detection threshold
        self.segmentor_config_path = segmentor_config_path  # semantic segmentation config path
        self.segmentor_weight_path = segmentor_weight_path  # semantic segmentation weight path
        self.seg_onehot_mapping = seg_onehot_mapping  # segmentation one-hot code against class_name
        self.seg_color_mapping = {cell_color.class_name: cell_color.mask_rgb for cell_color in Cell_Colors}  # map the segmentation class names to their colors
        self.stack_input = stack_input  # flag for stack image processing
        set_seeds(42); self.segmentor = mmseg_apis_init_model(self.segmentor_config_path, self.segmentor_weight_path, device='cpu')   # noqa: initialize a segmentor from config file

    def detect_cell(self,
                    image_paths: List[str],
                    if_resize_image: bool = True,
                    if_keep_ratio: bool = True,
                    if_visualize: bool = False,
                    if_auto_label: bool = True,
                    if_standard_pred: bool = False) -> List[np.ndarray]:
        """
        detect objects in a list of images and return their bounding boxes and masks
        each image is processed through an object detection model

        args:
        - image_paths (list[str]): list of image paths
        - if_resize_image (bool): if true, resize and pad image to target dimension before predictions
        - if_keep_ratio (bool): if true, maintains aspect ratio while resizing
        - if_visualize (bool): if true, visualize the detection results
        - if_auto_label (bool): if true, convet predictions to isat json files as well

        returns:
        - valid_predictions (list[dict]): a list of dictionaries containing detection results per image with keys 'image_path', 'category_id', 'category_name', 'bboxes', 'masks'
        """
        valid_predictions, focus_optimized_results = [], []  # initialize the lists

        def visualize_detections(valid_prediction: dict) -> None:
            """visualizes detection results on an image using bounding boxes and optional masks

            args:
            - valid_prediction (dict): containing 'bboxes' [x_min, y_min, x_max, y_max] and optionally 'masks' [[x1, y1], [x2, y2], ..., [xn, yn]]

            returns:
            - none. just plot the detection results
            """
            image = imread_rgb(valid_prediction['image_path'])  # load the image
            _, ax = plt.subplots(1); ax.imshow(image)  # noqa: add the image to plot
            if valid_prediction['masks'] is not None:  # check if masks exist
                for mask in valid_prediction['masks']:  # iterate through masks
                    polygon = plt.Polygon(mask, closed=True, fill=True, color='red', alpha=0.5)  # plot color and transparency
                    ax.add_patch(polygon)  # add masks if they exist

            for bbox in valid_prediction['bboxes']:  # iterate through bounding boxes
                box = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='blue', facecolor='none')  # box color
                ax.add_patch(box)  # always draw bounding boxes
            ax.set_title(f"{os.path.basename(valid_prediction['image_path'])}")  # add the image file name
            plt.axis('off'); plt.show()  # noqa: show the plot

        # modified image loading to handle stack images
        if self.stack_input:  # check if processing stack inputs
            images = [imread_rgb_stack(image_path) for image_path in image_paths]  # use the expanded frames list for processing
        else:  # single image processing
            images = [imread_rgb(image_path) for image_path in image_paths]  # single-frame image processing

        # handle resizing differently for stack vs single images
        if if_resize_image:  # check if resizing is required
            if self.stack_input:  # handle stack images
                # process each stack while maintaining the stack structure
                resized_stacks, resizing_metadata = [], []  # to collect the resized stacks and their metadata

                for stack_idx, stack in enumerate(images):  # iterate through each stack
                    # calculate metadata once per stack
                    first_frame = stack[0]  # get first frame
                    first_frame_pil = Image.fromarray(first_frame)  # convert to pil
                    original_width, original_height = first_frame_pil.size  # get dimensions

                    if if_keep_ratio:  # maintain aspect ratio
                        ratio = min(self.new_width / original_width, self.new_height / original_height)  # calculate ratio
                        resize_width, resize_height = int(original_width * ratio), int(original_height * ratio)  # apply ratio
                    else:  # use target dimensions directly
                        resize_width, resize_height = self.new_width, self.new_height  # use target dimensions

                    # store metadata for this stack
                    stack_metadata = {
                        'original_stack': stack.copy(),
                        'width_ratio': resize_width / original_width,
                        'height_ratio': resize_height / original_height,
                        'padding_horizontal': (self.new_width - resize_width) // 2,
                        'padding_vertical': (self.new_height - resize_height) // 2
                    }  # store metadata for all frames in stack

                    padding_h, padding_v = stack_metadata['padding_horizontal'], stack_metadata['padding_vertical']   # create resized stack with numpy operations

                    frames_batch = stack.reshape(-1, stack.shape[1], stack.shape[2], 3)  # reshape to (total_frames, height, width, channels)
                    resized_batch = np.zeros((frames_batch.shape[0], resize_height, resize_width, 3), dtype=np.uint8)  # create an empty array for resized batch
                    for idx in range(frames_batch.shape[0]):  # iterate through each frame
                        resized_batch[idx] = cv2.resize(frames_batch[idx], (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)  # resize all frames at once
                    resized_stack = np.zeros((stack.shape[0], self.new_height, self.new_width, 3), dtype=np.uint8)  # create output stack with padding
                    resized_stack[:, padding_v:padding_v + resize_height, padding_h:padding_h + resize_width, :] = resized_batch  # add all frames to their padded positions
                    resized_stacks.append(resized_stack)  # append the complete resized stack
                    resizing_metadata.append(stack_metadata)  # store metadata for this stack
                images = resized_stacks  # replace with resized stacks
            else:  # handle single images
                # original single-image resizing logic
                resized_images, resizing_metadata = [], []  # to collected the resizing information
                for image in images:  # iterate through each image
                    metadata = {}  # to collect the resizing metadata for each image
                    image_pil = Image.fromarray(image)  # np.array to pil image
                    original_width, original_height = image_pil.size  # the original width and height

                    if if_keep_ratio:  # maintain aspect ratio
                        ratio = min(self.new_width / original_width, self.new_height / original_height)  # get the closest ratio to destination dimension
                        resize_width, resize_height = int(original_width * ratio), int(original_height * ratio)  # resize width and height according to the ratio
                    else:  # use target dimensions directly
                        resize_width, resize_height = self.new_width, self.new_height  # resize width and height to destination dimensions

                    width_ratio, height_ratio = resize_width / original_width, resize_height / original_height  # the resize ratio

                    image_resized = image_pil.resize((resize_width, resize_height), Image.LANCZOS)  # noqa: resize the image
                    new_image = Image.new('RGB', (self.new_width, self.new_height), (0, 0, 0))  # create a black image in target dimension
                    padding_horizontal, padding_vertical = (self.new_width - resize_width) // 2, (self.new_height - resize_height) // 2  # calculate padding values
                    new_image.paste(image_resized, (padding_horizontal, padding_vertical))  # pad the image to center
                    resized_images.append(np.array(new_image))  # append the resized image
                    metadata = {
                        'original_image': image.copy(),
                        'width_ratio': width_ratio,
                        'height_ratio': height_ratio,
                        'padding_horizontal': padding_horizontal,
                        'padding_vertical': padding_vertical
                    }  # collect the resize ratio and the padding values
                    resizing_metadata.append(metadata)  # append the resizing metadata
                images = resized_images  # replace the images variable with the resized ones

        valid_predictions = []  # for predictions whose score > threshold

        # TODO: handle stack input case
        if not self.use_sahi:  # check if using regular mmdet detection
            mmdet_utils_register_all_modules(init_default_scope=False)  # initialize mmdet scope
            detector = mmdet_apis_init_detector(self.detector_config_path, self.detector_weight_path, device=device)  # initialize a detector from config file
            # print(detector.cfg)
            # print(detector)
            category_names = detector.dataset_meta['classes']  # get the category names
            category_mapping = {str(inx): category_name for inx, category_name in enumerate(category_names)}  # get the catergory mapping
            for idx, image in tqdm(enumerate(images), total=len(images)):  # process each image with progress bar
                prediction = mmdet_apis_inference_detector(detector, image)  # inference image(s) with the detector
                valid_indices = torch.where(prediction.pred_instances.scores > self.detector_threshold)[0]   # filter based on the detection threshold
                if valid_indices.numel() == 0:  # check if any valid predictions
                    result_dict = {'category_id': None, 'category_name': None, 'bboxes': None, 'masks': None}  # if no detections exceed the threshold, store none for this image
                else:  # process valid predictions
                    category_id = prediction.pred_instances.labels[valid_indices].cpu().numpy()  # get category ids
                    masks = prediction.pred_instances.masks[valid_indices].cpu().numpy() if hasattr(prediction.pred_instances, 'masks') else None  # get masks if available
                    result_dict = {
                        'image_path': image_paths[idx],
                        'category_id': category_id,
                        'category_name': [category_mapping.get(str(category_id), 'Unknown') for category_id in category_id],
                        'bboxes': np.array(prediction.pred_instances.bboxes[valid_indices].cpu().numpy(), dtype=np.int32),
                        'masks': [UtilsISAT.mask2segmentation(mask) for mask in masks] if masks is not None else None
                    }  # collect prediction metadata
                    valid_predictions.append(result_dict)  # add result to valid predictions

        elif self.use_sahi:  # using sahi for detection
            detector = AutoDetectionModel.from_pretrained(
                model_type='mmdet',  # name of the detection framework
                model_path=self.detector_weight_path,  # path of the detection model (ex. 'model.pt')
                config_path=self.detector_config_path,  # path of the config file (ex. 'mmdet/configs/cascade_rcnn_r50_fpn_1x.py')
                confidence_threshold=self.detector_threshold,  # all predictions with score < confidence_threshold will be discarded
                image_size=self.slice_width,  # inference input size
                device=device  # device, "cpu" or "cuda:0"
            )  # initialize the sahi detector

            def is_straight_edge(p1: tuple, p2: tuple, min_length: int) -> bool:
                """
                check if two adjacent polygon points form a straight edge (horizontal/vertical) longer than specified length

                args:
                    p1 (tuple): first point coordinates (x1, y1)
                    p2 (tuple): second point coordinates (x2, y2)
                    min_length (int): minimum edge length in pixels to consider

                returns:
                    bool: true if edge is horizontal/vertical and longer than min_length
                """
                dx = abs(p1[0] - p2[0])  # horizontal distance between points
                dy = abs(p1[1] - p2[1])  # vertical distance between points
                return (dy == 0 and dx > min_length) or (dx == 0 and dy > min_length)  # check for long straight edges

            def has_straight_line_edges(coco_polygon: list, min_straight_length: int = 50) -> bool:
                """
                check if coco-format polygon contains any straight edges longer than specified length

                args:
                    coco_polygon (list): polygon in coco format [x1,y1,x2,y2,...]
                    min_straight_length (int): minimum straight edge length to detect (default=50px)

                returns:
                    bool: true if polygon contains qualifying straight edges
                """
                if len(coco_polygon) < 4:  # check if we have enough points
                    return False  # need at least 2 points (4 coordinates) to form an edge

                points = list(zip(coco_polygon[0::2], coco_polygon[1::2]))  # convert to [(x1,y1), (x2,y2),...]
                # check all consecutive point pairs including last-to-first connection
                for i in range(len(points)):  # iterate through points
                    p1 = points[i]  # current point
                    p2 = points[(i + 1) % len(points)]  # wrap index for final edge connection
                    if is_straight_edge(p1, p2, min_straight_length):  # check if this edge is straight
                        return True  # found a straight edge
                return False  # no qualifying straight edges found

            if self.stack_input:
                # create focus-stacked base images for each stack
                focus_bases = [focus_stack(stack) for stack in images]  # focus stack each image stack

                for stack_idx, stack in tqdm(enumerate(images), total=len(images)):  # process each stack with progress bar
                    base_image = focus_bases[stack_idx].copy()  # copy the focus stack base image
                    all_frame_predictions = []  # store predictions from all frames

                    for frame_idx, frame in enumerate(stack):  # iterate through each frame
                        result = get_sliced_prediction(
                            image=frame,
                            detection_model=detector,
                            slice_width=self.slice_width,
                            slice_height=self.slice_height,
                            overlap_height_ratio=self.sahi_overlap_ratio,
                            overlap_width_ratio=self.sahi_overlap_ratio,
                            perform_standard_pred=if_standard_pred,
                            postprocess_type='GREEDYNMM',
                            postprocess_match_metric='IOU',
                            postprocess_match_threshold=0.1,
                            postprocess_class_agnostic=False,
                            verbose=0,
                            merge_buffer_length=None,
                            auto_slice_resolution=True,
                            slice_export_prefix=None,
                            slice_dir=None,
                        ).to_coco_annotations()  # get predictions in coco format

                        valid_items = []  # store valid detections
                        for item in result:
                            if 'segmentation' in item and item['segmentation'] and len(item['segmentation'][0]) >= 4:
                                seg_points = np.array(item['segmentation'][0])
                                x_coords = seg_points[0::2]  # all even indices (0, 2, 4, ...)
                                y_coords = seg_points[1::2]  # all odd indices (1, 3, 5, ...)
                                # compute bbox from segmentation points using NumPy
                                if len(x_coords) > 0 and len(y_coords) > 0:
                                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                                    # COCO format: [x_min, y_min, width, height]
                                    width, height = x_max - x_min, y_max - y_min
                                    computed_bbox = [float(x_min), float(y_min), float(width), float(height)]
                                    # update the item's bbox
                                    item['bbox'] = computed_bbox
                                    valid_items.append(item)

                        # filter out items with straight edges
                        valid_pairs = []  # store valid detection pairs
                        for item in valid_items:  # check each valid item
                            mask = item['segmentation']  # get segmentation mask
                            # check if mask is not empty and has expected structure
                            if mask and len(mask) > 0:  # ensure mask is not empty
                                if not has_straight_line_edges(mask[0], min_straight_length=20):  # check for straight edges
                                    valid_pairs.append({
                                        'frame_idx': frame_idx,
                                        'mask': mask,
                                        'bbox': np.array(item['bbox'], dtype=np.float32),
                                        'score': item['score'],
                                        'category_id': item['category_id'],
                                        'category_name': item['category_name'],
                                        'object_id': None  # will be assigned in grouping step
                                    })  # add valid pair with frame index
                            else:
                                print(f"Skipping item with invalid mask: {mask}")  # print when skipping invalid mask

                        all_frame_predictions.extend(valid_pairs)  # add valid pairs to all predictions

                    object_id, processed_predictions = 0, []  # group detections that represent the same object across frames

                    while all_frame_predictions:  # process until no predictions remain
                        current = all_frame_predictions.pop(0)  # get first prediction
                        current['object_id'] = object_id  # assign object id
                        same_object_predictions = [current]  # initialize list with current prediction

                        # find all detections of the same object across different frames
                        idx = 0  # initialize index
                        while idx < len(all_frame_predictions):  # iterate through remaining predictions
                            candidate = all_frame_predictions[idx]  # get candidate prediction
                            # calculate iou between bboxes
                            box1 = UtilsISAT.bbox_convert(current['bbox'], 'COCO2ISAT')  # convert current bbox to isat format
                            box2 = UtilsISAT.bbox_convert(candidate['bbox'], 'COCO2ISAT')  # convert candidate bbox to isat format
                            # calculate intersection coordinates
                            x1 = max(box1[0], box2[0])  # get maximum left coordinate
                            y1 = max(box1[1], box2[1])  # get maximum top coordinate
                            x2 = min(box1[2], box2[2])  # get minimum right coordinate
                            y2 = min(box1[3], box2[3])  # get minimum bottom coordinate

                            if x2 > x1 and y2 > y1:  # there is overlap
                                intersection = (x2 - x1) * (y2 - y1)  # calculate intersection area
                                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  # calculate first box area
                                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  # calculate second box area
                                union = box1_area + box2_area - intersection  # calculate union area
                                iou = intersection / union  # calculate iou

                                if iou > 0.5:  # threshold for same object
                                    candidate['object_id'] = object_id  # assign same object id
                                    same_object_predictions.append(candidate)  # add to same object predictions
                                    all_frame_predictions.pop(idx)  # remove from remaining predictions
                                    continue  # skip incrementing index
                            idx += 1  # increment index

                        # select best prediction for this object (highest score)
                        best_pred = max(same_object_predictions, key=lambda x: x['score'])  # get prediction with highest score
                        processed_predictions.append(best_pred)  # add to processed predictions
                        object_id += 1  # increment object id

                    # create focus-optimized image by patching best predictions
                    focus_optimized_image = base_image.copy()  # copy base image
                    for pred in processed_predictions:  # process each prediction
                        bbox_isat = UtilsISAT.bbox_convert(pred['bbox'], 'COCO2ISAT')  # convert bbox to isat format
                        x1, y1, x2, y2 = [int(c) for c in bbox_isat]  # get bbox coordinates as integers
                        # add padding to bbox for better visual result
                        pad = int(min(x2 - x1, y2 - y1) * 0.1)  # calculate padding as 10% of smaller dimension
                        x1_pad = max(0, x1 - pad)  # ensure left coordinate isn't negative
                        y1_pad = max(0, y1 - pad)  # ensure top coordinate isn't negative
                        x2_pad = min(focus_optimized_image.shape[1], x2 + pad)  # ensure right coordinate isn't beyond image width
                        y2_pad = min(focus_optimized_image.shape[0], y2 + pad)  # ensure bottom coordinate isn't beyond image height

                        # copy the best detection region from its source frame to the base image
                        source_frame = stack[pred['frame_idx']]  # get source frame
                        focus_optimized_image[y1_pad:y2_pad, x1_pad:x2_pad] = source_frame[y1_pad:y2_pad, x1_pad:x2_pad]  # copy region from source frame

                    base_dir = os.path.dirname(image_paths[stack_idx])  # get base directory from original image path
                    focus_opt_dir = os.path.join(base_dir, "focus_optimized")  # create path for focus_optimized folder
                    os.makedirs(focus_opt_dir, exist_ok=True)  # create focus_optimized directory if it doesn't exist
                    original_filename = os.path.basename(image_paths[stack_idx])  # extract original filename
                    output_filename = os.path.join(focus_opt_dir, original_filename)  # create path for output file

                    if if_resize_image:  # check if resizing was applied
                        stack_metadata = resizing_metadata[stack_idx]  # get metadata for current stack
                        original_height, original_width = stack_metadata['original_stack'][0].shape[:2]  # get original dimensions
                        padding_vertical = stack_metadata['padding_vertical']  # get vertical padding
                        padding_horizontal = stack_metadata['padding_horizontal']  # get horizontal padding
                        height, width = focus_optimized_image.shape[:2]  # get current image dimensions

                        if padding_vertical > 0 or padding_horizontal > 0:  # check if padding was applied
                            content_image = focus_optimized_image[  # crop out padding
                                padding_vertical:height - padding_vertical,
                                padding_horizontal:width - padding_horizontal
                            ]  # crop padded region
                        else:  # no padding
                            content_image = focus_optimized_image  # use full image if no padding

                        resized_image = cv2.resize(  # resize image back to original dimensions
                            content_image,
                            (original_width, original_height),
                            interpolation=cv2.INTER_LANCZOS4
                        )  # resize with high quality interpolation
                        cv2.imwrite(output_filename, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))  # save resized image

                        adjusted_bboxes = []  # initialize list for adjusted bounding boxes
                        adjusted_masks = []  # initialize list for adjusted masks

                        for pred in processed_predictions:  # process each prediction
                            x1, y1, x2, y2 = UtilsISAT.bbox_convert(pred['bbox'], 'COCO2ISAT')  # convert bbox to isat format

                            # remove padding and apply inverse scaling to bbox coordinates
                            adj_x1 = max(0, int((x1 - padding_horizontal) / stack_metadata['width_ratio']))  # adjust x1
                            adj_y1 = max(0, int((y1 - padding_vertical) / stack_metadata['height_ratio']))  # adjust y1
                            adj_x2 = min(original_width, int((x2 - padding_horizontal) / stack_metadata['width_ratio']))  # adjust x2
                            adj_y2 = min(original_height, int((y2 - padding_vertical) / stack_metadata['height_ratio']))  # adjust y2

                            adjusted_bboxes.append([adj_x1, adj_y1, adj_x2, adj_y2])  # add adjusted bbox to list

                            if pred['mask']:  # check if mask exists
                                mask_coords = pred['mask'][0]  # get mask coordinates
                                adjusted_mask = []  # initialize adjusted mask

                                for idx in range(0, len(mask_coords), 2):  # iterate through mask coordinates (x,y pairs)
                                    if idx + 1 < len(mask_coords):  # ensure we have both x and y
                                        mx = mask_coords[idx]  # get x coordinate
                                        my = mask_coords[idx + 1]  # get y coordinate

                                        # remove padding and apply inverse scaling to mask coordinates
                                        adj_mx = (mx - padding_horizontal) / stack_metadata['width_ratio']  # adjust x
                                        adj_my = (my - padding_vertical) / stack_metadata['height_ratio']  # adjust y

                                        adjusted_mask.extend([adj_mx, adj_my])  # add adjusted coordinates to mask

                                adjusted_masks.append([adjusted_mask] if adjusted_mask else [])  # add adjusted mask to list
                            else:  # no mask
                                adjusted_masks.append([])  # add empty mask if none exists

                        # convert adjusted coordinates to isat format
                        isat_bboxes = np.array(adjusted_bboxes, dtype=np.int32) if adjusted_bboxes else np.array([], dtype=np.int32)  # convert bboxes to numpy array
                        isat_masks = [UtilsISAT.coco_mask2isat_mask(mask[0]) for mask in adjusted_masks if mask and mask[0]] if adjusted_masks else []  # convert masks to isat format
                    else:  # no resizing
                        cv2.imwrite(output_filename, cv2.cvtColor(focus_optimized_image, cv2.COLOR_RGB2BGR))  # save image as is

                        # use predictions as-is, just convert format
                        isat_bboxes = np.array([UtilsISAT.bbox_convert(pred['bbox'], 'COCO2ISAT') for pred in processed_predictions], dtype=np.int32) if processed_predictions else np.array([], dtype=np.int32)  # convert bboxes
                        isat_masks = [UtilsISAT.coco_mask2isat_mask(pred['mask'][0]) if pred['mask'] else [] for pred in processed_predictions] if processed_predictions else []  # convert masks

                    # create result dictionary with properly scaled annotations
                    focus_result = {
                        'image_path': output_filename,  # path to focus-optimized image
                        'category_id': np.array([pred['category_id'] for pred in processed_predictions], dtype=np.int64),  # category ids
                        'category_name': [pred['category_name'] for pred in processed_predictions],  # category names
                        'bboxes': isat_bboxes,  # adjusted bounding boxes
                        'masks': isat_masks  # adjusted masks
                    }  # create result dictionary

                    focus_optimized_results.append(focus_result)  # add to focus-optimized results for json generation

                    vis_result = focus_result.copy()  # copy result for visualization
                    vis_result['image_path'] = image_paths[stack_idx]  # use original image path for visualization
                    valid_predictions.append(vis_result)  # add to valid predictions
            else:  # original sahi code for non-stack images
                # original sahi code for non-stack images
                for idx, image in tqdm(enumerate(images), total=len(images)):  # process each image with progress bar
                    result = get_sliced_prediction(
                        image=image,  # location of image or numpy image matrix to slice
                        detection_model=detector,  # model.detectionmodel
                        slice_width=self.slice_width,  # width of each slice
                        slice_height=self.slice_height,  # height of each slice
                        overlap_height_ratio=self.sahi_overlap_ratio,  # fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels)
                        overlap_width_ratio=self.sahi_overlap_ratio,  # fractional overlap in width of each window
                        perform_standard_pred=if_standard_pred,  # perform a standard prediction on top of sliced predictions to increase large object detection accuracy
                        postprocess_type='GREEDYNMM',  # type of the postprocess to be used after sliced inference while merging/eliminating predictions. options are 'nmm', 'greedynmm' or 'nms'. default is 'greedynmm'
                        postprocess_match_metric='IOU',  # metric to be used during object prediction matching after sliced prediction. 'iou' for intersection over union, 'ios' for intersection over smaller area.
                        postprocess_match_threshold=0.1,  # sliced predictions having higher iou than postprocess_match_threshold will be postprocessed after sliced prediction.
                        postprocess_class_agnostic=False,  # if true, postprocess will ignore category ids.
                        verbose=0
                    ).to_coco_annotations()  # get the prediction results in mscoco format

                    valid_items = []  # initialize array for valid items
                    for item in result:
                        # Check if item has a valid segmentation
                        if 'segmentation' in item and item['segmentation'] and len(item['segmentation'][0]) >= 4:
                            # Get segmentation points in MSCOCO format [x1, y1, x2, y2, ...]
                            seg_points = np.array(item['segmentation'][0])
                            
                            # Extract x and y coordinates using NumPy operations
                            x_coords = seg_points[0::2]  # All even indices (0, 2, 4, ...)
                            y_coords = seg_points[1::2]  # All odd indices (1, 3, 5, ...)
                            
                            # Compute bbox from segmentation points using NumPy
                            if len(x_coords) > 0 and len(y_coords) > 0:
                                x_min = np.min(x_coords)
                                y_min = np.min(y_coords)
                                x_max = np.max(x_coords)
                                y_max = np.max(y_coords)
                                
                                # COCO format: [x_min, y_min, width, height]
                                width = x_max - x_min
                                height = y_max - y_min
                                computed_bbox = [float(x_min), float(y_min), float(width), float(height)]
                                
                                # Update the item's bbox
                                item['bbox'] = computed_bbox
                                valid_items.append(item)

                    # filter valid items (filter out masks with straight edges)
                    valid_pairs = []  # store valid detection pairs
                    for item in valid_items:  # check each valid item
                        mask = item['segmentation']  # get segmentation mask
                        if not mask or not has_straight_line_edges(mask[0], min_straight_length=20):  # check if mask has straight edges
                            valid_pairs.append((
                                mask,  # add mask
                                np.array(item['bbox'], dtype=np.float32),  # add bbox
                                item['score'],  # add confidence score
                                item['category_id'],  # add category id
                                item['category_name']  # add category name
                            ))  # collect detection tuple

                    # extract values safely
                    if valid_pairs:  # check if we have valid pairs
                        # unpack valid_pairs - ensure we have exactly 5 values per item
                        masks = []  # initialize masks list
                        bboxes = []  # initialize bboxes list
                        scores = []  # initialize scores list
                        category_ids = []  # initialize category ids list
                        category_names = []  # initialize category names list

                        for pair in valid_pairs:  # process each valid pair
                            if len(pair) == 5:  # make sure we have exactly 5 elements
                                masks.append(pair[0])  # add mask
                                bboxes.append(pair[1])  # add bbox
                                scores.append(pair[2])  # add score
                                category_ids.append(pair[3])  # add category id
                                category_names.append(pair[4])  # add category name
                    else:  # no valid pairs found
                        masks = []  # empty masks list
                        bboxes = []  # empty bboxes list
                        scores = []  # empty scores list
                        category_ids = []  # empty category ids list
                        category_names = []  # empty category names list
                    result_dict = {
                        'image_path': image_paths[idx],  # store original image path
                        'category_id': np.array(category_ids, dtype=np.int64),  # store category ids as int64 array
                        'category_name': list(category_names),  # store category names as list
                        'bboxes': np.array([UtilsISAT.bbox_convert(bbox, 'COCO2ISAT') for bbox in bboxes], dtype=np.int32) if bboxes else np.array([], dtype=np.int32),  # convert and store bboxes
                        'masks': [UtilsISAT.coco_mask2isat_mask(mask[0]) for mask in masks if mask] if masks else []  # convert and store masks
                    }  # collect prediction metadata
                    valid_predictions.append(result_dict)  # append the bboxes of each image

        # Process focus-optimized results separately
        if if_auto_label and focus_optimized_results:
            # Process focus-optimized images only with the correct paths
            Anything2ISAT().from_openmmlab(valid_predictions=focus_optimized_results)

        if if_auto_label and not self.stack_input:
            # Only process non-stack images - will not run when processing stacks
            Anything2ISAT().from_openmmlab(valid_predictions=valid_predictions)

        if if_visualize and len(valid_predictions) > 0:
            for valid_prediction in valid_predictions:
                visualize_detections(valid_prediction)  # plot the detection results
        return valid_predictions

    def segment_cell(self,
                     json_paths: List[str],
                     if_visualize: bool = False,
                     if_auto_label: bool = True,
                     resize_to: tuple = (2048, 2048)) -> List[np.ndarray]:
        """
        Segment objects within the detected bboxes (padded)

        Args:
        - image_paths (List[str]): list of image paths
        - if_visualize (bool): if True, visualize the detection results
        - if_auto_label (bool): if True, convet predictions to ISAT json files as well

        Returns:
        - objects_list (List[dict]): a list of dictionaries containing segmentation results per image with keys 'category', 'group', 'segmentation', 'bbox', etc
        """
        def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
            """fill holes in binary mask"""
            if not np.any(binary_mask > 0):
                return binary_mask  # if blank
            labeled_mask = label(ndi.binary_fill_holes(binary_mask), connectivity=2)  # fill in labels and label regions
            label_counts = np.bincount(labeled_mask.ravel())  # count number of regions
            if len(label_counts) <= 1:
                return None
            largest_label = label_counts[1:].argmax() + 1  # find the largest label
            largest_mask = labeled_mask == largest_label  # find the largest mask
            return largest_mask

        self.segmentor.to(device)  # move segmentor to device ('cuda')

        for json_path in tqdm(json_paths, total=len(json_paths)):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            image_extension = os.path.splitext(data['info']['name'])[1]  # get the image path
            image_path = os.path.splitext(json_path)[0] + image_extension  # get the image path
            image = imread_rgb(image_path)  # load the image
            max_height, max_width = image.shape[:2]  # get the image dimensions
            objects_list, layer = [], 1.0  # to store objects for the json file
            for idx, obj in enumerate(data['objects']):
                if 'stoma' not in obj['category']:
                    continue
                x_min, y_min, x_max, y_max = obj['bbox']
                padded_bbox = UtilsISAT.pad_bbox(obj['bbox'], int(self.crop_padding_ratio * max((x_max - x_min), (y_max - y_min))), max_width, max_height)
                stoma_patch = UtilsISAT.crop_image_with_padding(image, padded_bbox, 0, allow_negative_crop=False)  # crop the image with padded bbox for bigger fild of view
                resized_patch, final_padding, padded_dimension, initial_padding_amount = resize_and_pad_image(stoma_patch, initial_padding_ratio=0, target_size=resize_to)  # resized and pad the image to target dimensions
                mmseg_utils_register_all_modules(init_default_scope=True)  # initialize mmmseg scope
                prediction = mmseg_apis_inference_model(self.segmentor, resized_patch)  # inference on the given image (batch inference is not supported)
                prediction = prediction.pred_sem_seg.data.cpu().numpy()[0]  # move prediction from GPU to CPU
                unique_classes = np.unique(prediction)  # fetch the unique classes
                resized_masks = []  # resizing the segmentation masks
                for cls in unique_classes:
                    class_mask = prediction == cls  # get the one-hot class mask
                    resized_boolean_mask = restore_original_dimensions(class_mask, final_padding, padded_dimension, initial_padding_amount)  # resize the bool mask
                    class_values_mask = resized_boolean_mask * cls  # ressign class code to the bool mask
                    resized_masks.append(class_values_mask)  # collect the resized mask
                prediction = np.max(np.stack(resized_masks, axis=0), axis=0)
                seg_mask = np.zeros([prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8)  # create a black image

                for idx_j in np.unique(prediction):
                    class_name = self.seg_onehot_mapping.get(idx_j, None)  # get the class name
                    color = self.seg_color_mapping.get(class_name, [0, 0, 0])  # pick the color other wise balck background
                    mask = prediction == idx_j  # pick the prediction class
                    for idx_k in range(3):
                        seg_mask[:, :, idx_k] = np.where(mask, color[idx_k], seg_mask[:, :, idx_k])  # fill in colors for the predicted class
                if if_visualize:
                    plt.imshow(seg_mask); plt.show()  # noqa: show the segmentation visualization
                for seg_class in ['stomatal complex', 'stoma', 'outer ledge', 'pore']:
                    if seg_class in self.seg_onehot_mapping.values():
                        x_1, y_1, x_2, y_2 = padded_bbox  # stoma location
                        seg_class_region = np.all(seg_mask == self.seg_color_mapping.get(seg_class), axis=-1)
                        if np.all(seg_class_region == 0):
                            continue
                        seg_class_region = fill_holes(np.all(seg_mask == self.seg_color_mapping.get(seg_class), axis=-1))  # get the class bool segmentation mask
                        full_mask = np.zeros(image.shape[:2], dtype=bool)  # create a empty full image bool mask
                        full_mask[y_1:y_2, x_1:x_2] = seg_class_region  # map the cropped path segmentation back to the entire image
                        if np.sum(full_mask) > 10:
                            obj = {
                                'category': seg_class,  # assign the segmentation class
                                'group': data['objects'][idx]['group'],  # same as the patch number
                                'segmentation': UtilsISAT.mask2segmentation(full_mask),  # convert bool mask to ISAT segmentations
                                'area': int(np.sum(full_mask)),
                                'layer': layer,  # the overlay layer
                                'bbox': UtilsISAT.mask2bbox(full_mask),  # compute the bbox
                                'iscrowd': False,
                                'note': 'Auto'}
                            objects_list.append(obj); layer += 1.0  # noqa
            if if_auto_label:
                Anything2ISAT.seg2isat(info_dict=data['info'], objects_list=objects_list, output_filename=json_path)
        return objects_list
