"""Module providing functions autolabeling images with OpenMMlab models"""

# pylint: disable=line-too-long, import-error, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args, wrong-import-position, undefined-loop-variable, unused-import
import os  # interact with the operating system
import sys; sys.path.append(os.path.abspath(os.path.join('.', 'Rein')))  # noqa: to add rein into path
from typing import List  # to support type hints
import warnings; warnings.filterwarnings('ignore', message='.*in an upcoming release, it will be required to pass the indexing argument.*'); warnings.filterwarnings('ignore', message='Failed to add*'); warnings.filterwarnings('ignore', message='xFormers is available'); warnings.filterwarnings('ignore', module=r'.*dino_layers.*')  # noqa: suppress torch.meshgrid warnings
import numpy as np  # NumPy
from PIL import Image  # Pillow image processing
import torch  # PyTorch
from tqdm import tqdm  # progress bar
from matplotlib import pyplot as plt  # for image visualization
import matplotlib.patches as patches
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from sahi.auto_model import AutoDetectionModel  # sahi wrapper for mmdetection
from sahi.predict import get_sliced_prediction  # sahi sliced prediction
import rein  # noqa: rein, parameter-efficient backbone finetuning
from mmengine.config import Config  # for mmsegmentation config
from mmseg.utils import register_all_modules as mmseg_utils_register_all_modules  # register mmseg modules
from mmseg.apis import init_model as mmseg_apis_init_model  # initialize mmseg model
from mmseg.apis import inference_model as mmseg_apis_inference_model  # mmseg inference segmentor
from mmseg.apis import show_result_pyplot as mmseg_apis_show_result_pyplot  # visualize segment results
from ..core.core import device, Cell_Colors, imread_rgb, resize_and_pad_image  # import core elements
from ..core.isat import UtilsISAT, Anything2ISAT  # to interact with ISAT jason files
from ..utils.data4training import Data4Training  # the traning processing pipelines


class OpenMMlab(Data4Training):
    """
    Automatic mask generation with mmdet (https://github.com/open-mmlab/mmdetection) and mmsegmentation (https://github.com/open-mmlab/mmsegmentation)
    """
    def __init__(self,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 detector_threshold: float = 0.2,
                 segmentor_config_path: str = None,
                 segmentor_weight_path: str = None,
                 seg_onehot_mapping: dict = {cell_color.class_encoding: cell_color.class_name for cell_color in Cell_Colors},
                 **kwargs):
        super().__init__(**kwargs)
        self.detector_config_path = detector_config_path  # object detection config path
        self.detector_weight_path = detector_weight_path  # object detection weight path
        self.detector_threshold = detector_threshold  # object detection threshold
        self.segmentor_config_path = segmentor_config_path  # semantic segmentation config path
        self.segmentor_weight_path = segmentor_weight_path  # semantic segmentation weight path
        self.seg_onehot_mapping = seg_onehot_mapping  # segmentation one-hot code against class_name
        self.seg_color_mapping = {cell_color.class_name: cell_color.mask_rgb for cell_color in Cell_Colors}  # mapp the segmentation class names to their colors

    def detect_objects(self,
                       image_paths: List[str],
                       if_resize_image: bool = True,
                       if_keep_ratio: bool = True,
                       if_visualize: bool = False,
                       if_auto_label: bool = False,
                       if_standard_pred: bool = False) -> List[np.ndarray]:
        """
        Detect objects in a list of images and return their bounding boxes and masks
        Each image is processed through an object detection model

        Args:
        - image_paths (List[str]): list of image paths
        - if_resize_image (bool): if True, resize and pad image to target dimension before predictions
        - if_keep_ratio (bool): if True, maintains aspect ratio while resizing
        - if_visualize (bool): if True, visualize the detection results
        - if_auto_label (bool): if True, convet predictions to ISAT json files as well

        Returns:
        - valid_predictions (List[dict]): a list of dictionaries containing detection results per image with keys 'image_path', 'category_id', 'category_name', 'bboxes', 'masks'
        """
        def visualize_detections(image: np.ndarray, valid_prediction: dict) -> None:
            """Visualizes detection results on an image using bounding boxes and optional masks

            Args:
            - image (np.ndarray): the image on which to overlay detections.
            - valid_prediction (dict): containing 'bboxes' [x_min, y_min, x_max, y_max] and optionally 'masks' [[x1, y1], [x2, y2], ..., [xn, yn]]

            Returns:
            - None. Just plot the detection results
            """
            _, ax = plt.subplots(1); ax.imshow(image)  # noqa: add the imag to plot
            if valid_prediction['masks'] is not None:
                for mask in valid_prediction['masks']:
                    polygon = plt.Polygon(mask, closed=True, fill=True, color='red', alpha=0.5)  # plot color and transparency
                    ax.add_patch(polygon)  # add masks if they exist

            for bbox in valid_prediction['bboxes']:
                box = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='blue', facecolor='none')  # box color
                ax.add_patch(box)  # always draw bounding boxes
            ax.set_title(f"{os.path.basename(valid_prediction['image_path'])}")  # add the image file name
            plt.axis('off'); plt.show()  # noqa: show the plot

        images = [imread_rgb(image_path) for image_path in image_paths]  # load all images

        if if_resize_image:
            resized_images, resizing_metadata = [], []  # to collected the resizing information
            for image in images:
                metadata = {}  # to collect the resizing metadata for each image
                image_pil = Image.fromarray(image)  # np.array to pil image
                original_width, original_height = image_pil.size  # the original width and height

                if if_keep_ratio:
                    ratio = min(self.new_width / original_width, self.new_height / original_height)  # get the closest ratio to destination dimension
                    resize_width, resize_height = int(original_width * ratio), int(original_height * ratio)  # resize width and height according to the ratio
                else:
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
        if not self.use_sahi:
            mmdet_utils_register_all_modules(init_default_scope=False)  # initialize mmdet scope
            detector = mmdet_apis_init_detector(self.detector_config_path, self.detector_weight_path, device=device)  # initialize a detector from config file
            # print(detector.cfg)
            # print(detector)
            category_names = detector.dataset_meta["classes"]  # get the category names
            category_mapping = {str(inx): category_name for inx, category_name in enumerate(category_names)}  # get the catergory mapping
            for idx, image in tqdm(enumerate(images), total=len(images)):
                prediction = mmdet_apis_inference_detector(detector, image)  # inference image(s) with the detector
                valid_indices = torch.where(prediction.pred_instances.scores > self.detector_threshold)[0]   # filter based on the detection threshold
                if valid_indices.numel() == 0:
                    result_dict = {'category_id': None, 'category_name': None, 'bboxes': None, 'masks': None}  # if no detections exceed the threshold, store None for this image
                else:
                    category_id = prediction.pred_instances.labels[valid_indices].cpu().numpy()
                    masks = prediction.pred_instances.masks[valid_indices].cpu().numpy() if hasattr(prediction.pred_instances, 'masks') else None
                    result_dict = {
                        'image_path': image_paths[idx],
                        'category_id': category_id,
                        'category_name': [category_mapping.get(str(category_id), 'Unknown') for category_id in category_id],
                        'bboxes': np.array(prediction.pred_instances.bboxes[valid_indices].cpu().numpy(), dtype=np.int32),
                        'masks': [UtilsISAT.mask2segmentation(mask) for mask in masks] if masks is not None else None
                    }  # collect prediction metadata
                valid_predictions.append(result_dict)

        elif self.use_sahi:
            detector = AutoDetectionModel.from_pretrained(
                model_type='mmdet',  # name of the detection framework
                model_path=self.detector_weight_path,  # path of the detection model (ex. 'model.pt')
                config_path=self.detector_config_path,  # path of the config file (ex. 'mmdet/configs/cascade_rcnn_r50_fpn_1x.py')
                confidence_threshold=self.detector_threshold,  # all predictions with score < confidence_threshold will be discarded
                image_size=self.slice_width,  # inference input size
                device=device  # device, "cpu" or "cuda:0"
            )
            for idx, image in tqdm(enumerate(images), total=len(images)):
                result = get_sliced_prediction(
                    image=image,  # location of image or numpy image matrix to slice
                    detection_model=detector,  # model.DetectionModel
                    slice_width=self.slice_width,  # width of each slice
                    slice_height=self.slice_height,  # height of each slice
                    overlap_height_ratio=self.sahi_overlap_ratio,  # fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels)
                    overlap_width_ratio=self.sahi_overlap_ratio,  # fractional overlap in width of each window
                    perform_standard_pred=if_standard_pred,  # perform a standard prediction on top of sliced predictions to increase large object detection accuracy
                    postprocess_type='GREEDYNMM',  # type of the postprocess to be used after sliced inference while merging/eliminating predictions. Options are 'NMM', 'GREEDYNMM' or 'NMS'. Default is 'GREEDYNMM'
                    postprocess_match_metric='IOU',  # metric to be used during object prediction matching after sliced prediction. 'IOU' for intersection over union, 'IOS' for intersection over smaller area.
                    postprocess_match_threshold=0.1,  # sliced predictions having higher iou than postprocess_match_threshold will be postprocessed after sliced prediction.
                    postprocess_class_agnostic=False,  # if True, postprocess will ignore category ids.
                    verbose=0,  # 0: no print; 1: print number of slices (default); 2: print number of slices and slice/prediction durations
                    merge_buffer_length=None,  # the length of buffer for slices to be used during sliced prediction, which is suitable for low memory
                    auto_slice_resolution=True,  # if slice parameters (slice_height, slice_width) are not given, it enables automatically calculate these params from image resolution and orientation
                    slice_export_prefix=None,  # prefix for the exported slices. Defaults to None
                    slice_dir=None,  # directory to save the slices. Defaults to None
                ).to_coco_annotations()  # get the prediction results in MSCOCO format
                masks = [item['segmentation'] for item in result]  # get the segmentation masks in MSCOCO format
                result_dict = {
                    'image_path': image_paths[idx],
                    'category_id': np.array([item['category_id'] for item in result], dtype=np.int64),
                    'category_name': [item['category_name'] for item in result],
                    'bboxes': np.array([UtilsISAT.bbox_convert(item['bbox'], 'COCO2ISAT') for item in result], dtype=np.int32),
                    'masks': [UtilsISAT.coco_mask2isat_mask(mask[0]) for mask in masks] if len(masks[0]) > 0 else None
                }  # collect prediction metadata
                valid_predictions.append(result_dict)  # append the bboxes of each image

        if if_resize_image:
            for idx, prediction in enumerate(valid_predictions):
                metadata = resizing_metadata[idx]  # get the resizing information for the given image
                adjusted_bboxes, adjusted_masks = [], []  # to collect the adjusted bboxes and masks back to orginal image
                for bbox in prediction['bboxes']:
                    x1, y1, x2, y2 = bbox  # load each individual object bbox
                    adj_x1 = int((x1 - metadata['padding_horizontal']) / metadata['width_ratio'])  # adjust x1
                    adj_y1 = int((y1 - metadata['padding_vertical']) / metadata['height_ratio'])  # adjust y1
                    adj_x2 = int((x2 - metadata['padding_horizontal']) / metadata['width_ratio'])  # adjust x2
                    adj_y2 = int((y2 - metadata['padding_vertical']) / metadata['height_ratio'])  # adjust y2
                    adjusted_bboxes.append([adj_x1, adj_y1, adj_x2, adj_y2])  # append the update bbox

                for mask in prediction['masks']:
                    if mask is not None:
                        updated_points = []  # to collect all points
                        for point in mask:
                            point[0] = (point[0] - padding_horizontal) / width_ratio   # update x coordinate
                            point[1] = (point[1] - padding_vertical) / height_ratio   # update y coordinate
                            updated_points.append([point[0], point[1]])  # collect the updated points
                        adjusted_masks.append(updated_points)  # append the update mask
                valid_predictions[idx]['bboxes'] = np.array(adjusted_bboxes, dtype=np.int32)  # update the all bboxes for the given image
                valid_predictions[idx]['masks'] = adjusted_masks if len(adjusted_masks) > 0 else None  # update the all masks for the given image

            images = [metadata['original_image'] for metadata in resizing_metadata]  # restore original image

        if if_auto_label:
            Anything2ISAT().from_openmmlab(valid_predictions=valid_predictions)  # convert detection results to ISAT json files

        if if_visualize:
            for idx, image in enumerate(images):
                visualize_detections(image, valid_predictions[idx])  # plot the detection results
        return valid_predictions

    def rein_segmentor(self):
        """Load rein segmentor"""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        mmseg_utils_register_all_modules(init_default_scope=False)  # initialize mmmseg scope
        config = Config.fromfile(self.segmentor_config_path)  # load the segmentor config
        network: torch.nn.Module = mmseg_apis_init_model(config, self.segmentor_weight_path, device)  # load the segmentor weights
        network.cfg = config  # remap the config file to the neural network
        torch.set_grad_enabled(False)  # eval model
        return network

    def stomata_segmentor(self,
                          image_paths: List[str],
                          if_resize_image: bool = True,
                          if_keep_ratio: bool = True,
                          if_visualize: bool = False,
                          if_auto_label: bool = False,
                          if_standard_pred: bool = False) -> List[np.ndarray]:
        """
        Detect objects in a list of images and return their bounding boxes and masks
        Each image is processed through an object detection model

        Args:
        - valid_predictions (List[dict]): a list of dictionaries containing detection results per image with keys 'image_path', 'category_id', 'category_name', 'bboxes', 'masks'
        - if_resize_image (bool): if True, resize and pad image to target dimension before predictions
        - if_keep_ratio (bool): if True, maintains aspect ratio while resizing
        - if_visualize (bool): if True, visualize the detection results
        - if_auto_label (bool): if True, convet predictions to ISAT json files as well

        Returns:
        - List[dict]: a list of dictionaries containing detection results per image with keys 'image_path', 'category_id', 'category_name', 'bboxes', 'masks'
        """
        # valid_predictions = self.detect_objects(image_paths, if_visualize=False, if_auto_label=False)
        torch.backends.cudnn.enabled = False
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        segmentor = self.rein_segmentor()  # load the segmentor
        # for valid_prediction in valid_predictions:
        for image_path in image_paths:
            # image_path = valid_prediction['image_path']  # get the image path
            image = imread_rgb(image_path)  # load the image
            # bboxes = valid_prediction['bboxes']  # get the predicted bboxes
            # stoma_patches = [np.array(Image.fromarray(image).crop(bbox)) for bbox in bboxes]  # crop the detected patches
            # for stoma_patch in stoma_patches:
                # resized_patch, padding, scale = resize_and_pad_image(stoma_patch, target_size=(512, 512))  # resized and pad the image to target dimensions
                # result = mmseg_apis_inference_model(segmentor, resized_patch)  # inference on the given image
                # vis_img = mmseg_apis_show_result_pyplot(segmentor, resized_patch, result)  # visualize the segmentation results
                # plt.imshow(vis_img)
                # plt.axis('off')
            result = mmseg_apis_inference_model(segmentor, image)  # inference on the given image
            vis_img = mmseg_apis_show_result_pyplot(segmentor, image, result)  # visualize the segmentation results
            plt.imshow(vis_img)
            plt.axis('off')
            plt.show()
        return result.pred_sem_seg.data.cpu().numpy()
