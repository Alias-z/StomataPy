"""Module providing functions inference stomata images"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, relative-beyond-top-level, wildcard-import, no-name-in-module
import os  # interact with the operating system
import time  # record time
import copy  # for deepcopy
import json  # manipulate json files
import cv2  # OpenCV
from PIL import Image, ImageOps  # Pillow image processing
from scipy import ndimage as ndi  # for hole filling
from skimage.measure import label, regionprops  # for using instance segmentation results as detected objects
import numpy as np  # NumPy
import torch  # PyTorch
from tqdm import tqdm  # progress bar
import pandas as pd  # for Excel sheet
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from mmseg.utils import register_all_modules as mmseg_utils_register_all_modules  # register mmseg modules
from mmseg.apis import init_model as mmseg_apis_init_model  # initialize mmseg model
from mmseg.apis import inference_model as mmseg_apis_inference_model  # mmseg inference segmentor
from ..core.core import device, image_types, Cell_Colors, imread_rgb, color_select, binary, lab_logo  # import core functions
from ..core.stoma_dimension import GetDiameter  # import core functions for stomatal aperture
from ..core.isat import UtilsISAT  # functions to manipulate ISAT segmentations


class StomataSeeker:
    """Inference stomata images"""
    def __init__(self,
                 input_dir: str,
                 output_name: str = 'Results aperture',
                 batch_size: int = 20,
                 pixels_per_micrometer: float = 8.0,
                 concatenate_excels: bool = True,
                 ensemble_detectors: bool = True,
                 object_detector_config_path: str = None,
                 object_detector_weight_path: str = None,
                 object_detector_threshold: float = 0.2,
                 instance_detector_config_path: str = None,
                 instance_detector_weight_path: str = None,
                 instance_detector_threshold: float = 0.6,
                 segmentor_config_path: str = None,
                 segmentor_weight_path: str = None,
                 seg_onehot_mapping: dict = {cell_color.class_encoding: cell_color.class_name for cell_color in Cell_Colors},
                 padding: int = 20,
                 empty_dataframe: pd.DataFrame = pd.DataFrame(columns=['image name', 'patch name', 'image height', 'image width', 'scale (pixels/\u03BCm)', 'stomata lenghth (\u03BCm)', 'stomata width (\u03BCm)',
                                                                       'outer ledge lenghth (\u03BCm)', 'outer ledge width (\u03BCm)', 'stomata area (\u03BCm\N{SUPERSCRIPT TWO})', 'outer ledge area (\u03BCm\N{SUPERSCRIPT TWO})',
                                                                       'pore area (\u03BCm\N{SUPERSCRIPT TWO})', 'operating aperture (%)', 'stomata orientation (°)', 'image area (\u03BCm\N{SUPERSCRIPT TWO})', 'number of stomata']),
                 ):
        self.input_dir = os.path.normpath(input_dir)  # input directory
        self.output_name = output_name  # output folder name
        self.batch_size = batch_size  # inference batch size
        self.pixels_per_micrometer = pixels_per_micrometer  # number of pixels per micrometer
        self.concatenate_excels = concatenate_excels  # concatenate Excel sheets from all subfolders
        self.ensemble_detectors = ensemble_detectors  # ensemble object detection and instance segmentation for stomata detection
        self.object_detector_config_path = object_detector_config_path  # object detection config path
        self.object_detector_weight_path = object_detector_weight_path  # object detection weight path
        self.object_detector_threshold = object_detector_threshold  # object detection threshold
        self.instance_detector_config_path = instance_detector_config_path  # instance segmentation config path
        self.instance_detector_weight_path = instance_detector_weight_path  # instance segmentation weight path
        self.instance_detector_threshold = instance_detector_threshold  # instance segmentation threshold
        self.segmentor_config_path = segmentor_config_path  # semantic segmentation config path
        self.segmentor_weight_path = segmentor_weight_path  # semantic segmentation weight path
        self.seg_onehot_mapping = seg_onehot_mapping  # segmentation one-hot code against class_name
        self.seg_color_mapping = {cell_color.class_name: cell_color.mask_rgb for cell_color in Cell_Colors}  # mapp the segmentation class names to their colors
        self.padding = padding  # padding for the bboxes
        self.empty_dataframe = empty_dataframe  # template dataframe

        if self.ensemble_detectors:
            self.batch_size //= 2  # reduce batch size to half for ensemble detectors
            print(f'\n \033[34m Note: ensemble_detectors=True, batch_size reduces to half ({self.batch_size}) \n')

    def get_images(self, subfolder_path: str) -> tuple:
        """Load images in a subfolder_path into a list, and get the properties of each image"""
        images = []  # to store values
        file_names = sorted(os.listdir(subfolder_path), key=str.casefold)  # sort file names
        file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        print('\n \x1b[31m 1. loading all input images')
        for name in tqdm(file_names, total=len(file_names)):
            image = imread_rgb(os.path.join(subfolder_path, name))  # load the image in RGB scale
            images.append(image)  # collect image arrays and dimensions
        return file_names, images

    def batch_images(self, subfolder_path: str) -> list:
        """Split images into batches to avoid CUDA out of memory."""
        output_dir = os.path.join(self.output_name, *os.path.normpath(subfolder_path).split(os.sep)[1:])  # create output dir
        os.makedirs(output_dir, exist_ok=True)  # create the output folder
        file_names, images = self.get_images(subfolder_path)  # get the file names, images and scales
        batches = []  # to store information of each batch
        for i in range(0, len(file_names), self.batch_size):
            file_names_batches = file_names[i: i + self.batch_size]  # split the names into batches
            images_batches = images[i: i + self.batch_size]  # split the images accordingly
            batches.append([subfolder_path, output_dir, file_names_batches, images_batches])  # summary
        return batches

    def detect_stomata(self, images: list) -> tuple:
        """Detect stomta"""
        mmdet_utils_register_all_modules(init_default_scope=False)  # initialize mmdet scope
        object_detector = mmdet_apis_init_detector(self.object_detector_config_path, self.object_detector_weight_path, device=device)  # initialize a object detector from config file
        object_predictions = mmdet_apis_inference_detector(object_detector, images)  # inference image(s) with the object detector
        object_indices = [torch.where(object_predictions[i].pred_instances.scores > self.object_detector_threshold)[0] for i in range(len(object_predictions))]  # get indices of detected objects
        object_bboxes = [object_predictions[i].pred_instances.bboxes[object_indices[i]].cpu().numpy() for i in range(len(object_predictions))]  # gte the bounding boxes of detected objects
        object_bboxes = [np.array(bboxes, dtype=np.int32) for bboxes in object_bboxes]  # convert the bboxes to int32

        if self.ensemble_detectors:
            instance_detector = mmdet_apis_init_detector(self.instance_detector_config_path, self.instance_detector_weight_path, device=device)  # initialize a instance detector from config file
            instance_predictions = mmdet_apis_inference_detector(instance_detector, images)  # inference image(s) with the instance detector.
            instance_indices = [torch.where(instance_predictions[i].pred_instances.scores > self.instance_detector_threshold)[0] for i in range(len(instance_predictions))]  # get indices of detected instances
            instance_masks = [instance_predictions[i].pred_instances.masks[instance_indices[i]].cpu().numpy() for i in range(len(instance_predictions))]  # gte the masks of detected objects
            instance_masks = [np.any(masks, axis=0) for masks in instance_masks]  # combine all boolean masks of detected instances into one
            instance_masks = [regionprops(label(instance_mask.astype(np.uint8))) for instance_mask in instance_masks]  # get the regionprops of detected instances
            instance_bboxes, instance_bool_masks = [], []  # to store the bounding boxes and bool masks of detected instances
            for idx, image in enumerate(images):
                bool_mask = np.zeros(image.shape[:2], dtype=bool)  # create a boolean mask for the image
                a_image = image.shape[0] * image.shape[1]  # get the area of the image
                filtered_instance_mask = [instance for instance in instance_masks[idx] if a_image * 0.001 < instance.area < a_image * 0.02]  # filter instances based on their area
                instance_bbox = [(int(instance.bbox[1]), int(instance.bbox[0]), int(instance.bbox[3]), int(instance.bbox[2])) for instance in filtered_instance_mask]  # bboxes of all stomata instances in one image
                instance_bboxes.append(np.array(instance_bbox, dtype=np.int32))
                for instance in filtered_instance_mask:
                    mask = np.zeros(image.shape[:2], dtype=bool)  # convert regionprops back to boolean mask
                    mask[instance.coords[:, 0], instance.coords[:, 1]] = True  # fill in true values in the given instance region
                    bool_mask[mask] = True  # fill in true values in all instance regions using loops
                instance_bool_masks.append(bool_mask)  # store the bool masks for the images

            def remove_inner_bboxes_area(bboxes: list) -> list:
                """Remove smaller bounding boxes that overlap with a larger one by more than 40%"""
                bboxes = np.array(bboxes)  # list to np.array
                to_remove = set()  # initialize the set of bboxes to be removed
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])  # calculate area for all bounding boxes
                for idx, _ in enumerate(bboxes):
                    for idx_j in range(idx + 1, len(bboxes)):
                        if idx in to_remove or idx_j in to_remove:
                            continue
                        bbox1 = bboxes[idx]; bbox2 = bboxes[idx_j]  # noqa: get bbox1 and bbox2
                        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
                        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                        intersection = x_overlap * y_overlap  # calculate intersection area
                        area1, area2 = areas[idx], areas[idx_j]  # get areas of both bounding boxes
                        larger_area = max(area1, area2)  # calculate the larger area among the two bboxes
                        if intersection > 0.4 * larger_area:
                            smaller_bbox_idx = idx if area1 < area2 else idx_j  # remove the smaller bbox
                            to_remove.add(smaller_bbox_idx)
                return np.delete(bboxes, list(to_remove), axis=0).tolist()

            stomata_bboxes = []  # to store ensembled the bboxes
            for idx, object_bbox in enumerate(object_bboxes):
                bboxes = object_bbox.tolist() + instance_bboxes[idx].tolist()  # ensemble the bboxes
                stomata_bboxes.append(remove_inner_bboxes_area(bboxes))
        else:
            stomata_bboxes = object_bboxes  # only use object detection bboxes

        detected_indices = [idx for idx, bboxes in enumerate(stomata_bboxes) if len(bboxes) > 0]  # get the idex of images that are detectable
        undetected_indices = [idx for idx, bboxes in enumerate(stomata_bboxes) if len(bboxes) == 0]  # get the idex of images that are not detectable

        for idx, image in enumerate(images):
            if idx in detected_indices:  # if the image is detectable
                width, height = Image.fromarray(image).size  # get image dimensions
                for bbox_idx, (x_1, y_1, x_2, y_2) in enumerate(stomata_bboxes[idx]):
                    x_1 = max(0, x_1 - self.padding)  # padded the bbox's left side
                    y_1 = max(0, y_1 - self.padding)  # padded the bbox's top side
                    x_2 = min(width, x_2 + self.padding)  # padded the bbox's right side
                    y_2 = min(height, y_2 + self.padding)  # padded the bbox's bottom side
                    stomata_bboxes[idx][bbox_idx] = (x_1, y_1, x_2, y_2)  # update the bbox's coordinates

        stomata_patches, stomata_locations = [], []  # list of patches and locations
        for idx, image in tqdm(enumerate(images), total=len(images)):
            if idx in detected_indices:  # if the image is detectable
                image = Image.fromarray(image)  # convert to PIL image
                stomata_patches.append([image.crop((x_1, y_1, x_2, y_2)) for x_1, y_1, x_2, y_2 in stomata_bboxes[idx]])  # crop the bboxes of each stomata len(...) = len(images)
                stomata_locations.append([(x_1, y_1, x_2, y_2) for x_1, y_1, x_2, y_2 in stomata_bboxes[idx]])   # get the coordinates of the each bbox len(...) = len(images)
        return stomata_patches, stomata_locations, detected_indices, undetected_indices

    def pad_image(self, image: Image, target_size: tuple = (440, 440)) -> tuple:
        """Pad an image to the target size as in sementic segmentation training pipeline"""
        width, height = image.size  # get image dimensions
        delta_w = target_size[0] - width  # get the delta width
        delta_h = target_size[1] - height  # get the delta height
        left_pad = delta_w // 2  # get the left padding
        right_pad = delta_w - left_pad  # get the right padding
        top_pad = delta_h // 2  # get the top padding
        bottom_pad = delta_h - top_pad  # get the bottom padding
        padding = (left_pad, top_pad, right_pad, bottom_pad)  # get the padding
        return ImageOps.expand(image, padding), padding

    def remove_padding(self, image: Image, padding) -> Image:
        """Remove padding from an image"""
        left_pad, top_pad, right_pad, bottom_pad = padding
        return image.crop((left_pad, top_pad, image.width - right_pad, image.height - bottom_pad))

    def segment_stomata(self, images: list) -> list:
        """Sgement a single stoma to get its aperture"""
        segmentor_results = []  # to collect segmentation results (filled with prediction colors)
        mmseg_utils_register_all_modules(init_default_scope=False)  # initialize mmmseg scope
        segmentor = mmseg_apis_init_model(self.segmentor_config_path, self.segmentor_weight_path, device=device)   # initialize a segmentor from config file
        for image in images:
            prediction = mmseg_apis_inference_model(segmentor, image)  # inference image with the segmentor (batch inference is not supported)
            prediction = prediction.pred_sem_seg.data.cpu().numpy()[0]  # move prediction from GPU to CPU
            empty = np.zeros([prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8)  # create a black image
            for idx in np.unique(prediction):
                class_name = self.seg_onehot_mapping.get(idx, None)  # get the class name
                color = self.seg_color_mapping.get(class_name, [0, 0, 0])  # pick the color other wise balck background
                mask = prediction == idx  # pick the prediction class
                for idx_j in range(3):
                    empty[:, :, idx_j] = np.where(mask, color[idx_j], empty[:, :, idx_j])  # fill in colors for the predicted class
            segmentor_results.append(empty)  # collect the images that has been filled with prediction colors
        return segmentor_results

    @staticmethod
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

    def seg_postprocess(self, seg_masks: list, label_mode: bool = False) -> tuple:
        """Refine stomata segmentation results based on knwon facts"""
        refined_seg_masks, contour_masks, contour_areas = [], [], []  # to collect refined segmentation results (filled with prediction colors)
        for seg_mask in tqdm(seg_masks, total=len(seg_masks)):
            refined_seg_mask = np.zeros((*seg_mask.shape[:2], 3), dtype=np.uint8)  # create a black image
            stoma_region = self.fill_holes(np.all(seg_mask == self.seg_color_mapping.get('stoma'), axis=-1))  # where the stoma region is
            guard_cell_region = self.fill_holes(np.all(seg_mask == self.seg_color_mapping.get('guard cell'), axis=-1))  # select the guard cell region
            refined_seg_mask[stoma_region] = self.seg_color_mapping.get('stoma')  # fill in the stoma region
            refined_seg_mask[guard_cell_region] = self.seg_color_mapping.get('stoma')  # combine the guard cell region to stoma (as inclusive)
            stoma_contour, _ = cv2.findContours(binary(refined_seg_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # get the contours of the stoma region

            refined_seg_mask[guard_cell_region] = self.seg_color_mapping.get('guard cell')  # fill in the guard cell region
            # guard_cell_contour, _ = cv2.findContours(binary(color_select(refined_seg_mask, refined_seg_mask, self.seg_color_mapping.get('guard cell'))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # contours of the guard cell

            if np.any(guard_cell_region > 0):
                outer_ledge_region = color_select(seg_mask, refined_seg_mask, self.seg_color_mapping.get('guard cell'))  # only within guard cell region
            else:
                outer_ledge_region = color_select(seg_mask, refined_seg_mask, self.seg_color_mapping.get('stoma'))  # only within stoma region if no guard cell class
            outer_ledge_region = self.fill_holes(binary(color_select(outer_ledge_region, outer_ledge_region, self.seg_color_mapping.get('outer ledge'))))  # select the outer ledge region
            refined_seg_mask[outer_ledge_region] = self.seg_color_mapping.get('outer ledge')  # fill in the outer ledge region
            outer_ledge_contour, _ = cv2.findContours(binary(color_select(refined_seg_mask, refined_seg_mask, self.seg_color_mapping.get('outer ledge'))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # contours of the outer ledge

            pore_region = np.all(seg_mask == self.seg_color_mapping.get('pore'), axis=-1)  # where the pore region is
            pore_region = color_select(pore_region, refined_seg_mask, self.seg_color_mapping.get('outer ledge'))  # only within outer ledge region
            refined_seg_mask[pore_region] = self.seg_color_mapping.get('pore')  # fill in the pore region
            refined_seg_masks.append(refined_seg_mask)

            outer_ledge_contour_mask = np.zeros((*seg_mask.shape[:2], 3), dtype=np.uint8)  # create a black image
            cv2.drawContours(outer_ledge_contour_mask, outer_ledge_contour, -1, (255, 255, 255), 1)  # draw the outer ledge contour
            stoma_contour_mask = copy.deepcopy(outer_ledge_contour_mask)  # make a copy of the outer ledge contour
            if label_mode:
                stoma_contour_mask = np.zeros((*seg_mask.shape[:2], 3), dtype=np.uint8)  # does not need outer ledge on stoma
            cv2.drawContours(stoma_contour_mask, stoma_contour, -1, (255, 255, 255), 1)  # draw the stomata contour
            contour_masks.append([stoma_contour_mask, outer_ledge_contour_mask])
            if len(stoma_contour) > 0 and len(outer_ledge_contour) > 0:
                contour_areas.append([cv2.contourArea(stoma_contour[0]), cv2.contourArea(outer_ledge_contour[0])])
            else:
                contour_areas.append([0, 0])
        return refined_seg_masks, contour_masks, contour_areas

    def get_aperture(self, seg_mask: np.ndarray) -> list:
        """Calculate stomatal aperture"""
        a_pore = len(np.where(np.all(seg_mask == self.seg_color_mapping.get('pore'), axis=-1))[0])  # pore area
        a_outer_ledge = len(np.where(np.all(seg_mask == self.seg_color_mapping.get('outer ledge'), axis=-1))[0]) + a_pore  # outer ledge area
        a_stoma = len(np.where(np.all(seg_mask == self.seg_color_mapping.get('stomata'), axis=-1))[0]) + a_outer_ledge  # stoma area
        stoma_pixel_ratio = a_stoma / seg_mask.shape[0] / seg_mask.shape[1] * 100  # as a quality check metric
        a_pore_absolute = a_pore * (1 / self.pixels_per_micrometer) ** 2  # pore area in square micrometer
        a_outer_ledge_absolute = a_outer_ledge * (1 / self.pixels_per_micrometer) ** 2  # outer ledge area in square micrometer
        a_stoma_absolute = a_stoma * (1 / self.pixels_per_micrometer) ** 2  # stoma area in square micrometer
        return [a_pore_absolute, a_outer_ledge_absolute, a_stoma_absolute, stoma_pixel_ratio]

    def text4name(self, patch_stitched: np.ndarray, patch_name: str) -> np.ndarray:
        """"Create a text region for the stiched image"""
        text_image = np.zeros((80, patch_stitched.shape[1], 3), np.uint8)  # create a black image with the same width as the concatenated image for writing text
        font_scale, font_color, line_type, font = 1, (255, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX  # noqa: define the font and line type
        (text_width, text_height) = cv2.getTextSize(patch_name, font, font_scale, line_type)[0]
        text_x = (text_image.shape[1] - text_width) // 2  # text x coordinate
        text_y = (text_image.shape[0] + text_height) // 2  # text y coordinate
        cv2.putText(text_image, patch_name, (text_x, text_y), font, font_scale, font_color, line_type)  # write the text to the image
        return cv2.vconcat([text_image, patch_stitched])

    def is_float(self, string: str) -> bool:
        """Check if a string can be converted to float"""
        try:
            float(string)
            return True
        except ValueError:
            return False

    def if_seg_on_edges(self, seg_mask: np.ndarray, edge_width: int = 3) -> bool:
        """Check if the segmentation mask has been cut off by any edges"""
        top_edge = seg_mask[:edge_width, :]  # pixels on the top edge
        bottom_edge = seg_mask[-edge_width:, :]  # bottom edge
        left_edge = seg_mask[:, :edge_width]  # left edge
        right_edge = seg_mask[:, -edge_width:]  # right edge
        edges = [top_edge, bottom_edge, left_edge, right_edge]  # all four edges
        return any(np.any(edge != [0, 0, 0]) for edge in edges)

    def get_stomata(self, batch: list) -> tuple:
        """The overall function that automaticly detect and segment stomata for a image batch"""
        subfolder_path, output_dir, file_names_batch, images_batch = batch  # load parameters for the batch
        with open(os.path.join(subfolder_path, 'scale.txt'), 'r', encoding='utf-8') as file:
            scale = file.readline().strip()  # stripping potential whitespace
            scale = float(scale) if self.is_float(scale) else None  # checking if the scale is a float or not
        if scale is not None:
            self.pixels_per_micrometer = scale  # replace default value with the scale

        print('\n \x1b[31m 2. detecting stomata \n')
        stomata_patches, stomata_locations, detected_indices, undetected_indices = self.detect_stomata(images_batch)  # get the detected stomata patches and locations
        bad_images_dir = os.path.join(output_dir, 'bad_images'); os.makedirs(bad_images_dir, exist_ok=True)  # noqa: create a folder containing undetected images (bad)
        for idx in undetected_indices:
            cv2.imwrite(os.path.join(bad_images_dir, file_names_batch[idx]), cv2.cvtColor(images_batch[idx], cv2.COLOR_RGB2BGR))  # export the bad images

        if len(detected_indices) > 0:
            file_names_batch = [file_names_batch[idx] for idx in detected_indices]  # select files that stomata are detected
            images_batch = [images_batch[idx] for idx in detected_indices]  # select images that stomata are detected

            print('\n \x1b[31m 3. segmenting stomata \n')
            padded_stomata_patches = [self.pad_image(patch) for stomata_patch in stomata_patches for patch in stomata_patch]  # add padding (PIL.Image)
            flattened_patches, paddings = zip(*padded_stomata_patches)  # get the PIL.Image and its padding values
            flattened_patches = [np.array(patch) for patch in flattened_patches]  # PIL.Image to np.array
            seg_masks = self.segment_stomata(flattened_patches)  # segment the open stomata patches
            seg_masks = [Image.fromarray(seg_mask.astype(np.uint8)) for seg_mask in seg_masks]  # np.array to PIL.Image
            seg_masks = [self.remove_padding(seg_mask, pad) for seg_mask, pad in zip(seg_masks, paddings)]  # remove the padding
            seg_masks = [np.array(seg_mask) for seg_mask in seg_masks]  # PILImage to np.array
            seg_masks, contour_masks, contour_areas = self.seg_postprocess(seg_masks)  # refine the stomata segmentation results

            print('\n \x1b[31m 4. calculating stomata aperture \n')
            batch_results = pd.DataFrame(columns=['image name', 'patch name', 'image height', 'image width', 'scale (pixels/\u03BCm)', 'stomata lenghth (\u03BCm)', 'stomata width (\u03BCm)',
                                                  'outer ledge lenghth (\u03BCm)', 'outer ledge width (\u03BCm)', 'stomata area (\u03BCm\N{SUPERSCRIPT TWO})', 'outer ledge area (\u03BCm\N{SUPERSCRIPT TWO})',
                                                  'pore area (\u03BCm\N{SUPERSCRIPT TWO})', 'operating aperture (%)', 'stomata orientation (°)', 'image area (\u03BCm\N{SUPERSCRIPT TWO})'])
            mask_count, good_stomata_count, n_stomata_list = 0, 0, []  # counting the indices of seg_masks and good stomata patches
            patch_output_dir = os.path.join(output_dir, 'Patches'); os.makedirs(patch_output_dir, exist_ok=True)  # noqa: create path output directory
            for image_id, image in tqdm(enumerate(images_batch), total=len(images_batch)):
                for idx, location in enumerate(stomata_locations[image_id]):
                    x_1, y_1, x_2, y_2 = location  # stomata location
                    patch = np.array(stomata_patches[image_id][idx])  # get the stomata patch
                    seg_mask = seg_masks[mask_count]  # get its segmentation mask
                    patch_aperture = self.get_aperture(seg_mask)  # calculate its aperture
                    stoma_contour_area, _ = contour_areas[mask_count]  # stoma contour mask and outer ledge contour area (both are 0 if anyone of them is 0)
                    if stoma_contour_area > 0 and not self.if_seg_on_edges(seg_mask):
                        patch_name = f'Patch {idx + 1}'  # get the name of the good stomata patch
                        patch_contour_masks = contour_masks[mask_count]  # stoma contour mask and outer ledge contour mask
                        stoma_lenghth_pixel, stoma_width_pixel, stoma_trait, angle = GetDiameter(patch_contour_masks[0], shrink_ratio=1.2, line_thickness=2).pca()  # stoma lenghth and width
                        stoma_lenghth, stoma_width = stoma_lenghth_pixel * (1 / self.pixels_per_micrometer), stoma_width_pixel * (1 / self.pixels_per_micrometer)  # stoma lenghth and width in micrometer
                        try:
                            outer_ledge_lenghth_pixel, outer_ledge_width_pixel, outer_ledge_trait, _ = GetDiameter(patch_contour_masks[1], shrink_ratio=1, line_thickness=2).pca()  # outerledge lenghth and width
                            outer_ledge_lenghth, outer_ledge_width = outer_ledge_lenghth_pixel * (1 / self.pixels_per_micrometer), outer_ledge_width_pixel * (1 / self.pixels_per_micrometer)  # outer ledge lenghth and width in micrometer
                        except ValueError:
                            outer_ledge_lenghth, outer_ledge_width, outer_ledge_trait = 0, 0, np.zeros((*stoma_trait.shape[:2], 3), dtype=np.uint8)  # in case no outer ledge area
                        patch_stitched = np.hstack((patch, seg_mask, stoma_trait, outer_ledge_trait))  # visualize the stoma patch and its segmentation mask
                        patch_stitched = self.text4name(patch_stitched, os.path.splitext(os.path.basename(patch_name))[0])  # write the patch name on the stitched result
                        cv2.imwrite(os.path.join(patch_output_dir, f'{os.path.splitext(file_names_batch[image_id])[0]} {patch_name}{os.path.splitext(file_names_batch[image_id])[1]}'), cv2.cvtColor(patch_stitched, cv2.COLOR_RGB2BGR))  # export the stitched result
                        result = {'image name': [file_names_batch[image_id]],
                                  'patch name': [patch_name],
                                  'image height': [image.shape[0]],
                                  'image width': [image.shape[1]],
                                  'scale (pixels/\u03BCm)': [self.pixels_per_micrometer],
                                  'stomata lenghth (\u03BCm)': [stoma_lenghth],
                                  'stomata width (\u03BCm)': [stoma_width],
                                  'outer ledge lenghth (\u03BCm)': [outer_ledge_lenghth],
                                  'outer ledge width (\u03BCm)': [outer_ledge_width],
                                  'stomata area (\u03BCm\N{SUPERSCRIPT TWO})': [patch_aperture[2]],
                                  'outer ledge area (\u03BCm\N{SUPERSCRIPT TWO})': [patch_aperture[1]],
                                  'pore area (\u03BCm\N{SUPERSCRIPT TWO})': [patch_aperture[0]],
                                  'operating aperture (%)': [patch_aperture[1] / patch_aperture[2] * 100],
                                  'stomata orientation (°)': [angle],
                                  'image area (\u03BCm\N{SUPERSCRIPT TWO})': [image.shape[0] * image.shape[1] * (1 / self.pixels_per_micrometer) ** 2]}
                        result = pd.DataFrame(data=result)  # collect result in a pd dataframe for exporting to an Excel sheet
                        batch_results = pd.concat([batch_results, result], axis=0)  # concatenate all results
                        n_stomata_list.append(len(stomata_locations[image_id]))
                        good_stomata_count += 1  # count number of good stomata patches
                        cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (0, 0, 255), 4)  # paint the good stomata bbox with green
                    else:
                        cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (255, 0, 0), 4)  # paint the blur stomata bbox with red
                    mask_count += 1  # go to the next mask
                cv2.imwrite(os.path.join(output_dir, file_names_batch[image_id]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # export the image
            batch_results['number of stomata'] = n_stomata_list  # record the number of detected stomata
            return len(file_names_batch), mask_count, good_stomata_count, batch_results
        else:
            print('\n \x1b[31m skip this batch since no stomata detected \n')
            batch_results = copy.deepcopy(self.empty_dataframe)
            return 0, 0, 0, batch_results

    def batch_predict(self) -> None:
        """Detect starch for all subfolders that contain 'scale.txt' file"""
        lab_logo()  # print lab logo
        start, images_num, stomata_num, good_stomata_num, folders = time.time(), 0, 0, 0, []  # initialize default values
        dataframes = []  # to collect the results from all subfolders
        folders = sorted([root for root, _, files in os.walk(self.input_dir) if 'scale.txt' in files], key=str.casefold)  # sort folders by name
        for folder in folders:
            print(f'\n \033[34m processing {folder} \n')
            results = copy.deepcopy(self.empty_dataframe)  # empty dataframe to collect the values
            batches = self.batch_images(folder)  # get subfolder image batches
            for idx, batch in enumerate(batches):
                print(f'\n \033[34m batch {idx + 1} out of {len(batches)} \n')
                n_files, mask_count, good_stomata_count, batch_results = self.get_stomata(batch)  # run starch detection pipline for each batch
                results = pd.concat([results, batch_results], axis=0)  # concatenate results from all batches
                images_num += n_files  # record the total number of images processed
                stomata_num += mask_count  # record the total number of stomata patches
                good_stomata_num += good_stomata_count  # record the number of good stomata patches
            if 'folder name' in results.columns:
                results.drop('folder name', axis=1, inplace=True)  # drop
            results.insert(0, 'folder name', os.path.basename(folder))  # replace the first column with the folder name
            output_dir = os.path.join(self.output_name, os.path.join(*folder.split(os.sep)[1:])); os.makedirs(output_dir, exist_ok=True)  # noqa: create the output folder
            results['stomata density (stomata mm\u207B\u00B2)'] = results['number of stomata'] / results['image area (\u03BCm\N{SUPERSCRIPT TWO})'] * 1e6  # calculate the stomata density
            results.to_excel(os.path.join(output_dir, 'stomata aperture.xlsx'), index=False)  # export results to Excel
            dataframes.append(results)  # for summarizing results
        if self.concatenate_excels and images_num != 0:
            print('\n \x1b[31m 5. concatenating all Excel sheets \n')
            dataframes = pd.concat(dataframes, axis=0)  # concatenate all the DataFrames
            dataframes.rename(columns={dataframes .columns[0]: 'folder name'}, inplace=True)  # rename the first column
            dataframes.to_excel(os.path.join(self.output_name, os.path.join(*self.input_dir.split(os.sep)[1:]), 'stomata aperture summary.xlsx'), index=False)  # export the summarized results to Excel
        end = time.time()  # stop the timer
        print('\n \033[34m Done! \n')
        print(f'\033[34m processed {images_num} images in {(end-start)/60} min')
        print(f'\033[34m detected {stomata_num } stomata and measured {good_stomata_num} stomata')
        if images_num == 0:
            print('\n \n \x1b[31m there is no image provided any use in this run; please check the following possibilities:')
            print('\n 1. check if your input directory is correct')
            print('\n 2. maybe you forgot to put the "scale.txt" file in your folder?')
            print('\n 3. is the image quality good enough?')
            print(f'\n 4. are image formats not supported? currently the program supports \n {image_types}')
        return None

    def auto_label(self, batch: list) -> None:
        """Labeling images automaticly for a image batch"""
        _, output_dir, file_names_batch, images_batch = batch  # load parameters for the batch

        print('\n \x1b[31m 2. detecting stomata \n')
        stomata_patches, stomata_locations, detected_indices, _ = self.detect_stomata(images_batch)  # get the detected stomata patches and locations

        if len(detected_indices) > 0:
            file_names_batch = [file_names_batch[idx] for idx in detected_indices]  # select files that stomata are detected
            images_batch = [images_batch[idx] for idx in detected_indices]  # select images that stomata are detected

            print('\n \x1b[31m 3. segmenting stomata \n')
            padded_stomata_patches = [self.pad_image(patch) for stomata_patch in stomata_patches for patch in stomata_patch]  # add padding (PIL.Image)
            try:
                flattened_patches, paddings = zip(*padded_stomata_patches)  # get the PIL.Image and its padding values
                flattened_patches = [np.array(patch) for patch in flattened_patches]  # PIL.Image to np.array
                seg_masks = self.segment_stomata(flattened_patches)  # segment the stomata patches
                seg_masks = [Image.fromarray(seg_mask.astype(np.uint8)) for seg_mask in seg_masks]  # np.array to PIL.Image
                seg_masks = [self.remove_padding(seg_mask, pad) for seg_mask, pad in zip(seg_masks, paddings)]  # remove the padding
                seg_masks = [np.array(seg_mask) for seg_mask in seg_masks]  # PILImage to np.array
                seg_masks, _, _ = self.seg_postprocess(seg_masks)  # refine the stomata segmentation results

                print('\n \x1b[31m 4. storing segmentation to json files \n')
                mask_count = 0  # counting the indices of seg_masks
                for image_id, image in tqdm(enumerate(images_batch), total=len(images_batch)):
                    info_dict = {
                        'description': 'ISAT',
                        'folder': output_dir,
                        'name': file_names_batch[image_id],
                        'width': image.shape[1],
                        'height': image.shape[0],
                        'depth': image.shape[2],
                        "note": ''}
                    json_path = os.path.join(output_dir, os.path.splitext(file_names_batch[image_id])[0] + '.json')  # the output json file name
                    objects_list, layer = [], 1.0  # to store objects for the json file
                    for idx, location in enumerate(stomata_locations[image_id]):
                        x_1, y_1, x_2, y_2 = location  # stoma location
                        seg_mask = seg_masks[mask_count]  # get its segmentation mask
                        for seg_class in set(['stoma', 'guard cell', 'outer ledge', 'pore']):
                            if seg_class in self.seg_onehot_mapping.values():
                                seg_class_region = self.fill_holes(np.all(seg_mask == self.seg_color_mapping.get(seg_class), axis=-1))  # get the class bool segmentation mask
                                full_mask = np.zeros(image.shape[:2], dtype=bool)  # create a empty full image bool mask
                                full_mask[y_1:y_2, x_1:x_2] = seg_class_region  # map the cropped path segmentation back to the entire image
                                if np.any(full_mask):
                                    obj = {
                                        'category': seg_class,  # assign the segmentation class
                                        'group': mask_count,  # same as the patch number
                                        'segmentation': UtilsISAT.mask2segmentation(full_mask),  # convert bool mask to ISAT segmentations
                                        'area': int(np.sum(full_mask)),
                                        'layer': layer,  # the overlay layer
                                        'bbox': UtilsISAT.mask2bbox(full_mask),  # compute the bbox
                                        'iscrowd': False,
                                        'note': 'Auto'}
                                    objects_list.append(obj); layer += 1.0  # noqa
                        mask_count += 1  # go to the next mask
                        with open(json_path, 'w', encoding='utf-8') as file:
                            json.dump({'info': info_dict, 'objects': objects_list}, file, indent=4)
                return None
            except ValueError:
                print('\n \x1b[31m skip this batch since no stomata detected \n')
                return None

    def batch_label(self) -> None:
        """Convert predictions into ISAT format for auto labeling"""
        folders = sorted([root for root, _, files in os.walk(self.input_dir)], key=str.casefold)  # sort folders by name
        for folder in folders:
            print(f'\n \033[34m processing {folder} \n')
            batches = self.batch_images(folder)  # get subfolder image batches
            for idx, batch in enumerate(batches):
                print(f'\n \033[34m batch {idx + 1} out of {len(batches)} \n')
                self.auto_label(batch)  # run starch detection pipline for each batch
        return None
