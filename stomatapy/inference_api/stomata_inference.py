"""Module providing functions inference stomata images"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, relative-beyond-top-level, wildcard-import, no-name-in-module
import os  # interact with the operating system
import json  # manipulate json files
import cv2  # OpenCV
import numpy as np  # NumPy
from tqdm import tqdm  # progress bar
import pandas as pd  # for Excel sheet
from ..core.core import device, image_types, Cell_Colors, imread_rgb, color_select, binary, lab_logo  # import core functions
from ..utils.stoma_dimension import GetDiameter  # import core functions for stomatal aperture
from ..core.isat import UtilsISAT  # functions to manipulate ISAT segmentations

from typing import List  # for typing hints
from ..core.core import get_paths   # import core functions


class StomataInference:
    """Inference stomata images"""
    def __init__(self,
                 input_dir: str = None,
                 output_name: str = 'Results aperture',
                 batch_size: int = 20,
                 pixels_per_micrometer: float = 8.0,
                 concatenate_excels: bool = True,
                 seg_onehot_mapping: dict = {cell_color.class_encoding: cell_color.class_name for cell_color in Cell_Colors},
                 padding: int = 20):
        self.input_dir = os.path.normpath(input_dir)  # input directory
        self.output_name = output_name  # output folder name
        self.batch_size = batch_size  # inference batch size
        self.pixels_per_micrometer = pixels_per_micrometer  # number of pixels per micrometer
        self.concatenate_excels = concatenate_excels  # concatenate Excel sheets from all subfolders
        self.seg_onehot_mapping = seg_onehot_mapping  # segmentation one-hot code against class_name
        self.seg_color_mapping = {cell_color.class_name: cell_color.mask_rgb for cell_color in Cell_Colors}  # mapp the segmentation class names to their colors
        self.padding = padding  # padding for the bboxes

        if self.ensemble_detectors:
            self.batch_size //= 2  # reduce batch size to half for ensemble detectors
            print(f'\n \033[34m Note: ensemble_detectors=True, batch_size reduces to half ({self.batch_size}) \n')

    def if_seg_on_edges(self, seg_mask: np.ndarray, edge_width: int = 3) -> bool:
        """
        Check if the segmentation mask has been cut off by any edges

        Args:
        - seg_mask (np.ndarray): the segmentation mask in ISAT format
        - edge_width (int): the threshold for edge width. Default is 3

        Returns:
        - bool: True or False if segmentation mask is not cut off by any edges
        """
        top_edge = seg_mask[:edge_width, :]  # pixels on the top edge
        bottom_edge = seg_mask[-edge_width:, :]  # bottom edge
        left_edge = seg_mask[:, :edge_width]  # left edge
        right_edge = seg_mask[:, -edge_width:]  # right edge
        edges = [top_edge, bottom_edge, left_edge, right_edge]  # all four edges
        return any(np.any(edge) for edge in edges)

    @staticmethod
    def json2excel(input_dir, output_dir, scale: float = 2.9):
        """Get stomata triats from ISAT json files of input folders, and store the results as an Excel sheet"""
        batch_results = pd.DataFrame()  # to store results into a DataFrame
        json_paths = []  # to store the ISAT json file paths to be inferenced
        for dir_path, dir_names, file_names in os.walk(input_dir):
            dir_json_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in ['.json'])]  # json files only
            dir_json_paths = [os.path.join(dir_path, dir_json_name) for dir_json_name in dir_json_names]  # get the json file paths of the given directory
            json_paths.append(dir_json_paths)

        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json file
            image_name = data['info']['name']  # get the image path stored in the json file
            image_extension = os.path.splitext(image_name)[:1]  # get the image extension
            image_path = json_path.replace('.json', image_extension)  # get the image path in case user moved the files (robust as the json file should be close to the image file)
            if not os.path.exists(image_path):
                continue
            image = imread_rgb(image_path)  # load the image
            image_dimension = (data['info']['height'], data['info']['width'])  # get the image dimension
            for idx, obj in enumerate(data['objects']):
                overlay_color = np.array([0, 0, 255])
                mask = obj['segmentation']  # get the ISAT format segmentation mask
                mask_bool = UtilsISAT.segmentation2mask(mask, image_dimension)  # convert the ISAT mask to a bool mask
                if not StomataInference.if_seg_on_edges(mask_bool):
                    mask_area = np.sum(mask_bool) * (1 / scale) ** 2  # get the mask area
                    bbox = obj['bbox']  # get the object bbox
                    padding = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1])) // 4  # calculate the padding value of the bbox as max 25% of either dimension
                    mask_filled = np.uint8(np.stack([mask_bool, mask_bool, mask_bool], axis=-1) * 255)  # fill in the bool mask with RGB white color for dimension
                    cropped_mask = UtilsISAT.crop_image_with_padding(mask_filled, bbox, padding, allow_negative_crop=True)  # crop the mask to be focused
                    dimension = GetDiameter(cropped_mask, shrink_ratio=0.8, line_thickness=1).pca()  # get the lenghth and the width of the object
                    length, width = dimension['length'] * (1 / scale), dimension['width'] * (1 / scale)  # convert the length and the width to microns
                    length_points, width_points = dimension['length_points'], dimension['width_points']
                    x_min_padded, y_min_padded = bbox[0] - padding, bbox[1] - padding
                    original_length_points = [(int(x + x_min_padded), int(y + y_min_padded)) for x, y in length_points]
                    original_width_points = [(int(x + x_min_padded), int(y + y_min_padded)) for x, y in width_points]
                    overlay_color = np.array([255, 0, 0])
                    cv2.line(image, original_length_points[0], original_length_points[1], (0, 255, 0), 2)  # draw the length
                    cv2.line(image, original_width_points[0], original_width_points[1], (0, 0, 255), 2)  # draw the width
                    result = {
                        'folder': os.path.dirname(json_path),
                        'image_name': image_name,
                        'object_idx': idx,
                        'object_category': 'stomatal complex',
                        'area  (\u03BCm\N{SUPERSCRIPT TWO})': mask_area,
                        'length (\u03BCm)': length,
                        'width (\u03BCm)': width,
                        'angle (°)': dimension['angle']
                    }
                    result = pd.DataFrame(data=[result])  # collect result in a pd dataframe for exporting to an Excel sheet
                    batch_results = pd.concat([batch_results, result], axis=0)  # concatenate all results
                image[mask_bool] = image[mask_bool] * 0.5 + overlay_color * 0.5  # create starch overlay on the original image
                cv2.imwrite(os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_prediction.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # export the image
        batch_results.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)  # export the summarized results to Excel
