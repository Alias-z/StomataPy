"""Output Excel results from ISAT json annoation"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import json  # manipulate json files
import cv2  # OpenCV
import numpy as np  # NumPy
import pandas as pd  # for Excel sheet
from tqdm import tqdm  # progress bar
from matplotlib import pyplot as plt  # for visualization
from ..core.core import get_paths, imread_rgb  # import core functions
from ..utils.stoma_dimension import GetDiameter  # import core functions for stomatal aperture
from ..core.isat import UtilsISAT  # functions to manipulate ISAT segmentations


def if_seg_on_edges(seg_mask: np.ndarray, edge_width: int = 3) -> bool:
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


def json2excel(input_dir, output_dir, scale: float = 2.9, show_prediction: bool = True):
    """Get stomata triats from ISAT json files of input folders, and store the results as an Excel sheet"""
    batch_results = pd.DataFrame()  # to store results into a DataFrame
    # json_paths = []  # to store the ISAT json file paths to be inferenced
    # for dir_path, dir_names, file_names in os.walk(input_dir):
    #     dir_json_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in ['.json'])]  # json files only
    #     dir_json_paths = [os.path.join(dir_path, dir_json_name) for dir_json_name in dir_json_names]  # get the json file paths of the given directory
    #     json_paths.append(dir_json_paths)
    os.makedirs(output_dir, exist_ok=True)
    json_paths = get_paths(input_dir, '.json')

    for json_path in tqdm(json_paths, total=len(json_paths)):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # load the json file
        image_name = data['info']['name']  # get the image path stored in the json file
        image_extension = os.path.splitext(image_name)[1]  # get the image extension
        image_path = json_path.replace('.json', image_extension)  # get the image path in case user moved the files (robust as the json file should be close to the image file)
        if not os.path.exists(image_path):
            continue
        image = imread_rgb(image_path)  # load the image
        image_dimension = (data['info']['height'], data['info']['width'])  # get the image dimension
        for idx, obj in enumerate(data['objects']):
            overlay_color = np.array([0, 0, 255])
            mask = obj['segmentation']  # get the ISAT format segmentation mask
            mask_bool = UtilsISAT.segmentation2mask(mask, image_dimension)  # convert the ISAT mask to a bool mask
            if not if_seg_on_edges(mask_bool):
                mask_area = np.sum(mask_bool) * (1 / scale) ** 2  # get the mask area
                bbox = obj['bbox']  # get the object bbox
                padding = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1])) // 4  # calculate the padding value of the bbox as max 25% of either dimension
                mask_filled = np.uint8(np.stack([mask_bool, mask_bool, mask_bool], axis=-1) * 255)  # fill in the bool mask with RGB white color for dimension
                cropped_mask = UtilsISAT.crop_image_with_padding(mask_filled, bbox, padding, allow_negative_crop=True)  # crop the mask to be focused
                # Skip objects with invalid masks
                if cropped_mask is None or cropped_mask.size == 0:
                    continue
                try:
                    dimension = GetDiameter(cropped_mask, shrink_ratio=0.9, line_thickness=1).pca()
                except (ValueError, TypeError):
                    # plt.imshow(cropped_mask)
                    # plt.show()
                    continue  # Skip objects that fail dimension calculation
                # Add validation for dimension results
                if not dimension or 'length' not in dimension:
                    continue

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
            if show_prediction:
                plt.imshow(image)
                plt.show()
    batch_results.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)  # export the summarized results to Excel
