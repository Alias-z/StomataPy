"""Module providing functions preparing stomata images for training"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import time  # waiting for file to be closed if necessary
import shutil  # for copying files
import json  # manipulate json files
import random  # for random sampling
from typing import Literal, Tuple  # to support type hints
import numpy as np  # NumPy
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True  # noqa: Pillow image processing
from tqdm import tqdm  # progress bar
import torch  # PyTorch
from sahi.slicing import slice_coco  # slice COCO dataset images and annotations into grids
import pycocotools.mask as mask_util  # interface for manipulating masks stored in RLE format
from ..core.core import image_types, Cell_Colors, imread_rgb  # import core functions
from ..core.isat import UtilsISAT, ISAT2Anything  # import functions for ISAT json files
from .feature_clustering import FeatureClustering  # to computer image feature similarity
from .data_statistics import DataStatistics  # to filter datasets

random.seed(123)  # set random seed for reproducibility


class Data4Training:
    """Prapre training data"""
    def __init__(self,
                 input_dir: str = None,
                 aim: Literal['semantic segmentation', 'object detection', 'instance segmentation'] = 'semantic segmentation',
                 new_width: int = 4352,
                 new_height: int = 1844,
                 r_train: float = 0.74,
                 r_test: float = 0.1,
                 crop_padding_ratio: float = 0.2,
                 use_sahi: bool = True,
                 slice_width: int = 1280,
                 slice_height: int = 1024,
                 sahi_overlap_ratio: float = 0.2,
                 remove_copy: bool = True,
                 remove_pore: bool = False):
        self.input_dir = input_dir  # input directory
        self.aim = aim  # aim for stomata model training
        self.new_width = new_width  # new width after resizing
        self.new_height = new_height  # new height after resizing
        self.r_train = r_train  # ratio of training data
        self.r_test = r_test  # ratio of test data
        self.crop_padding_ratio = crop_padding_ratio  # to padding value after cropping (for segmentation)
        self.use_sahi = use_sahi  # if use sahi to slice the images and annotations (for detection)
        self.slice_width = slice_width  # the target slice width
        self.slice_height = slice_height  # the target slice height
        self.sahi_overlap_ratio = sahi_overlap_ratio  # fractional overlap in width of each slice
        self.remove_copy = remove_copy  # remove copied folders
        self.remove_pore = remove_pore  # remove 'pore' class

        valid_aims = ['semantic segmentation', 'object detection', 'instance segmentation']
        assert self.aim in valid_aims, f'Invalid aim value. Must be one of {valid_aims}'
        assert self.r_train + self.r_test < 1, 'r_train + r_test >= 1'

    @staticmethod
    def calculate_optimal_slicing(slice_width: int, slice_height: int, overlap_ratio: float, num_width_slices: int, num_height_slices: int) -> Tuple[int, int]:
        """
        Calculate the minimum image dimensions required to fit a specified number of slices with overlaps.

        Args:
        - slice_width (int): the width of each slice in pixels
        - slice_height (int): the height of each slice in pixels
        - overlap_ratio (float): the ratio of overlap between slices (e.g., 0.2 for 20% overlap)
        - num_width_slices (int): the desired number of slices along the width of the image
        - num_height_slices (int): the desired number of slices along the height of the image

        Returns:
        - target_image_dimensions (tuple): a tuple containing the calculated minimum image width and height in pixels to accommodate the slices.
        """
        overlap_width, overlap_height = int(slice_width * overlap_ratio), int(slice_height * overlap_ratio)  # calculate the overlap in pixels
        effective_width, effective_height = slice_width - overlap_width, slice_height - overlap_height  # calculate the non-overlapping effective area of each slice
        if num_width_slices > 1:
            image_width = slice_width + (num_width_slices - 1) * effective_width  # calculate the minimum dimensions of the image to fit the desired number of slices
        else:
            image_width = slice_width
        if num_height_slices > 1:
            image_height = slice_height + (num_height_slices - 1) * effective_height  # same for the height
        else:
            image_height = slice_height
        return (image_width, image_height)  # f(1280, 1024, 0.2, 4, 2) = (4352, 1844)

    def get_padded_bbox(self, bbox: list, image_width: int, image_height: int, padding=9) -> list:
        """Padding bbox for cropping image"""
        xmin, ymin, width, height = bbox  # MSCOCO format
        xmin = max(0, xmin - padding)  # min x after padding
        ymin = max(0, ymin - padding)  # min y after padding
        xmax = min(image_width, xmin + width + 2 * padding)  # max x after padding
        ymax = min(image_height, ymin + height + 2 * padding)  # max y after padding
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def crop_coco(self, input_dir: str) -> None:
        """Crop images accoring to needs based on COCO.json"""
        json_file = os.path.join(input_dir, 'COCO.json')  # open COCO.json
        coco_data = json.load(open(json_file, 'r', encoding='utf-8'))  # load info from json
        categories = coco_data['categories']  # get 'categories', e.g. {'name': 'open stomata', 'id': 2, 'supercategory': None}
        annotations = coco_data['annotations']  # get 'annotations' e.g. {'iscrowd': 0, 'image_id': 0, 'image_name': xx.png', 'category_id': 3, 'id': 0, 'segmentation': [[...]],  'area': 1845.0, 'bbox': [1612.0, 1062.0, 34.0, 79.0]}
        images = coco_data['images']  # get 'images', e.g. {'license': None, 'url': None, 'file_name': 'xxx.png', 'height': 1920, 'width': 2560, 'date_captured': None, 'id': 0}
        color_mapping = {'pavement cell': Cell_Colors[5].mask_rgb, 'stomatal complex': Cell_Colors[1].mask_rgb, 'stoma': Cell_Colors[2].mask_rgb,
                         'outer ledge': Cell_Colors[3].mask_rgb, 'pore': Cell_Colors[4].mask_rgb}  # mapping segmentation classes with assigned colors
        palette = [seg_color.mask_rgb for seg_color in Cell_Colors]  # the one-hot color mapping
        os.makedirs(os.path.join(input_dir, 'images'), exist_ok=True); os.makedirs(os.path.join(input_dir, 'labels'), exist_ok=True)  # noqa: create folders for images and labels
        for image in tqdm(images, total=len(images)):
            mask = np.zeros((image['height'], image['width'], 3), dtype=np.uint8)  # create empty black mask
            # The following oder MATTERS! Since the later catergories will overlay on the former ones
            pavement_cell_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']
                                         and next((c['name'] for c in categories if c['id'] == annotation['category_id']), None) == 'pavement cell']  # noqa: get 'pavement cell' annotations
            stomata_complex_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']
                                           and next((c['name'] for c in categories if c['id'] == annotation['category_id']), None) == 'stomatal complex']  # noqa: get 'stomatal complex' annotations
            stoma_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']
                                 and next((c['name'] for c in categories if c['id'] == annotation['category_id']), None) == 'stoma']  # noqa: get 'stoma' annotations
            outer_ledge_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']
                                       and next((c['name'] for c in categories if c['id'] == annotation['category_id']), None) == 'outer ledge']  # noqa: get 'outer ledge' annotations
            pore_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']
                                and next((c['name'] for c in categories if c['id'] == annotation['category_id']), None) == 'pore']  # noqa: get 'pore' annotations
            for annotation in pavement_cell_annotations + stomata_complex_annotations + stoma_annotations + outer_ledge_annotations + pore_annotations:
                category_id = annotation['category_id']  # get 'category_id'
                category_name = next((c['name'] for c in categories if c['id'] == category_id), None)  # get 'category_name'
                rle = annotation['segmentation']  # load MSCOCO RLE format segmentation
                rle = mask_util.frPyObjects(rle, image['height'], image['width'])  # convert polygon, bbox, and uncompressed RLE to encoded RLE mask
                binary_mask = mask_util.decode(rle)  # shape (height, width, 1)
                binary_mask = np.squeeze(binary_mask)  # remove singleton dimensions, shape (height, width)
                for channel in range(3):
                    mask[:, :, channel] = np.where(binary_mask == 1, color_mapping[category_name][channel], mask[:, :, channel])  # fill in category specific colors
            if len(stomata_complex_annotations) > 0:
                annotation2crop = stomata_complex_annotations  # always select the largest region to crop
            else:
                annotation2crop = stoma_annotations  # if stomatal complex does not exist, stomata would be the largest region to crop
            for annotation in annotation2crop:
                image_name = image['file_name']  # get the image name
                image_pil = Image.open(os.path.join(input_dir, image_name))  # open image
                bbox = annotation['bbox']  # get the annotation bbox
                _, _, width, height = bbox
                padding = int(self.crop_padding_ratio * max(width, height))
                padded_bbox = self.get_padded_bbox(bbox=annotation['bbox'], image_width=image_pil.width, image_height=image_pil.height, padding=padding)  # get padded bbox
                cropped_image = image_pil.crop(padded_bbox)  # crop image with padded bbox
                cropped_mask = np.array(Image.fromarray(mask, 'RGB').crop(padded_bbox))  # RGB
                cropped_mask_onehot = np.zeros(cropped_mask.shape[:2], dtype=np.uint8)  # create an empty grayscale mask with the same size
                for seg_color in Cell_Colors:
                    cropped_mask_onehot[np.all(cropped_mask == np.array(seg_color.mask_rgb), axis=-1)] = seg_color.class_encoding  # RGB to one-hot encoding
                cropped_mask_onehot = Image.fromarray(cropped_mask_onehot)  # convert the numpy array back to a PIL imag
                output_name = f"{os.path.splitext(image_name)[0]} object id {annotation['id']}.png"  # names for output
                # palette = []  # palette for visualization
                # for idx in np.unique(cropped_mask_onehot).tolist():
                #     palette.extend(Cell_Colors[idx].mask_rgb)
                cropped_mask_onehot.putpalette(np.array(palette, dtype=np.uint8))  # for visualize mask
                cropped_image.save(os.path.join(input_dir, 'images', output_name))  # save cropped image
                cropped_mask_onehot.save(os.path.join(input_dir, 'labels', output_name))  # save cropped mask

        file_names = [name for name in os.listdir(input_dir) if any(name.endswith(file_type) for file_type in image_types)]  # get original images' names
        for file_name in file_names:
            os.remove(os.path.join(input_dir, file_name))  # remove original image files
        os.remove(os.path.join(input_dir, 'COCO.json'))  # remove the COCO.json
        return None

    @staticmethod
    def find_similary_jsons(datasets_root: str = 'Datasets', sample_size: int = 1) -> None:
        """Find the target dataset based on image feature similarity"""

        def sample_metadata(selected_jsons: dict, sample_size: int = 1) -> dict:
            """"Randomly sample the metadata enrties from the selected jons files"""""
            sampled_jsons = {}
            for dataset_name, entries in selected_jsons.items():
                if len(entries) >= sample_size:
                    sampled_jsons[dataset_name] = random.sample(entries, sample_size)   # if the dataset has enough entries, sample 'n' entries randomly
                else:
                    sampled_jsons[dataset_name] = random.sample(entries, len(entries))  # if the dataset does not have enough entries, sample all the entries
            return sampled_jsons

        selected_jsons = DataStatistics(root_dir=datasets_root).select_species_folders()  # get a list of jsons files that have been annotated
        sampled_jsons = sample_metadata(selected_jsons, sample_size)  # randomly sample jsons from each dataset - species folder to computer image feature similarity

        for _, metadata_list in tqdm(sampled_jsons.items(), total=len(sampled_jsons.items())):
            for metadata in metadata_list:
                image = imread_rgb(metadata.get('image_path'))  # load the image
                image_feature = FeatureClustering.extract_features(image=image, model_type='DINOv2')  # extract image features
                metadata['image_features'] = image_feature
        torch.save(sampled_jsons, os.path.join(datasets_root, 'extracted_features.pth'))
        return sampled_jsons

    def remove_black_images(self, coco_json_path: str, coco_images_dir: str, threshold: float = 0.5) -> None:
        """
        Remove images that are mostly black based on a pixel threshold and update the COCO.json accordingly.

        Args:
        - coco_json_path (str): the path to the COCO.json file
        - coco_images_dir (str): the directory containing the image files
        - threshold (float): the threshold for determining if an image is mostly black
        """

        def check_black_pixels(image_path: str, threshold: float = 0.5) -> bool:
            """Helper function to check the ratio of black pixels in an image"""
            with Image.open(image_path) as image:
                image = np.array(image)  # load the image
                black_pixels = np.all(image == 0, axis=-1)  # compute the number of black pixels
                black_ratio = np.sum(black_pixels) / black_pixels.size  # compute the black pixels ratio
            return black_ratio > threshold

        def safe_remove(image_path):
            """Attempt to remove the file with retries."""
            for _ in range(5):  # retry up to 5 times
                try:
                    os.remove(image_path)  # remove the file
                    # print(f"Successfully deleted {image_path}")
                    return
                except PermissionError:
                    # print(f"Permission error deleting {image_path}:, retrying...")
                    time.sleep(1)  # wait a bit for the file to be released
            print(f'Failed to delete {image_path} after several attempts')

        with open(coco_json_path, 'r', encoding='utf-8') as file:
            coco_data = json.load(file)  # load the COCO.json file

        images_to_keep, annotations_to_keep = [], []  # to store the final annotations set
        annotations_map = {ann['image_id']: [] for ann in coco_data['annotations']}
        for ann in coco_data['annotations']:
            annotations_map[ann['image_id']].append(ann)

        for image in coco_data['images']:
            image_path = os.path.join(coco_images_dir, image['file_name'])  # get the image path from COCO.json
            if os.path.exists(image_path):
                if check_black_pixels(image_path, threshold):
                    safe_remove(image_path)  # remove images where black pixels > threshold using safe remove
                else:
                    images_to_keep.append(image)  # keep the other images
                    annotations_to_keep.extend(annotations_map.get(image['id'], []))

        new_coco_data = {
            'images': images_to_keep,
            'annotations': annotations_to_keep,
            'categories': coco_data['categories']  # Assuming categories remain unchanged
        }  # create a new COCO.json file with the updated dataset

        with open(coco_json_path, 'w', encoding='utf-8') as new_file:
            json.dump(new_coco_data, new_file)  # replace the old COCO.json file
        return None

    def data4training(self, remove_subgroups: bool = True, if_resize_isat: bool = True, output_rename: str = None) -> None:
        """Generate data for training from ISAT json files"""
        input_copy_dir = self.input_dir + ' - Copy'  # folder copy dir
        UtilsISAT.copy_folder(self.input_dir, input_copy_dir)  # create a copy
        if self.aim != 'semantic segmentation':
            UtilsISAT.select_class(input_copy_dir, action='rename class', source_class='stoma', destination_class='stomatal complex')  # replace non-duplicated stoma with stomatal complex
            output_name = 'Epidermal_segmentation' if self.aim == 'instance segmentation' else 'Stomata_detection'  # output directory name
            classes2remove = ['stoma', 'outer ledge', 'pore'] if self.aim == 'instance segmentation' else ['stoma', 'outer ledge', 'pore', 'pavement cell']  # the catergories to be removed
            if remove_subgroups:
                for category in classes2remove:
                    UtilsISAT.select_class(input_copy_dir, category=category, action='remove')  # remove catergoreis that are not 'stomatal complex' or 'pavement cell'
            if self.use_sahi:
                UtilsISAT.shapely_valid_transform(input_copy_dir)  # transform the polygons to be valid as shapely require

        elif self.aim == 'semantic segmentation':
            if self.remove_pore:
                UtilsISAT.select_class(input_copy_dir, category='pore', action='remove')  # remove all 'pore' annotations
            if if_resize_isat:
                UtilsISAT.resize_isat(input_copy_dir, new_width=self.new_width, new_height=self.new_height, if_keep_ratio=True)  # resize images and annotations
            output_name = 'Stomata_segmentation'  # for semantic segmentation

        if output_rename is not None:
            output_name = output_rename  # rename the output directory

        output_dir = os.path.join(os.path.split(input_copy_dir)[0], output_name)  # COCO json output dir
        train_dir, val_dir, test_dir = os.path.join(output_dir, 'train'), os.path.join(output_dir, 'val'), os.path.join(output_dir, 'test')  # COCO json train/val/test directory
        UtilsISAT.data_split(input_copy_dir, output_dir, r_train=self.r_train, r_test=self.r_test)  # split train, val and test
        for directory in [train_dir, val_dir, test_dir]:
            if if_resize_isat and directory != test_dir:
                UtilsISAT.resize_isat(directory, new_width=self.new_width, new_height=self.new_height, if_keep_ratio=True)  # resize images and annotations
            ISAT2Anything.to_coco(directory, output_dir=os.path.join(directory, 'COCO.json'))  # convert train/val ISAT json files to COCO
            if self.use_sahi and self.aim != 'semantic segmentation' and directory != test_dir:
                output_dir = f'{directory}_sahi'  # the output_dir
                slice_coco(
                    coco_annotation_file_path=os.path.join(directory, 'COCO.json'),
                    image_dir=directory,
                    output_coco_annotation_file_name='sahi',
                    output_dir=output_dir,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width,
                    overlap_height_ratio=self.sahi_overlap_ratio,
                    overlap_width_ratio=self.sahi_overlap_ratio,
                    min_area_ratio=0.2
                )  # slice the MSCOCO images and annotations
                shutil.rmtree(directory)  # remove the orginal COCO directory
                self.remove_black_images(coco_json_path=os.path.join(output_dir, 'sahi_coco.json'), coco_images_dir=output_dir, threshold=0.3)  # filter out images contain mostlt black pixels
        if self.remove_copy:
            shutil.rmtree(input_copy_dir)  # remove copied foder

        if self.aim == 'semantic segmentation':
            self.crop_coco(os.path.join(output_dir, 'train'))  # crop stomata for training set
            self.crop_coco(os.path.join(output_dir, 'val'))  # crop stomata for val set
            splits_dir = os.path.join(output_dir, 'splits'); os.makedirs(splits_dir, exist_ok=True)  # noqa: train/val file dir
            train_names = [name for name in os.listdir(os.path.join(train_dir, 'images')) if any(name.endswith(file_type) for file_type in image_types)]  # training image names
            val_names = [name for name in os.listdir(os.path.join(val_dir, 'images')) if any(name.endswith(file_type) for file_type in image_types)]  # validation image names
            with open(os.path.join(splits_dir, 'train.txt'), 'w', encoding='utf-8') as file:
                file.writelines(os.path.splitext(name)[0] + '\n' for name in train_names)  # creat a txt file for train
            with open(os.path.join(splits_dir, 'val.txt'), 'w', encoding='utf-8') as file:
                file.writelines(os.path.splitext(name)[0] + '\n' for name in val_names)  # create a txt file for validation
            images_dir = os.path.join(output_dir, 'images'); os.makedirs(images_dir, exist_ok=True)  # noqa: images dir
            labels_dir = os.path.join(output_dir, 'labels'); os.makedirs(labels_dir, exist_ok=True)  # noqa: labels dir
            for folder in [train_dir, val_dir]:
                image_names = [name for name in os.listdir(os.path.join(folder, 'images')) if any(name.endswith(file_type) for file_type in image_types)]  # load the image names
                for image_name in image_names:
                    source_path = os.path.join(folder, 'images', image_name)  # source image path
                    destination_path = os.path.join(images_dir, image_name)  # destination image path
                    shutil.copy(source_path, destination_path)  # paste images
                label_names = [name for name in os.listdir(os.path.join(folder, 'labels')) if any(name.endswith(file_type) for file_type in image_types)]  # load the mask names
                for label_name in label_names:
                    source_path = os.path.join(folder, 'labels', label_name)  # source mask path
                    destination_path = os.path.join(labels_dir, label_name)  # destination mask path
                    shutil.copy(source_path, destination_path)  # paste labels
            shutil.rmtree(train_dir); shutil.rmtree(val_dir)  # noqa: remove 'train' and 'val' folders
        return None
