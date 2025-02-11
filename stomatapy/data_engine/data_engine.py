"""Module converting community data to unified ISAT format"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, wildcard-import, no-member, too-many-function-args
import os  # interact with the operating system
import shutil  # for copying files
import json  # manipulate json files
import re  # regular expression operations
from typing import Literal, List  # to support type hints
import xml.etree.ElementTree as ET  # to load xml annotation files
import cv2  # OpenCV
from PIL import Image, ImageDraw  # Pillow image processing
import numpy as np  # NumPy
from tqdm import tqdm  # progress bar
import pandas as pd  # for Excel sheet and CSV file
from skimage import measure  # to measure properties of labeled image regions
from matplotlib import pyplot as plt  # for image visualization
from ..models.sam_hq import SAMHQ  # Segment Anything in High Quality
from ..core.core import image_types, get_paths, imread_rgb  # import core functions
from ..core.isat import UtilsISAT, Anything2ISAT  # import functions related to the ISAT format


class StomataPyData:
    """
    Thi is the base class for processing community data.
    The core function is to unify the file name of original data in the following order:
        Species + source name + original file name
        Example: A.thaliana Lucia 05.07.2021 col-0 ps t0 07131528.png

    Every folder should be organized in the following order:
    The folder
    ├── Original
    ├── Processed
        ├── Species 1
        ...
        ├── Species n
    ├── source.txt
    ├── discard.txt

    The original folder contains a copy of the community data
    The processed folder stores renamed image files and annotations, group by spceies names
    The sorce.txt file indicates the downloading link of the community data
    The discard.txt indicates the images to be removed
    """
    def __init__(self,
                 index: int = None,
                 input_dir: str = None,
                 species: str = None,
                 source_name: str = None):
        self.index = index,  # the species index
        self.input_dir = input_dir  # input directory
        self.species = species  # species
        self.source_name = source_name  # source name
        self.aono2021_dir = 'Datasets//Aono2021//Original'   # the directory of Aono2021
        self.casadogarcia2020_dir = 'Datasets//Casado-Garcia2020//Original'  # the directory of Casado-Garcia2020
        self.dey2023_dir = 'Datasets//Dey2023//Original'  # directory of Dey2023
        self.ferguson2021_dir = 'Datasets//Ferguson2021//Original'  # directory of Ferguson2021
        self.fetter2019_dir = 'Datasets//Fetter2019//Original'  # directory of Fetter2019
        self.jayakody2017_dir = 'Datasets//Jayakody2017//Original'  # directory of Jayakody2017
        self.koheler2023_dir = 'Datasets//Koheler2023//Original'  # directory of Koheler2023
        self.koheler2024_dir = 'Datasets//Koheler2024//Original'  # directory of Koheler2024
        self.li2022_dir = 'Datasets//Li2022//Original'  # directory of Li2022
        self.li2023_dir = 'Datasets//Li2023//Original'  # directory of Li2023
        self.meeus2020_dir = 'Datasets//Meeus2020//Original'  # directory of Meeus2020
        self.meng2023_dir = 'Datasets//Meng2023//Original'  # directory of Meng2023
        self.pathoumthong2023_dir = 'Datasets//Pathoumthong2023//Original'  # directory of Pathoumthong2023
        self.sultana2021_dir = 'Datasets//Sultana2021//Original'  # directory of Sultana2021
        self.sun2021_dir = 'Datasets//Sun2021//Original'  # directory of Sun2021
        self.sun2023_dir = 'Datasets//Sun2023//Original'  # directory of Sun2023
        self.takagi2023_dir = 'Datasets//Takagi2023//Original'  # directory of Takagi2023
        self.thathapalliprakash2021_dir = 'Datasets//ThathapalliPrakash2021//Original'  # directory of ThathapalliPrakash2021
        self.toda2018_dir = 'Datasets//Toda2018//Original'  # directory of Toda2018
        self.toda2021_dir = 'Datasets//Toda2021//Original'  # directory of Toda2021
        self.vofely2019_dir = 'Datasets//Vofely2019//Original'  # directory of Vofely2019
        self.wangrenninger2023_dir = 'Datasets//WangRenninger2023//Original'  # directory of WangRenninger2023
        self.xie2021_dir = 'Datasets//Xie2021//Original'  # directory of Xie2021
        self.yang2021_dir = 'Datasets//Yang2021//Original'  # directory of Yang2021
        self.yates_dir = 'Datasets//Yates2018//Original'  # directory of Yates2018
        self.zhu2021_dir = 'Datasets//Zhu2021//Original'  # directory of Zhu2021
        self.liang2022_dir = 'Datasets//2022 Liang et al//Original'  # directory of Liang2022

    @staticmethod
    def ensemble_files(input_dir: str, subfolders: list, output_dir: str, file_types: list = image_types, folder_rename: bool = False) -> list:
        """Ensemble all defined files types"""
        os.makedirs(output_dir, exist_ok=True)  # create output directory
        for subfolder in subfolders:
            file_names = sorted(os.listdir(os.path.join(input_dir, subfolder)), key=str.casefold)  # sort file names
            file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in file_types)]  # image files only
            for file_name in file_names:
                file_path = os.path.join(input_dir, subfolder, file_name)  # file path
                output_path = os.path.join(output_dir, file_name)  # output path
                if folder_rename:
                    output_path = os.path.join(output_dir, f'{subfolder} {file_name}')  # output path with folder name in front of file name
                shutil.copy2(file_path, output_path)  # copy file for ensembling
        return file_names

    @staticmethod
    def abbreviate_species(full_name: str) -> str:
        """
        Abbreviate species name.
        For example: Quercus texana Buckley -> Q. texana
        If name is short: Populus L. -> Populus spp
        """
        parts = full_name.split()  # split species name
        if len(parts) == 1 or (len(parts) > 1 and parts[1] == 'L.'):
            return f'{parts[0]} spp'  # if only the genus name is present, append ' spp.'
        elif len(parts) == 1 or (len(parts) > 1 and parts[1] == 'sp'):
            return f'{parts[0]} sp'  # if only the genus name is present, append ' sp.'
        else:
            return f'{parts[0][0]}. {parts[1]}'  # for names with genus and species, simplify the name

    @staticmethod
    def batch_rename(input_dir: str, file_names: list, new_names: list) -> None:
        """Batch rename files under a given directory"""
        for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
            source_dir = os.path.join(input_dir, file_name)  # original file name
            destination_dir = os.path.join(input_dir, new_names[idx])  # new file name
            os.rename(source_dir, destination_dir)  # rename
        return None

    @staticmethod
    def discard_files(discard_txt_path: str, input_dir: str) -> None:
        """Remove images in the discard.txt file"""
        n_removed = 0  # count removed files
        with open(discard_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines = [line.strip() for line in file]  # to store paths of images to be discarded
        image_names = set([name for name in os.listdir(input_dir) if any(name.lower().endswith(file_type) for file_type in image_types)])
        for file_name in list(image_names):
            for line in lines:
                if line in file_name:
                    file_path = os.path.join(input_dir, file_name)  # get the file path
                    if os.path.exists(file_path):
                        os.remove(file_path)  # remove images in the discard.txt file
                        n_removed += 1  # increment n_removed by 1
        print(f'Selected {len(image_names) - n_removed} images!')  # print out the number of selected images
        return None

    @staticmethod
    def create_species_folders(input_dir: str, species_names: set) -> None:
        """Group files based on plant species with subfolders"""
        file_names = os.listdir(input_dir)  # list of file names
        for species_name in species_names:
            subfolder_dir = os.path.join(input_dir, species_name); os.makedirs(subfolder_dir, exist_ok=True)  # noqa: create the species subfolder
            for file_name in file_names:
                if species_name in file_name:
                    file_path = os.path.join(input_dir, file_name)  # file path
                    output_path = os.path.join(subfolder_dir, file_name)  # output path
                    shutil.move(file_path, output_path)  # group the file by species
        return None

    @staticmethod
    def template_match(image_path: str = None, patch_paths: list = None, matching_threshold: float = 0.5, bbox_treshold: float = 0.2, visualize: bool = False) -> np.ndarray:
        """Map back cropped patches to the original image using template matching for those using sliding window + classification method"""
        isat_bboxes = []  # to store the bboxes of matched patches
        image_gray = cv2.imread(image_path, 0)  # read the original image in gray scale
        for patch_path in patch_paths:
            patch = cv2.imread(patch_path, 0)  # load the cropped patch in gray scale
            reslut = cv2.matchTemplate(image_gray, patch, cv2.TM_CCOEFF_NORMED)  # perform template matching
            _, max_value, _, max_location = cv2.minMaxLoc(reslut)  # get best matching position, and score
            if max_value >= matching_threshold:
                coco_bbox = np.array([max_location[0], max_location[1], patch.shape[1], patch.shape[0]])  # MSCOCO bbox
                isat_bboxes.append(UtilsISAT.bbox_convert(coco_bbox, 'COCO2ISAT'))  # append matched results
        isat_bboxes = UtilsISAT.isat_bboxes_filter(isat_bboxes, bbox_treshold)  # filter out the small overlapping bboxes
        if visualize:
            image = imread_rgb(image_path)  # read the image in RGB scale
            for isat_bbox in isat_bboxes:
                x_min, y_min, x_max, y_max = isat_bbox  # bbox coordinates
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)  # draw the bbox
            plt.imshow(image); plt.show()  # noqa: visualze matched patches
        return isat_bboxes

    @staticmethod
    def draw_xml_bbox(input_dir: str, image_extension: str = '.jpg') -> None:
        """Find all images and xml bbox annoation file, return the images with bboxes drawwn"""

        def draw_bboxes(image_path: str) -> None:
            """ Annotate an image with bounding boxes from the corresponding XML file"""
            xml_path = image_path.replace(os.path.splitext(image_path)[1], '.xml')  # get the corresponding xml file path
            tree = ET.parse(xml_path)  # represents the whole XML document as a tree structure
            root = tree.getroot()  # retrieves the root element of that tree
            xml_bboxes = []  # to store bboxes
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')  # looking for bndbox section
                xmin = int(bbox.find('xmin').text)  # convert xmin to int
                ymin = int(bbox.find('ymin').text)  # same for ymin
                xmax = int(bbox.find('xmax').text)  # xmax
                ymax = int(bbox.find('ymax').text)  # and y max
                xml_bboxes.append([xmin, ymin, xmax, ymax])  # get the converted bbox
            image = Image.open(image_path)  # load the image
            draw = ImageDraw.Draw(image)  # draw mode on the image
            for bbox in xml_bboxes:
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='red', width=2)  # draw the bboxes on the image
            image.save(image_path)  # save the annotated image
            return None

        image_paths = get_paths(input_dir, image_extension)  # get the image file paths
        for image_path in image_paths:
            draw_bboxes(image_path)  # draw the bboxes on the images under a given directory
        return None


class Aono2021(StomataPyData):
    """
    Aono et al., 2021  https://doi.org/10.1371/journal.pone.0258679
    Dataset source: https://zenodo.org/records/3938047

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Aono2021
    ├── Original
        ├── data
            ├── dataset
                ├── ERRO (ignored)
                ├── STOMA
                    ├── 1_10.jpeg
                    ...
                    ├── 200_179.jpeg
            ├── deep_learn
                ├── teste
                    ├── erro (ignored)
                    ├── stoma
                        ├── 5_49.jpeg
                        ...
                        ├── 200_179.jpeg
                ├── train
                    ├── erro (ignored)
                    ├── stoma
                        ├── 1_10.jpeg
                        ...
                        ├── 199_226.jpeg
            ├── original2
                ├── 1.jpg
                ...
                ├── 200.jpg
        ├── img (ignored)
        ├── src (ignored)
        ├── LICENSE
        ├── README.md
    ├── Processed
        ├── Z. mays
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images
    3. Map the patches from "Positive" back to renamed images to generate json fiels with bounding boxes
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.aono2021_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species_name = 'Z. mays'  # plant species
        self.source_name = 'Aono2021'  # source name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.patches_dir_1 = os.path.join(self.input_dir, 'data', 'dataset', 'STOMA')  # cropped stomata patches dir 1
        self.patches_dir_2 = os.path.join(self.input_dir, 'deep_learn', 'teste', 'stoma')  # cropped stomata patches dir 2
        self.patches_dir_3 = os.path.join(self.input_dir, 'deep_learn', 'train', 'stoma')  # cropped stomata patches dir 3
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.01}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""
        os.makedirs(self.species_folder_dir, exist_ok=True)  # create the species folder
        self.ensemble_files(os.path.join(self.input_dir, 'data'), ['original2'], self.processed_dir)  # move files to 'Processed' of image files
        file_paths = get_paths(self.processed_dir, '.jpg')  # get image path names
        for file_path in file_paths:
            new_basename = f'{self.species_name} {self.source_name} original2 {os.path.basename(file_path)}'  # get new path basename
            os.rename(file_path, os.path.join(self.species_folder_dir, new_basename))  # rename and move to the speceis folder
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.species_folder_dir)  # remove unwanted images
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        path_paths_1 = get_paths(self.patches_dir_1, '.jpeg')  # get the paths of path 1
        path_paths_2 = get_paths(self.patches_dir_2, '.jpeg')  # get the paths of path 2
        path_paths_3 = get_paths(self.patches_dir_3, '.jpeg')  # get the paths of path 3
        patch_paths = path_paths_1 + path_paths_2 + path_paths_3  # combine all patches paths
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt:
                    isat_bboxes = self.template_match(image_path, patch_paths)  # get ISAT format bboxes from template matching
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class CasadoGarcia2020(StomataPyData):
    """
    Casado-García et al., 2020 https://doi.org/10.1016/j.compag.2020.105751
    Dataset source: https://github.com/ancasag/labelStoma

    Rights and permissions:
    MIT License
    Copyright (c) <2015-Present> Tzutalin

    Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba, William T. Freeman

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Casado-Garcia2020
    ├── Original
        ├── all
            ├── test
                ├── JPEGImages
                    ├── (01)62_3_2Bx10.jpg
                    ├── (01)62_3_2Bx10.xml
                    ...
                    ├── T4024_2_A5.jpg
                    ├── T4024_2_A5.xml
            ├── train
                ├── JPEGImages
                    ├── (01)60_1_2Bx10.jpg
                    ├── (01)60_1_2Bx10.xml
                    ...
                    ├── T4024_2_B3.jpg
                    ├── T4024_2_B3.xml
        ├── bean (ignored)
        ├── bearley (ignored)
        ├── commonBean (ignored)
    ├── Processed
        ├── G. max
        ├── H. vulgare
        ├── P. vulgaris
    ├── source.txt
    ├── discard.txt

    1. Combine 'test' and 'train' under 'all'
    2. Since the 'all' folder does not indicating species names, we need to find them from 'bean' and 'commonBean' as images in 'bearley' (Barley) could be patches
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.casadogarcia2020_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Casado-Garcia2020'  # source name
        self.samhq_configs = {
            'G. max': {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04},
            'H. vulgare': {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.02},
            'P. vulgaris': {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}}

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""
        def names_by_species(species_folder_dir: str) -> list:
            """Get image names under a given speceis folder"""
            train_dir = os.path.join(species_folder_dir, 'train', 'JPEGImages')  # train directory
            train_names = [os.path.basename(image_path) for image_path in get_paths(train_dir, '.jpg')]  # image names under train directory
            test_dir = os.path.join(species_folder_dir, 'test', 'JPEGImages')  # test directory
            test_names = [os.path.basename(image_path) for image_path in get_paths(test_dir, '.jpg')]  # image names under test directory
            return train_names + test_names

        bean_names = names_by_species(os.path.join(self.input_dir, 'bean'))  # images names of G. max
        common_bean_names = names_by_species(os.path.join(self.input_dir, 'commonBean'))  # image names of P. vulgaris
        all_train_dir = os.path.join(self.input_dir, 'all', 'train')  # the 'all' train directory
        all_test_dir = os.path.join(self.input_dir, 'all', 'test')  # the 'all' train directory
        self.ensemble_files(all_train_dir, ['JPEGImages'], self.processed_dir, ['.jpg', '.xml'])  # move files to 'Processed' of train files
        self.ensemble_files(all_test_dir, ['JPEGImages'], self.processed_dir, ['.jpg', '.xml'])  # move files to 'Processed' of test files
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        for file_path in get_paths(self.processed_dir, file_extension='.xml'):
            if not os.path.exists(file_path.replace('.xml', '.jpg')):
                os.remove(file_path)  # remove the xml file whose image has been discarded
        file_names = os.listdir(self.processed_dir)  # the remaining file names
        bean_folder_dir = os.path.join(self.processed_dir, 'G. max'); os.makedirs(bean_folder_dir, exist_ok=True)  # noqa: create the G. max folder
        common_bean_folder_dir = os.path.join(self.processed_dir, 'P. vulgaris'); os.makedirs(common_bean_folder_dir, exist_ok=True)  # noqa: create the P. vulgaris folder
        barley_dir = os.path.join(self.processed_dir, 'H. vulgare'); os.makedirs(barley_dir, exist_ok=True)  # noqa: create the H. vulgare folder
        for file_name in file_names:
            file_path = os.path.join(self.processed_dir, file_name)  # get file path
            if file_name in set(common_bean_names) or len(file_name) <= 11:
                new_name = f'P. vulgaris {self.source_name} {file_name}'  # unify naming
                new_path = os.path.join(common_bean_folder_dir, new_name)  # path to P. vulgaris directory
            elif file_name in set(bean_names) or 'set' in file_name or len(file_name) <= 16 and 'x' not in os.path.splitext(file_name)[0]:
                new_name = f'G. max {self.source_name} {file_name}'  # unify naming
                new_path = os.path.join(bean_folder_dir, new_name)  # path to G. max directory
            else:
                new_name = f'H. vulgare {self.source_name} {file_name}'  # unify naming
                new_path = os.path.join(barley_dir, new_name)  # path to H. vulgare directory
            os.rename(file_path, new_path)  # rename image and move to its species directory
        return None

    def load_xml_bbox(self, xml_file_path: str) -> np.ndarray:
        """Load bbox of stomata from xml annotation files"""
        root, bboxes = ET.parse(xml_file_path).getroot(), []  # to store information from xml annotations
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')  # get each bbox
            xmin = int(bbox.find('xmin').text)  # find the xmin
            ymin = int(bbox.find('ymin').text)  # find the ymin
            xmax = int(bbox.find('xmax').text)  # find the xmax
            ymax = int(bbox.find('ymax').text)  # find the ymax
            bboxes.append((xmin, ymin, xmax, ymax))  # collect bbox to the list
        return np.array(bboxes)

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        for species in self.samhq_configs:
            species_config = self.samhq_configs.get(species)  # load the species config for SAM-HQ
            points_per_side, min_mask_ratio, max_mask_ratio = species_config['points_per_side'], species_config['min_mask_ratio'], species_config['max_mask_ratio']  # get config parameters
            subfolder_dir = os.path.join(self.processed_dir, species)  # get the species subfolder directory
            image_paths = get_paths(subfolder_dir, '.jpg')  # get the image paths
            for image_path in tqdm(image_paths, total=len(image_paths)):
                image = imread_rgb(image_path)  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                    if bbbox_prompt:
                        isat_bboxes = self.load_xml_bbox(image_path.replace('.jpg', '.xml'))  # get ISAT format bboxes from xml annotations
                        if len(isat_bboxes) > 0:
                            prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                            masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                    else:
                        masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    print(f'{image_path} cannot be segmented by SAM-HQ')
        return None


class Dey2023(StomataPyData):
    """
    Dey et al., 2023   https://doi.org/10.1016/j.ecoinf.2023.102128
    Dataset source: https://doi.org/10.17632/4brcwhmvyk.4

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Dey2023
    ├── Original
        ├── Code (ignored)
        ├── Raw data
            ├── rawdata
                ├── Aglaia cucullata
                    ├── 0118.jpg
                    ...
                    ├── 0200.jpg
                ├── Barringtonia acutangula
                    ├── Hijol (1).jpg
                    ...
                    ├── Hijol (81).jpg
                ├── Bruguiera gymnorrhiza
                    ├── Lal_kakra (1).jpg
                    ...
                    ├── Lal_kakra (100).jpg
                ├── Bruguiera sexangula
                    ├── SK (1).jpg
                    ...
                    ├── SK (176).jpg
                ├── Cerbera manghas
                    ├── 0002.jpg
                    ...
                    ├── vcc.jpg
                ├── Ceriops decandra
                    ├── Goran (1).jpg
                    ...
                    ├── Goran (101).jpg
                ├── Excoecaria agallocha
                    ├── Gewa (1).jpg
                    ...
                    ├── Gewa (60).jpg
                ├── Heritiera fomes
                    ├── 0001.jpg
                    ...
                    ├── 0083.jpg
                ├── Pongamia pinnata
                    ├── koroch (1).jpg
                    ...
                    ├── koroch (124).jpg
                ├── Sonneratia apetala
                    ├── kewra (1).jpg
                    ...
                    ├── kewra (52).jpg
                ├── Xylocarpus moluccensis
                    ├── pasur (1).jpg
                    ...
                    ├── pasur (106).jpg
            ├── readme.csv
        ├── Stomatal traits (ignored)
        ├── Trained models (ignored)
        ├── Stomata ISO19139 Metadata.xml (ignored)
    ├── Processed
        ├── A. cucullata
        ├── P. pinnata
        ├── X. moluccensis
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.dey2023_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.data_dir = os.path.join(self.input_dir, 'Raw data', 'rawdata')  # data directory
        self.source_name = 'Dey2023'  # source name
        self.species_names = []  # to store species names
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.data_dir, os.listdir(self.data_dir), self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store old and new names
        for image_path in get_paths(self.processed_dir, '.jpg'):
            image_basename = os.path.basename(image_path)  # get the image basename
            file_names.append(image_basename)  # populate file names
            for species_name in os.listdir(self.data_dir):
                if species_name in image_path:
                    species_name = self.abbreviate_species(species_name)  # abbreviate species name
                    new_names.append(f'{species_name} {self.source_name} {image_basename}')  # populate the new names
                    self.species_names.append(species_name)  # populate the species names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create species folder
        return None

    def load_xml_bbox(self, xml_file_path: str) -> np.ndarray:
        """Load bbox of stomata from xml annotation files"""
        root, bboxes = ET.parse(xml_file_path).getroot(), []  # to store information from xml annotations
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')  # get each bbox
            xmin = int(bbox.find('xmin').text)  # find the xmin
            ymin = int(bbox.find('ymin').text)  # find the ymin
            xmax = int(bbox.find('xmax').text)  # find the xmax
            ymax = int(bbox.find('ymax').text)  # find the ymax
            bboxes.append((xmin, ymin, xmax, ymax))  # collect bbox to the list
        return np.array(bboxes)

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species_folder in set(os.listdir(self.processed_dir)):
            image_paths = get_paths(os.path.join(self.processed_dir, species_folder), '.jpg')  # get the image paths under the species folder
            for image_path in tqdm(image_paths, total=len(image_paths)):
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.7)  # get the auto labelled masks
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class Ferguson2021(StomataPyData):
    """
    Ferguson et al., 2021  https://doi.org/10.1093/plphys/kiab346
    Dataset source: https://doi.org/10.13012/B2IDB-5565022_V2

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Ferguson2021
    ├── Original
        ├── Accessions_2016_jpg
            ├── 2016
                ├── 16EF0041_2_1_raw.jpg
                ...
                ├── FF1056_2_6_raw.jpg
        ├── Accessions_2017_jpg
            ├── 2017
                ├── 2-17MW0795_#2_1_raw.jpg
                ...
                ├── 17MW0960_#2_4_raw.jpg
    ├── Processed
        ├── S. bicolor
    ├── source.txt
    ├── discard2016.txt
    ├── discard2017.txt

    1. Rename images. As these FOVs are in 393 x 393 (too samll) and great in numbers, we stich them according to naming prefix.
    # [2. Since the most common prefix consists of 8 patches, we stich them with 2 x 4 pattern and save only the stiched images.]
    2. Stiching leads to too small stomata, so we enlarge the image instead
    3. Dsicard unwanted images and their annotations
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.ferguson2021_dir  # directory of Sun2023  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Ferguson2021'  # source name
        self.species_name = 'S. bicolor'  # to store species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    # def extract_prefixes(self, directory: str, folder: Literal['2016', '2017'] = '2016') -> set:
    #     """Extract the the prefixes of the FOVs"""
    #     if folder == '2016':
    #         prefix_pattern = re.compile(r'(EF\d{4})_')  # prefix pattern of 2016
    #     elif folder == '2017':
    #         prefix_pattern = re.compile(r'(17EF\d{4})_')  # prefix pattern of 2017
    #     unique_prefixes = set()   # to store unique prefixes
    #     for filename in os.listdir(directory):
    #         match = prefix_pattern.match(filename[5:])  # if the prefix matches
    #         if match:
    #             unique_prefixes.add(match.group(1))  # populate the unique prefixes
    #     return unique_prefixes

    # def stitch(self, prefix: str, directory: str, rows: int = 2, cols: int = 4) -> None:
    #     """Stich patches in from right to left, from top to bottom"""
    #     patches = [Image.open(os.path.join(directory, f'{prefix}_{x}_{y}_raw.jpg')) for x in range(1, 3) for y in range(1, 5)]  # open all patches to be stitched
    #     total_width = sum(image.size[0] for image in patches[:cols])  # the width of the final stitched image
    #     total_height = sum(image.size[1] for image in patches[::cols])   # the height of the final stitched image
    #     stitched_image = Image.new('RGB', (total_width, total_height))  # create an empty stitched image
    #     for row in range(rows):
    #         for col in range(cols):
    #             index = row * cols + (col if row % 2 == 0 else (cols - 1 - col))  # calculate the index in the zigzag order
    #             x_position = col * patches[index].size[0] if row % 2 == 0 else (cols - col - 1) * patches[index].size[0]  # calculate the x position where the current image should be pasted
    #             y_position = row * patches[index].size[1]  # same for the y position
    #             stitched_image.paste(patches[index], (x_position, y_position))  # paste the current image into the stitched image
    #     save_path = os.path.join(self.processed_dir, f'{self.species_name} {self.source_name} {os.path.basename(directory)} {prefix}_stiched.jpg')  # the path to the stitched image to be saved
    #     stitched_image.save(save_path)  # save the stitched image
    #     return None

    # def stitch_simple(self, directory: str, rows: int = 2, cols: int = 4) -> None:
    #     """Stich patches in from right to left, from top to bottom, , without specific prefixes"""
    #     image_files = [file for file in os.listdir(directory) if file.endswith('_raw.jpg')]
    #     max_stitches = len(image_files) // (rows * cols)  # maximum possible stiched images
    #     for batch in range(max_stitches):
    #         patches = [Image.open(os.path.join(directory, image_files[idx])) for idx in range(batch * rows * cols, (batch + 1) * rows * cols)]  # open patches in a group
    #         total_width = sum(image.size[0] for image in patches[:cols])  # the width of the final stitched image
    #         total_height = sum(image.size[1] for image in patches[::cols])   # the height of the final stitched image
    #         stitched_image = Image.new('RGB', (total_width, total_height))  # create an empty stitched image
    #         for row in range(rows):
    #             for col in range(cols):
    #                 index = row * cols + (col if row % 2 == 0 else (cols - 1 - col))  # calculate the index in the zigzag order
    #                 x_position = col * patches[index].size[0] if row % 2 == 0 else (cols - col - 1) * patches[index].size[0]  # calculate the x position where the current image should be pasted
    #                 y_position = row * patches[index].size[1]  # same for the y position
    #                 stitched_image.paste(patches[index], (x_position, y_position))  # paste the current image into the stitched image
    #         save_path = os.path.join(self.processed_dir, f'{self.species_name} {self.source_name} {os.path.basename(directory)} stiched_{batch + 1}.jpg')  # the path to the stitched image to be saved
    #         stitched_image.save(save_path)  # save the stitched image
    #     return None

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        temp_dir_2016, temp_dir_2017 = os.path.join(self.processed_dir, 'Accessions_2016_jpg'), os.path.join(self.processed_dir, 'Accessions_2017_jpg')  # temporary 2016 and 2017 directories
        self.ensemble_files(os.path.join(self.input_dir, 'Accessions_2016_jpg'), ['2016'], temp_dir_2016, image_types, folder_rename=True)  # move image files to temporary 2016
        self.ensemble_files(os.path.join(self.input_dir, 'Accessions_2017_jpg'), ['2017'], temp_dir_2017, image_types, folder_rename=True)  # move image files to temporary 2017
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard2016.txt'), temp_dir_2016)  # remove unwanted images
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard2017.txt'), temp_dir_2017)  # remove unwanted images
        # prefixes_2016, prefixes_2017 = self.extract_prefixes(temp_dir_2016, '2016'), self.extract_prefixes(temp_dir_2017, '2017')  # FOV prefixes of the 2016 and 2017 folder
        # for prefix in list(prefixes_2016) + list(prefixes_2017):
        #    if '2017' in prefix:
        #        directory = temp_dir_2017  # point to the 2017 directory
        #    else:
        #        directory = temp_dir_2016  # point to the 2016 directory
        #    try:
        #        self.stitch(prefix, directory)  # try to do a 2x4 stiching
        #    except OSError:
        #        pass
        # for directory in [temp_dir_2016, temp_dir_2017]:
        #     self.stitch_simple(directory)  # try to do a 2x4 stiching
        self.ensemble_files(self.processed_dir, os.listdir(self.processed_dir), self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        shutil.rmtree(temp_dir_2016); shutil.rmtree(temp_dir_2017)  # noqa: remove the temporary directories
        file_paths = get_paths(self.processed_dir, '.jpg')  # get image path names
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            height, width = image.shape[:2]  # the original dimensions
            enlarged_image = cv2.resize(image, (width * 4, height * 4), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
            new_basename = f'{self.species_name} {self.source_name} {os.path.basename(file_path)}'  # get new path basename
            os.rename(file_path, os.path.join(self.processed_dir, new_basename))  # rename and move to the speceis folder
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        # print(f"Selected {len(get_paths(self.species_folder_dir, '.jpg'))} stiched images (2x4)!")
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Fetter2019(StomataPyData):
    """
    Fetter et al., 2019 https://doi.org/10.1111/nph.15892
    Dataset source: https://datadryad.org/stash/dataset/doi:10.5061/dryad.kh2gv5f

    Rights and permissions:
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license (https://creativecommons.org/publicdomain/zero/1.0/)

    No Copyright
    The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights
    to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
    You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

    Fetter2019
    ├── Original
        ├── sc_feb2019 (ignored)
        ├── test_set
            ├── cuticle
                ├── cropped_McNair_1950_Platanaceae_Platanus_occidentalis-1.jpg
                ...
                ├── MLC_FLMNH01355SEM_Moraceae_Ficus_angustissima_lwr300x_gray_400dpi.jpg
            ├── ginkgo_test
                ├── RSB_1272_A_1_L_Spot1_200x_82_grid_copy.jpg
                ...
                ├── USNAH_15049_8_1920_L_Spot1_200x_LeafB-98.jpg
            ├── poplar
                ├── Populus_balsamifera_IH_SKN_03_R3_B_A1.jpg
                ...
                ├── Populus_balsamifera_IH_WLK_10_R1_B_A2.jpg
            ├── usnm_test
                ├── Acer_leucoderme_USBG_01_A1.jpg
                ...
                ├── Zephyranthes_clintiae_USBG_01_A6.jpg
        ├── train_set
            ├── cuticle_train
                ├── MLC_FLMNH00212_Lauraceae_Ocotea_killipii_grouped_gray_400dpi.jpg
                ...
                ├── MLC_FLMNH02679_Annonaceae_Cyathostemma_excelsum_grp.jpg
            ├── ginkgo_train
                ├── Balt_RSB_1292_A_L_Spot2_200x_grid copy.jpg
                ...
                ├── USNAH_15049A_8_1920_L_Spot7_200x_grid copy.jpg
            ├── poplar_train
                ├── IH_CBI_02_R1_B_A1.jpg
                ...
                ├── VT_WTR_15_R3_B_A2.jpg
            ├── usnm_train
                ├── Acer_rubrum_USMALL_01_A1.jpg
                ...
                ├── Zizia_aurea_USMALL_01_A3.jpg
    ├── Processed
        ├── A. brachycarpa
        ...
        ├── G. biloba
        ...
        ├── P. balsamifera
        ...
        ├── Z. americanum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.fetter2019_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Fetter2019'  # source name
        self.discard_txt_path = self.input_dir.replace('/Original', 'discard.txt')  # discard.txt path
        self.prefix = f'{self.species} {self.source_name}'  # the prefix for renaming
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration
        self.species_names = ['G. biloba', 'P. balsamifera']  # to collect species names

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""

        def extract_species_cuticle(filename: str, folder: Literal['cuticle', 'usnm'] = 'cuticle') -> str:
            """Extract Genus and Species from the Cuticle Database file names"""
            if folder == 'cuticle':
                pattern = r'cropped_[^_]+_[^_]+_[^_]+_([^_]+)_([^_]+?)(?:-|_|\.jpg)'  # the species name pattern of cuticle images
            elif folder == 'usnm':
                pattern = r'^([^_]+)_([^_]+)'  # the species name pattern of usnm images
            match = re.search(pattern, filename)  # search the pattern for names
            if match:
                genus, species = match.groups()  # collect species names matching the pattern
                return f'{genus} {species}'
            else:
                return 'Unknown Unknown'  # use unknown names if no match found

        train_subfolders = ['cuticle_train', 'ginkgo_train', 'poplar_train', 'usnm_train']  # original subfolders under train_set
        test_subfolders = ['cuticle', 'ginkgo_test', 'poplar', 'usnm_test']  # original subfolders under test_set
        for subfolder in train_subfolders:
            UtilsISAT.copy_folder(os.path.join(self.input_dir, 'train_set', subfolder), os.path.join(self.processed_dir, 'train_set', subfolder))  # paste train subfolders
        for subfolder in test_subfolders:
            UtilsISAT.copy_folder(os.path.join(self.input_dir, 'test_set', subfolder), os.path.join(self.processed_dir, 'test_set', subfolder))  # paste test subfolders
        with open(self.discard_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines = [line.strip() for line in file]  # to store paths of images to be discarded
        for line in lines:
            os.remove(os.path.join(self.processed_dir, line.replace(' ', '//')))  # remove images in the discard.txt file
        for subfolder in train_subfolders:
            UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'train_set', subfolder), os.path.join(self.processed_dir, subfolder))  # paste train subfolders
        for subfolder in test_subfolders:
            UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'test_set', subfolder), os.path.join(self.processed_dir, subfolder))  # paste test subfolders
        shutil.rmtree(os.path.join(self.processed_dir, 'train_set'));  shutil.rmtree(os.path.join(self.processed_dir, 'test_set'))  # noqa: remove train_set and test_set folder
        cuticle_train_dir = os.path.join(self.processed_dir, 'cuticle_train')  # pasted cuticle train directory
        cuticle_train_paths = get_paths(cuticle_train_dir, '.jpg')  # get all image paths under cuticle train directory
        for file_path in cuticle_train_paths:
            with Image.open(file_path) as image:
                width, height = image.size  # get the image width and height
                cropped_image = image.crop((width // 2, 0, width, height))  # crop off the left half
                cropped_image.save(file_path.replace(os.path.basename(file_path), f'cropped_{os.path.basename(file_path)}').replace('cuticle_train', 'cuticle'))  # save the cropped image
        shutil.rmtree(cuticle_train_dir)  # remove the cuticle_train directory since cropped images has been saved to cuticle
        cuticle_paths = get_paths(os.path.join(self.processed_dir, 'cuticle'), '.jpg')  # get all image paths under cuticle directory
        for file_path in cuticle_paths:
            species_name = self.abbreviate_species(extract_species_cuticle(os.path.basename(file_path)))  # extract species name
            if species_name == 'c. grp':
                species_name = 'P. carpinterae'  # 'cropped_MLC_FLMNH00846_Pircfamnia_carpinterae_grp.jpg' does not follow naming pattern
            elif species_name == 'q. grp':
                species_name = 'P. quarternia'  # 'cropped_MLC_FLMNH00844_Pircfomnia_quarternia_grp.jpg' does not follow naming pattern
            elif species_name == 'l. grp':
                species_name = 'P. latifolia'  # 'cropped_MLC_FLMNH00845_Pircfamnia_latifolia_grp.jpg' does not follow naming pattern
            self.species_names.append(species_name)  # to collect the species names
            os.rename(file_path, file_path.replace(os.path.basename(file_path), f'{species_name} {self.source_name} {os.path.basename(file_path)}'))  # unify the name
        ginkgo_dir = os.path.join(self.processed_dir, 'ginkgo')  # combineed ginkgo directory
        UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'ginkgo_test'), ginkgo_dir); shutil.rmtree(os.path.join(self.processed_dir, 'ginkgo_test'))  # noqa: merged and remove the ginkgo_test
        UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'ginkgo_train'), ginkgo_dir); shutil.rmtree(os.path.join(self.processed_dir, 'ginkgo_train'))  # noqa: merged and remove the ginkgo_train
        ginkgo_dir_paths = get_paths(ginkgo_dir, '.jpg')  # get all image paths under ginkgo directory
        for file_path in ginkgo_dir_paths:
            if 'Gingko_biloba' not in os.path.basename(file_path):
                os.remove(file_path)  # remove images of unknown species
                continue
            os.rename(file_path, file_path.replace(os.path.basename(file_path), f'G. biloba {self.source_name} {os.path.basename(file_path)}'))  # unify the name
        poplar_train_dir = os.path.join(self.processed_dir, 'poplar_train')  # poplar_train directory
        poplar_train_paths = get_paths(poplar_train_dir, '.jpg')  # get all image paths under poplar_train directory
        for file_path in poplar_train_paths:
            try:
                os.rename(file_path, file_path.replace(os.path.basename(file_path), f'Populus_balsamifera_{os.path.basename(file_path)}').replace('poplar_train', 'poplar'))  # add Populus_balsamifera_ prefix and merge folder
            except OSError:
                pass  # this happens when Fetter et al., 2019 used the same image for training and testing :)
        shutil.rmtree(poplar_train_dir)  # remove the poplar_train directory
        poplar_paths = get_paths(os.path.join(self.processed_dir, 'poplar'), '.jpg')  # get all image paths under poplar directory
        for file_path in poplar_paths:
            os.rename(file_path, file_path.replace(os.path.basename(file_path), f'P. balsamifera {self.source_name} {os.path.basename(file_path)}'))  # unify the name
        usnm_dir = os.path.join(self.processed_dir, 'usnm')  # combineed usnm directory
        UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'usnm_test'), usnm_dir); shutil.rmtree(os.path.join(self.processed_dir, 'usnm_test'))  # noqa: merged and remove the usnm_test
        UtilsISAT.copy_folder(os.path.join(self.processed_dir, 'usnm_train'), usnm_dir); shutil.rmtree(os.path.join(self.processed_dir, 'usnm_train'))  # noqa: merged and remove the usnm_train
        usnm_paths = get_paths(os.path.join(self.processed_dir, 'usnm'), '.jpg')  # get all image paths under usnm directory
        for file_path in usnm_paths:
            species_name = self.abbreviate_species(extract_species_cuticle(os.path.basename(file_path), folder='usnm'))  # extract species name
            self.species_names.append(species_name)  # to collect the species names
            os.rename(file_path, file_path.replace(os.path.basename(file_path), f'{species_name} {self.source_name} {os.path.basename(file_path)}'))  # unify the name
        subfolders = ['cuticle', 'ginkgo', 'poplar', 'usnm']  # subfolders to be removed
        self.ensemble_files(self.processed_dir, subfolders, self.processed_dir)  # move files to 'Processed' of image files
        for subfolder in subfolders:
            shutil.rmtree(os.path.join(self.processed_dir, subfolder))  # remove these temporary subfolders
        image_numbers = len(get_paths(self.processed_dir, '.jpg'))  # get the total number of images
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create the species folders
        print(f'Selected {image_numbers} images!')  # print out the number of selected images
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        print('Show the progress bar of species folders instead')
        species_folders = os.listdir(self.processed_dir)  # get all species folders
        for folder in tqdm(species_folders, total=len(species_folders)):
            folder_path = os.path.join(self.processed_dir, folder)  # get the species folder path
            image_paths = get_paths(folder_path, '.jpg')  # get all image paths under the species folder
            points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
            for image_path in image_paths:
                image = imread_rgb(image_path)  # load the image in RGB scale
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.1, statistics_filter=False)  # get the auto labelled masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in auto_masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                Anything2ISAT.from_samhq(auto_masks, image, image_path, catergory=catergory)  # export the ISAT json file
        return None


class Jayakody2017(StomataPyData):
    """
    Jayakody et al., 2017 https://doi.org/10.1186/s13007-017-0244-9
    Dataset source: https://github.com/Smart-Robotic-Viticulture/Stomata_Aperture_Measurement_2017

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Jayakody2017
    ├── Original
        ├── raw_dataset_1
            ├── Image_001.jpg
            ...
            ├── Image_110.jpg
        ├── raw_dataset_2
            ├── Image_001.jpg
            ...
            ├── Image_108.jpg
        ├── Test_dataset (ignored)
        ├── Training_dataset
            ├── Negative (ignored)
            ├── Positive
                ├── 01.jpg
                ...
                ├── 555.jpg
    ├── Processed
        ├── V. vinifera
    ├── source.txt
    ├── discard.txt

    1. Combine 'raw_dataset_1' and 'raw_dataset_1', then Copy the images to 'Processed', and rename them with f'V. vinifera Jayakody2017 {orginal file name}'
    2. Dsicard unwanted images
    3. Map the patches from "Positive" back to renamed images to generate json fiels with bounding boxes
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.jayakody2017_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species = 'V. vinifera'  # plant species
        self.output_dir = os.path.join(self.processed_dir, self.species)
        self.source_name = 'Jayakody2017'  # source name
        self.prefix = f'{self.species} {self.source_name}'  # the prefix for renaming
        self.patches_dir = os.path.join(self.input_dir, 'Training_dataset', 'Positive')  # cropped stomata patches dir
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.01}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""
        subfolders = ['raw_dataset_1', 'raw_dataset_2']  # to combine images in these two folders
        for subfolder in subfolders:
            file_names = self.ensemble_files(self.input_dir, [subfolder], self.output_dir)  # move files to 'Processed' of image files
            new_names = [f'{self.prefix} {subfolder} {file_name}' for file_name in file_names]  # get renamings
            self.batch_rename(self.output_dir, file_names, new_names)  # rename
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.output_dir)  # remove unwanted images
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        patch_names = [name for name in os.listdir(self.patches_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # patch image files only
        image_names = [name for name in os.listdir(self.output_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        patch_paths = [os.path.join(self.patches_dir, name) for name in patch_names]  # get patch paths
        image_paths = [os.path.join(self.output_dir, name) for name in image_names]  # get image paths
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image = imread_rgb(image_path)  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt:
                    isat_bboxes = self.template_match(image_path, patch_paths)  # get ISAT format bboxes from template matching
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                print(f'{image_path} cannot be segmented by SAM-HQ')
        return None


class Koheler2023(StomataPyData):
    """
    Koheler et al., 2023  https://doi.org/10.1093/aob/mcac147
    Dataset source: provided by the authors

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Koheler2023
    ├── Original
        ├── All
            ├── 1.tif
            ...
            ├── Image_1402.tif
    ├── Processed
        ├── Z. mays
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.koheler2023_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species_name = 'Z. mays'  # plant species
        self.source_name = 'Koheler2023'  # source name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['All'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for image_path in get_paths(self.processed_dir, '.tif'):
            image_basename = os.path.basename(image_path)  # get the basename
            file_names.append(image_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {image_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.tif')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Koheler2024(StomataPyData):
    """
    Koheler et al., 2024  Manuscript in Preparation
    Dataset source: provided by the authors

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Koheler2024
    ├── Original
        ├── All
            ├── 1_1_1.jpg
            ├── 1_1_1.jpg_metadata.xml
            ...
            ├── size_0.2mm.jpg
            ├── size_2.0um.jpg_metadata.xml
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.koheler2024_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species_name = 'T. aestivum'  # plant species
        self.source_name = 'Koheler2024'  # source name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['All'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for image_path in get_paths(self.processed_dir, '.jpg'):
            image_basename = os.path.basename(image_path)  # get the basename
            file_names.append(image_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {image_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Li2022(StomataPyData):
    """
    Li et al., 2022  https://doi.org/10.1093/plcell/koac021
    Dataset source: https://leafnet.whu.edu.cn/suppdata

    Claimed open source in the publication https://doi.org/10.1093/plcell/koac021
    "
    We believe that the plant community needs more well-labeled datasets,and thus we shared all our training datasets,
    testing datasets, and the results from LeafNet and existing tools in the Download page of the LeafNet web server.
    "

    Li2022
    ├── Original
        ├── F1_training_data
            ├── label
                ├── 0.png
                ...
                ├── 139.png
            ├── sample
                ├── 0.png
                ...
                ├── 139.png
        ├── F2_validation_data
            ├── label
                ├── 0.png
                ...
                ├── 29.png
            ├── sample
                ├── 0.png
                ...
                ├── 29.png
        ├── F5_ABC_ground_truth
            ├── 0.png
                ...
            ├── 13.png
        ├── F5_ABC_raw
            ├── 0.png
                ...
            ├── 13.png
        ├── F5_E_training_data
            ├── label
                ├── 0.png
                ...
                ├── 5.png
            ├── sample
                ├── 0.png
                ...
                ├── 5.png
        ├── F5_FGH_ground_truth
            ├── 0.png
            ├── 1.png
        ├── F5_FGH_raw
            ├── z-projection
                ├── 0.png
                ├── 1.png
            ├── 0.tif (ignored)
            ├── 1.tif (ignored)
        ├── F7_ABCD_raw_data (ignored)
        ├── F7_ABCD_visualized_segmentation (ignored)
        ├── F7_EF_raw_data (ignored)
        ├── F7_EF_visualized_segmentation (ignored)
    ├── Processed
        ├── A. thaliana
        ├── N. tabacum
    ├── source.txt
    ├── discard.txt

    1. Unfiy file structures
    2. Resize F5_ABC_ground_truth from 696 x 520 back to 1392 x 1040
    3. Convert segmentation masks to ISAT with watershed
    4. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.li2022_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Li2022'  # source name
        self.species_names = ['A. thaliana', 'N. tabacum']  # to store species names

    def unify_folders(self) -> None:
        """Unify the images and labels structes for all the folders"""
        f5_adb_dir = os.path.join(self.processed_dir, 'F5_ABC')  # temporary F5_ABC directory
        f5_adb_sample_dir, f5_adb_label_dir = os.path.join(f5_adb_dir, 'sample'), os.path.join(f5_adb_dir, 'label')  # F5_ABC sample and label directories
        shutil.copytree(os.path.join(self.input_dir, 'F5_ABC_raw'), f5_adb_sample_dir)  # copy the F5_ABC label folder
        shutil.copytree(os.path.join(self.input_dir, 'F5_ABC_ground_truth'), f5_adb_label_dir)  # copy the F5_ABC label folder
        f5_fgh_dir = os.path.join(self.processed_dir, 'F5_FGH')  # temporary F5_FGH directory
        f5_fgh_sample_dir, f5_fgh_label_dir = os.path.join(f5_fgh_dir, 'sample'), os.path.join(f5_fgh_dir, 'label')  # F5_FGH sample and label directories
        shutil.copytree(os.path.join(self.input_dir, 'F5_FGH_raw', 'z-projection'), f5_fgh_sample_dir)  # copy the F5_FGH label folder
        shutil.copytree(os.path.join(self.input_dir, 'F5_FGH_ground_truth'), f5_fgh_label_dir)  # copy the F5_FGH label folder
        for folder in ['F1_training_data', 'F2_validation_data', 'F5_E_training_data']:
            source_dir = os.path.join(self.input_dir, folder)  # source directory to be copied
            destination_dir = os.path.join(self.processed_dir, folder)  # destination directory to be pasted
            shutil.copytree(source_dir, destination_dir)  # copy the source directory to the destination directory
        return None

    def label2mask(self, mask_path: str) -> List[np.ndarray]:
        """Convert the segmentation mask of LeafNet to ISAT mask"""
        mask_image = imread_rgb(mask_path)  # load the mask image
        region_masks = []  # to collect masks for each region
        strict_black_mask = (mask_image[:, :, 0] == 0) & (mask_image[:, :, 1] == 0) & (mask_image[:, :, 2] == 0)  # define a mask where all RGB values are zero
        labeled_strict_black, _ = measure.label(strict_black_mask, background=0, return_num=True, connectivity=2)  # label connected components in this strict black mask
        regions = measure.regionprops(labeled_strict_black)  # collect region properties and generate masks
        for region in regions:
            segmentation = labeled_strict_black == region.label  # create a bool mask for the current region
            bbox = UtilsISAT.boolmask2bbox(segmentation)  # get the bbox
            if (bbox[2] - bbox[0] > 1) and (bbox[3] - bbox[1] > 1):
                segmentation_isat = UtilsISAT.mask2segmentation(segmentation)  # convert the bool masks the ISAT format
                points = [[x, y] for x, y in segmentation_isat]  # get the segmentation coordinates
                if len(points) < 5:
                    continue
                mask = {'segmentation': segmentation_isat, 'area': np.sum(segmentation), 'bbox': bbox, 'category': 'pavement cell'}  # get mask information
                region_masks.append(mask)  # collect the bool mask
        blue_only_mask = (mask_image[:, :, 0] == 0) & (mask_image[:, :, 1] == 0) & (mask_image[:, :, 2] > 200)  # for stomata
        labeled_blue_only, _ = measure.label(blue_only_mask, background=0, return_num=True, connectivity=2)  # label stomata
        regions = measure.regionprops(labeled_blue_only)  # collect region properties and generate masks
        for region in regions:
            segmentation = labeled_blue_only == region.label  # create a bool mask for the stomata
            bbox = UtilsISAT.boolmask2bbox(segmentation)  # get the bbox
            if (bbox[2] - bbox[0] > 1) and (bbox[3] - bbox[1] > 1):
                segmentation_isat = UtilsISAT.mask2segmentation(segmentation)  # convert the bool masks the ISAT format
                points = [[x, y] for x, y in segmentation_isat]  # get the segmentation coordinates
                if len(points) < 5:
                    continue
                mask = {'segmentation': segmentation_isat, 'area': np.sum(segmentation), 'bbox': bbox, 'category': 'stoma'}  # get mask information
                region_masks.append(mask)  # collect the bool mask
        return region_masks

    def mask2isat(self) -> None:
        """Convert the original maks to ISAT json format"""
        self.unify_folders()  # unify the file structure
        for subfolder in os.listdir(self.processed_dir):
            mask_folder_dir = os.path.join(self.processed_dir, subfolder, 'label')  # the directory containing the segmentation masks
            mask_paths = get_paths(mask_folder_dir, '.png')  # get the paths of all the masks
            if subfolder == 'F5_ABC':
                for mask_path in mask_paths:
                    with Image.open(mask_path) as image:
                        resized_img = image.resize((1392, 1040), Image.LANCZOS)  # resize F5_ABC_ground_truth from 696 x 520 back to 1392 x 1040
                        resized_img.save(mask_path)  # save change in position
            for mask_path in tqdm(mask_paths, total=len(mask_paths)):
                label_image = imread_rgb(mask_path)  # load the original mask
                masks = self.label2mask(mask_path)  # convert the orginal mask to bool masks
                objects, group_bboxes, layer = [], [], 1.0  # to store information, initialize layer as a floating point number
                group = 0  # initialize group value
                for mask in masks:
                    group += 1  # ensure every mask is in a different group
                    if group == len(group_bboxes) + 1:
                        group_bboxes.append(mask['bbox'])  # populate with bbox
                    objects.append({
                        'category': mask['category'],  # 'pavemt cell' or 'stoma'
                        'group': group,  # group increases if the bbox is not within another
                        'segmentation': mask['segmentation'],  # bool mask to ISAT segmentation
                        'area': int(mask['area']),
                        'layer': layer,  # increment layer for each object
                        'bbox': mask['bbox'].tolist(),  # from np.array to list
                        'iscrowd': False,
                        'note': 'Auto'})
                    layer += 1.0  # increment the layer
                    info = {
                        'description': 'ISAT',
                        'folder': os.path.dirname(mask_path),  # output directory
                        'name': os.path.basename(mask_path),  # image basename
                        'width': label_image.shape[1],  # image width
                        'height': label_image.shape[0],  # image height
                        'depth': label_image.shape[2],  # image depth
                        'note': 'Anomocytic_Peels_Brightfield_MQ_2.2'  # scale bar is obtained from the supplemental Figure S8 (C)
                    }
                    with open(f"{os.path.splitext(mask_path.replace('label', 'sample'))[0]}.json", 'w', encoding='utf-8') as file:
                        json.dump({'info': info, 'objects': objects}, file, indent=4)
        return None

    def rename_files(self) -> None:
        """Rename the images and ISAT annotations"""
        def remove_palette(file_path: str) -> None:
            """Remove the PNG file palette"""
            with Image.open(file_path) as image:
                if image.mode == 'P':  # check if image is in palette mode
                    image = image.convert('RGB')
                    image.save(file_path)  # save the converted image back to the same file path
            return None

        subfolder_dirs = []  # to delete the subfolders later
        for subfolder in os.listdir(self.processed_dir):
            subfolder_dir = os.path.join(self.processed_dir, subfolder)  # get the subfolder dir
            subfolder_dirs.append(subfolder_dir)  # collect the subfolder dir
            self.ensemble_files(subfolder_dir, ['sample'], subfolder_dir, image_types + ['.json'], folder_rename=True)  # move sample image files out
        self.ensemble_files(self.processed_dir, os.listdir(self.processed_dir), self.processed_dir, image_types + ['.json'], folder_rename=True)  # ensemle all the files under

        for subfolder_dir in subfolder_dirs:
            shutil.rmtree(subfolder_dir)  # delete the subfolder directory

        file_names, new_names = [], []  # to store the old and new names
        for file_path in get_paths(self.processed_dir, '.png') + get_paths(self.processed_dir, '.json'):
            file_basename = os.path.basename(file_path)  # get the basename
            file_names.append(file_basename)  # populate the file_names
            if 'F5_ABC' in file_path:
                new_names.append(f'N. tabacum {self.source_name} {file_basename}')  # populate the new names with N. tabacum
            else:
                new_names.append(f'A. thaliana {self.source_name} {file_basename}')  # populate the new names with A. thaliana
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create species folder

        for species_folder in os.listdir(self.processed_dir):
            input_dir = os.path.join(self.processed_dir, species_folder)  # to organize annoations
            png_file_paths = get_paths(input_dir, '.png')  # get the paths of all PNG files
            for png_file_path in png_file_paths:
                remove_palette(png_file_path)  # remove the PNG file palette
            UtilsISAT.quality_check(input_dir)  # check the annotation quality
            UtilsISAT.sort_group(input_dir, if2rgb=False)  # sort categories
            UtilsISAT.shapely_valid_transform(input_dir)  # ensure valid polygons
            if species_folder == 'N. tabacum':
                shutil.rmtree(input_dir)  # the resized ground of N. tabacum resutlting in bad annotation, so we remove it
        return None


class Li2023(StomataPyData):
    """
    Li et al., 2023  https://doi.org/10.1049/ipr2.12617
    Dataset source: https://doi.org/10.5281/zenodo.6302921

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Li2023
    ├── Original
        ├── bean
            ├── Capture_1.jpg
            ...
            ├── Capture_850.jpg
        ├── validation dataset
            ├── ��֤�����·��.txt (ignored)
            ├── 001.jpg
            ...
            ├── 101.jpg
            ├── ������ȡ�ļ���.bat (ignored)
        ├── wheat
            ├── 001.jpg
            ...
            ├── 160.jpg
            ├── ������ȡ�ļ���.bat (ignored)
            ├── С�����ݼ����·��.txt (ignored)
    ├── Processed
        ├── V. faba
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.li2023_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Li2023'  # source name
        self.species_names = ['V. faba', 'T. aestivum']  # to store species names
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.005, 'max_mask_ratio': 0.05}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, os.listdir(self.input_dir), self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for image_path in get_paths(self.processed_dir, '.jpg'):
            image_basename = os.path.basename(image_path)  # get the basename
            file_names.append(image_basename)  # populate the file_names
            if 'bean' in image_path or 'validation dataset' in image_path:
                new_names.append(f'V. faba {self.source_name} {image_basename}')  # populate the new names
            elif 'wheat' in image_path:
                new_names.append(f'T. aestivum {self.source_name} {image_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species_folder in set(os.listdir(self.processed_dir)):
            image_paths = get_paths(os.path.join(self.processed_dir, species_folder), '.jpg')  # get the image paths under the species folder
            for image_path in tqdm(image_paths, total=len(image_paths)):
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class Meeus2020(StomataPyData):
    """
    Meeus et al., 2020  https://doi.org/10.1002/ece3.6571
    Dataset source: https://zenodo.org/records/3579227

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Meeus2020
    ├── Original
        ├── data
            ├── Carapa procera_CBMFO M3-53_leaf1-field1_BR0000013004071.jpg
            ...
            ├── Trilepisium madagascariense_CBMFO E1-41_leaf5-field3_BR0000013005047.jpg
        ├── train
            ├── training
                ├── negative (ignored)
                ├── positive
                    ├── Carapa procera_CBMFO G2-436_leaf1-field1_BR0000013009175_7168.jpg
                    ...
                    ├── Carapa procera_CBMFO M3-53_leaf4-field3_BR0000013004071_383.jpg
            ├── validation
                ├── negative (ignored)
                ├── positive
                    ├── Carapa procera_CBMFO G2-436_leaf1-field2_BR0000013009175_7210.jpg
                    ...
                    ├── Carapa procera_CBMFO M3-53_leaf4-field2_BR0000013004071_312.jpg
            ├── Carapa_procero_demo.jpg
    ├── Processed
        ├── C. chailluana
        ...
        ├── T. madagascariense
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images
    3. Map the patches from "Positive" back to renamed images to generate json fiels with bounding boxes
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.meeus2020_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species_names = []  # plant species
        self.source_name = 'Meeus2020'  # source name
        self.train_patches_dir = os.path.join(self.input_dir, 'train', 'training', 'positive')  # cropped stomata patches dir for training
        self.val_patches_dir = os.path.join(self.input_dir, 'train', 'val', 'positive')  # cropped stomata patches dir for validation
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.01}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""
        def extract_species(filename: str) -> str:
            """Extract Genus and Species from the Cuticle Database file names"""
            pattern = r'([A-Z][a-z]+) ([a-z]+)'   # the species name pattern
            match = re.search(pattern, filename)  # search the pattern for names
            if match:
                genus, species = match.groups()  # collect species names matching the pattern
                return f'{genus} {species}'
            else:
                return 'Unknown Unknown'  # use unknown names if no match found

        new_names = []  # to store new file names
        self.ensemble_files(self.input_dir, ['data'], self.processed_dir)  # move files to 'Processed' of image files
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.jpg')]  # updated file names after discarding
        for file_name in file_names:
            species_name = self.abbreviate_species(extract_species(file_name))  # etract species name from file name
            self.species_names.append(species_name)  # poplulate species names
            new_names.append(f'{species_name} {self.source_name} {file_name}')  # poplulate new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create the species folders
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        print('Show the progress bar of species folders instead')
        train_path_paths = get_paths(self.train_patches_dir, '.jpg')  # get the paths of train patches
        val_path_paths = get_paths(self.val_patches_dir, '.jpg')  # get the paths of val patches
        patch_paths = train_path_paths + val_path_paths  # combine train and val patches paths
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species_name in tqdm(os.listdir(self.processed_dir), total=len(os.listdir(self.processed_dir))):
            species_folder_dir = os.path.join(self.processed_dir, species_name)  # get the path of the species folder
            image_paths = get_paths(species_folder_dir, '.jpg')  # get the image paths under the given species folder
            for image_path in image_paths:
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                    if bbbox_prompt:
                        isat_bboxes = self.template_match(image_path, patch_paths)  # get ISAT format bboxes from template matching
                        if len(isat_bboxes) > 0:
                            prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                            masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                    else:
                        masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class Meng2023(StomataPyData):
    """
    Meng et al., 2023  https://doi.org/10.1007/s00425-023-04231-y
    Dataset source: Kindly provided by Prof. Yoichiro Hoshino via email: hoshino@fsc.hokudai.ac.jp

    Rights and permissions:
    Data use rights granted by corresponding authors

    Meng2023
    ├── Original
        ├── large photos
            ├── 0.jpg
            ...
            ├── 219.jpg
        ├── labels_my-project-name_2023-11-01-09-45-08.json (ignored: only 15 images labeled)
    ├── Processed
        ├── L. caerulea
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.meng2023_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Meng2023'  # source name
        self.species_name = 'L. caerulea'  # to store species name
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['large photos'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for image_path in get_paths(self.processed_dir, '.jpg'):
            image_basename = os.path.basename(image_path)  # get the basename
            file_names.append(image_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {image_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(os.path.join(self.processed_dir, self.species_name), '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Pathoumthong2023(StomataPyData):
    """
    Pathoumthong et al., 2023  https://doi.org/10.1186/s13007-023-01016-y
    Dataset source: https://github.com/rapidmethodstomata/rapidmethodstomata

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Pathoumthong2023
    ├── Original
        ├── Arabidopsis (ignored: no images)
        ├── Rice (ignored: no images)
        ├── Tomato (ignored: no images)
        ├── Wheat
            ├── WheatAnalysedImages (ignored: no images)
            ├── WheatImages
                ── Wheat_Scepter_2_8_W_Ad_X400_20210923.jpg
                ...
                ├── Wheat_Scepter_4_10_W_Ad_X400_20210923.jpg
            ├── WheatSegmentedImages
                ├── J1-7_M_Rep1_P3.jpg-part1.jpg
                ├── J1-7_M_Rep1_P3.jpg-part1.txt
                ...
                ├── J6_stb_IPO89011_Rep5.jpg-part4.jpg
                ├── J6_stb_IPO89011_Rep5.jpg-part4.txt
            ├── Bounding box extraction (ignored)
            ├── Wheat Detection (100×) (ignored)
            ├── Wheat Detection (200×) (ignored)
            ├── Wheat Detection (400×) (ignored)
            ├── Wheat Measurement (400×) (ignored)
            ├── Wheat_detection_100x.pt (ignored)
            ├── Wheat_detection_200x.pt.pt (ignored)
            ├── Wheat_detection_400x.pt.pt (ignored)
            ├── Wheat_measurement.pth (ignored)
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ (did not work), mannually adjust them
    4. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.pathoumthong2023_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.data_dir = os.path.join(self.input_dir, 'Wheat')  # data directory
        self.source_name = 'Pathoumthong2023'  # source name
        self.species_name = 'T. aestivum'  # to store species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.data_dir, ['WheatImages', 'WheatSegmentedImages'], self.processed_dir, ['.jpg', '.txt'], folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for file_path in get_paths(self.processed_dir, '.jpg') + get_paths(self.processed_dir, '.txt'):
            file_basename = os.path.basename(file_path)  # get the basename
            file_names.append(file_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {file_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt and os.path.exists(image_path.replace('.jpg', '.txt')):
                    isat_bboxes = []  # to store bboxes
                    with open(image_path.replace('.jpg', '.txt'), 'r', encoding='utf-8') as file:
                        for line in file:
                            parts = line.strip().split()  # split the line
                            isat_bboxes.append(UtilsISAT.bbox_convert(tuple(map(float, parts[1:5])), 'YOLO2ISAT', image.shape))  # to [x_min, y_min, x_max, y_max]
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Sultana2021(StomataPyData):
    """
    Sultana et al, 2021  https://doi.org/10.3390/plants10122714
    Dataset source: http://stomata.plantprofile.net

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Sultana2021
    ├── Original
        ├── images
            ├── 3.0030.jpg
            ...
            ├── 612.0005.jpg
        ├── labels
            ├── 3.0030.txt
            ...
            ├── 607.1.0241.txt
    ├── Processed
        ├── G. max
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Load the YOLO format bbox annotations as SAM-HQ prompt inputs
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.sultana2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Sultana2021'  # source name
        self.species_name = 'G. max'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.005, 'max_mask_ratio': 0.1}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and YOLO txt files to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['images'], self.processed_dir, image_types)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        self.ensemble_files(self.input_dir, ['labels'], self.processed_dir, ['.txt'])  # move txt files to 'Processed'
        os.makedirs(self.species_folder_dir, exist_ok=True)  # create the species folder
        file_paths = get_paths(self.processed_dir, '.jpg')  # get image paths
        for file_path in file_paths:
            new_basename = f'{self.species_name} {self.source_name} images {os.path.basename(file_path)}'  # get new path basename
            os.rename(file_path, os.path.join(self.species_folder_dir, new_basename))  # rename and move to the speceis folder
            try:
                os.rename(file_path.replace('.jpg', '.txt'), os.path.join(self.species_folder_dir, new_basename).replace('.jpg', '.txt'))  # rename and move to the speceis folder
            except OSError:
                pass  # 183 annotation files vs 386 images so does not match
        remain_txt_paths = get_paths(self.processed_dir, '.txt')  # get the paths of txt files to be removed
        for txt_path in remain_txt_paths:
            os.remove(txt_path)  # remove these txt files
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt and os.path.exists(image_path.replace('.jpg', '.txt')):
                    isat_bboxes = []  # to store bboxes
                    with open(image_path.replace('.jpg', '.txt'), 'r', encoding='utf-8') as file:
                        for line in file:
                            parts = line.strip().split()  # split the line
                            isat_bboxes.append(UtilsISAT.bbox_convert(tuple(map(float, parts[1:5])), 'YOLO2ISAT', image.shape))  # to [x_min, y_min, x_max, y_max]
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Sun2021(StomataPyData):
    """
    Sun et al, 2021  https://doi.org/10.34133/2021/9835961
    Dataset source: https://github.com/shem123456/stomata-segmantic-segmentation

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Sun2021
    ├── Original
        ├── dataset1_408
            ├── jpg
                ├── 0000000.jpg
                ...
                ├── 03078.jpg
            ├── png
                ├── 0000000.png
                ...
                ├── 03078.png
            ├── train.txt
        ├── SBOS (ignored)
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images and resize to 1600 x 1200
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.sun2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Sun2021'  # source name
        self.species_name = 'T. aestivum'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.orginal_masks_dir = os.path.join(self.species_folder_dir, 'original masks')  # to store the original masks under the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.002, 'max_mask_ratio': 0.05}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and masks png files to 'Processed' and rename them"""
        self.ensemble_files(os.path.join(self.input_dir, 'dataset1_408'), ['jpg', 'png'], self.processed_dir, image_types)  # move image files to 'Processed'
        self.ensemble_files(os.path.join(self.input_dir, 'dataset1_408'), ['png'], self.processed_dir, image_types)  # move txt files to 'Processed'
        os.makedirs(self.species_folder_dir, exist_ok=True); os.makedirs(self.orginal_masks_dir, exist_ok=True)  # noqa: create the species folder and a 'original masks' folder within
        file_paths = get_paths(self.processed_dir, '.jpg')  # get image paths
        for file_path in file_paths:
            new_basename = f'{self.species_name} {self.source_name} jpg {os.path.basename(file_path)}'  # get new path basename
            os.rename(file_path, os.path.join(self.species_folder_dir, new_basename))  # rename and move to the speceis folder
            try:
                os.rename(file_path.replace('.jpg', '.png'), os.path.join(self.species_folder_dir, 'original masks', new_basename).replace('.jpg', '.png'))  # rename and move to the speceis folder
            except OSError:
                pass  # 427 annotation files vs 408 images so does not match
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.species_folder_dir)  # remove unwanted images
        image_base_names = [os.path.basename(path) for path in get_paths(self.species_folder_dir, '.jpg')]  # get the base names of images
        mask_paths = get_paths(self.orginal_masks_dir, '.png')  # get mask paths
        for mask_path in mask_paths:
            if os.path.basename(mask_path).replace('.png', '.jpg') in image_base_names:
                continue
            else:
                os.remove(mask_path)  # remove these unwanted mask files
        file_paths = get_paths(self.species_folder_dir, '.jpg')  # get image paths
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            enlarged_image = cv2.resize(image, (1600, 1200), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
        shutil.rmtree(self.orginal_masks_dir)  # remove the orginal masks directory
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Sun2023(StomataPyData):
    """
    Sun et al., 2023  https://doi.org/10.1016/j.compag.2023.108120
    Dataset source: Kindly provided by the PhD student of Prof. Dong Jiang, Zhuangzhuang Sun via email: jiangd@njau.edu.cn

    Rights and permissions:
    Data use rights granted by corresponding authors

    Sun2023
    ├── Original
        ├── stomata
            ├── 0001.jpg
            ├── 0001.xml
            ...
            ├── 02190.jpg
            ├── 02190.xml
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images and resize to 1600 x 1200
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.sun2023_dir  # directory of Sun2023  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Sun2023'  # source name
        self.species_name = 'T. aestivum'  # to store species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['stomata'], self.processed_dir, ['.jpg', '.xml'], folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        for xml_path in get_paths(self.processed_dir, '.xml'):
            if not os.path.exists(xml_path.replace('.xml', '.jpg')):
                os.remove(xml_path)  # remove unwanted xml files
        file_names, new_names = [], []  # to store the old and new names
        for file_path in get_paths(self.processed_dir, '.jpg') + get_paths(self.processed_dir, '.xml'):
            file_basename = os.path.basename(file_path)  # get the basename
            file_names.append(file_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {file_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        file_paths = get_paths(self.species_folder_dir, '.jpg')  # get image paths
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            enlarged_image = cv2.resize(image, (1600, 1200), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
        return None

    def load_xml_bbox(self, xml_file_path: str) -> np.ndarray:
        """Load bbox of stomata from xml annotation files"""
        root, bboxes = ET.parse(xml_file_path).getroot(), []  # to store information from xml annotations
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')  # get each bbox
            xmin = int(bbox.find('xmin').text)  # find the xmin
            ymin = int(bbox.find('ymin').text)  # find the ymin
            xmax = int(bbox.find('xmax').text)  # find the xmax
            ymax = int(bbox.find('ymax').text)  # find the ymax
            bboxes.append((xmin, ymin, xmax, ymax))  # collect bbox to the list
        return np.array(bboxes)

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            xml_path = image_path.replace('.jpg', '.xml')  # the corresponding xml annotation file path
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt and os.path.exists(xml_path):
                    isat_bboxes = self.load_xml_bbox(xml_path)  # try to get the bboxes
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Takagi2023(StomataPyData):
    """
    Takagi et al., 2023  https://doi.org/10.1093/pcp/pcad018
    Dataset source: https://doi.org/10.5281/zenodo.7549842

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Takagi2023
    ├── Original
        ├── microscope_test_images
            ├── 20190507_Disk_Dark_DMSO_18.jpg
            ...
            ├── 20190507_Disk_Light_DMSO_10.jpg
        ├── test_images_with_mask (ignored)
        ├── 221121_micro_seg.onnx (ignored)
        ├── 221121_micro_yolox_s1920.onnx (ignored)
        ├── readme.txt (ignored)
    ├── Processed
        ├── A. thaliana
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.takagi2023_dir  # directory of Sun2023  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Takagi2023'  # source name
        self.species_name = 'A. thaliana'  # to store species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.01}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['microscope_test_images'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for file_path in get_paths(self.processed_dir, '.jpg'):
            file_basename = os.path.basename(file_path)  # get the basename
            file_names.append(file_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {file_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class ThathapalliPrakash2021(StomataPyData):
    """
    Thathapalli Prakash et al., 2021  https://doi.org/10.1093/jxb/erab166
    Dataset source: https://doi.org/10.5061/dryad.crjdfn33z

    Rights and permissions:
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license (https://creativecommons.org/publicdomain/zero/1.0/)

    No Copyright
    The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights
    to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
    You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

    ThathapalliPrakash2021
    ├── Original
        ├── Canopy_temperature_thermal_images (ignored)
        ├── Stomatal_density_leaf_surface_scan_images
            ├── Stomatal scans export images tiff
                ├── 16EF0041_2_1_raw.jpg
                ...
                ├── FF1056_2_6_raw.jpg
            ├── Stomatal scans export images tiff counted (ignored)
            ├── Stomatal scans mnt template files (ignored)
            ├── README.txt (ignored)
        ├── Plot_mean_values_for_above-ground_biomass.xlsx (ignored)
        ├── Plot_mean_values_for_canopy_temperature.xlsx (ignored)
        ├── Plot_mean_values_for_tiller_height__culm_height__and_panicle_emergence_date.xlsx (ignored)
        ├── Stomatal_density_per_field_of_view_of_the_Setaria_abaxial_leaf_surface.xlsx
    ├── Processed
        ├── S. italica
    ├── source.txt
    ├── discard.txt

    1. Resize and rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.thathapalliprakash2021_dir  # directory of Sun2023  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Prakash2021'  # source name
        self.species_name = 'S. italica'  # to store species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (32,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.004}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(os.path.join(self.input_dir, 'Stomatal_density_leaf_surface_scan_images'), ['Stomatal scans export images tiff'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_paths = get_paths(self.processed_dir, '.tif')  # get image path names
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            height, width = image.shape[:2]  # the original dimensions
            enlarged_image = cv2.resize(image, (width * 4, height * 4), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
            new_basename = f'{self.species_name} {self.source_name} {os.path.basename(file_path)}'  # get new path basename
            os.rename(file_path, os.path.join(self.processed_dir, new_basename))  # rename and move to the speceis folder
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.tif')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Toda2018(StomataPyData):
    """
    Toda2018 et al., 2018 https://doi.org/10.1101/365098
    Dataset source: https://github.com/totti0223/deepstomata/tree/master/examples

    Rights and permissions:
    MIT License

    Copyright (c) 2018 yosuke toda

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Toda2018
    ├── Original
        ├── 1.jpg
        ...
        ├── 11.jpg
    ├── Processed
        ├── C. benghalensis
    ├── source.txt
    ├── discard.txt

    1. Rename images with f'C. benghalensis Toda2018 {orginal file name}'
    2. Generate segmentation masks with SAM-HQ; then mannually adjust them
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.toda2018_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species = 'C. benghalensis'  # plant species
        self.output_dir = os.path.join(self.processed_dir, self.species)
        self.source_name = 'Toda2018'  # source name
        self.prefix = f'{self.species} {self.source_name}'  # the prefix for renaming
        self.samhq_configs = {'points_per_side': (12,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images and annotation files to 'Processed' and rename them"""
        file_names = self.ensemble_files(self.input_dir.replace('//Original', ''), ['Original'], self.output_dir)  # move files to 'Processed' of image files
        new_names = [f'{self.prefix} {file_name}' for file_name in file_names]  # get renamings
        self.batch_rename(self.output_dir, file_names, new_names)  # rename
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.output_dir)  # remove low quality images
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        image_names = [name for name in os.listdir(self.output_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        image_paths = [os.path.join(self.output_dir, name) for name in image_names]  # get image paths
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image = imread_rgb(image_path)  # load the image in RGB scale
            auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.1, statistics_filter=False)  # get the auto labelled masks
            if visualize:
                visual_masks = [mask['segmentation'] for mask in auto_masks]  # get only bool masks
                SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
            Anything2ISAT.from_samhq(auto_masks, image, image_path, catergory=catergory)  # export the ISAT json file
        return None


class Toda2021(StomataPyData):
    """
    Toda et al, 2021  https://doi.org/10.3389/fpls.2021.715309
    Dataset source: https://github.com/totti0223/onsite_stomata_platform

    Rights and permissions:
    MIT License

    Copyright (c) 2020 yosuke toda

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Toda2021
    ├── Original
        ├── test
            ├── 20200904_135153__0.jpg
            ...
            ├── 20200904_195414__0.jpg
        ├── trainval
            ├── ckax3cnblosbi0712q9c2bjr6.jpg
            ...
            ├── ckbzuch5y3huu0z5r5533b6ja.jpg
        ├── test_coco.json
        ├── test_labelbox.json (ignored)
        ├── trainval_coco.json (ignored: there is image ids mismatching and redundancy)
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Load the json format bbox annotations as SAM-HQ prompt inputs
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.toda2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Toda2021'  # source name
        self.species_name = 'T. aestivum'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['test', 'trainval'], self.processed_dir, image_types)  # move image files to 'Processed'
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.jpg')]  # get file basenames
        new_names = [f'{self.species_name} {self.source_name} {file_name}' for file_name in file_names]  # get renamings
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        for image_path in get_paths(self.species_folder_dir, '.jpg'):
            gray_image = Image.open(image_path).convert('L')  # open the 8 bit image
            rgb_image = Image.merge("RGB", (gray_image, gray_image, gray_image))  # merge 3 times to RGB so SAM-HQ can work
            rgb_image.save(image_path)  # replace the 8 bit image with synthetic RGB image
        return None

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        def load_coco_bbox(coco_json_path: str) -> dict:
            """Split the COCO json file for many images into json files for each image"""
            with open(coco_json_path, 'r', encoding='utf-8') as file:
                coco_data = json.load(file)  # load the json file
            image_bboxes = {}  # to stor image bboxes
            id_to_filename = {str(image['id']): image['file_name'] for image in coco_data['images']}  # map image ids to image names
            for annotation in coco_data['annotations']:
                image_name = id_to_filename.get(str(annotation['image_id']))  # get the image name
                if image_name:
                    bbox = annotation.get('bbox')  # get the bbox
                    if bbox:
                        if image_name not in image_bboxes:
                            image_bboxes[image_name] = []  # to store all bboxes of the given image
                        image_bboxes[image_name].append(bbox)  # collect these bboxes
            return image_bboxes

        test_bboxes = load_coco_bbox(os.path.join(self.input_dir, 'test_coco.json'))  # load the bboxes from test_coco.json
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                if bbbox_prompt and os.path.exists(image_path.replace('.jpg', '.txt')):
                    isat_bboxes = test_bboxes.get(os.path.basename(image_path), [])  # try to get the bboxes
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Vofely2019(StomataPyData):
    """
    Vőfély et al, 2019  https://doi.org/10.1111/nph.15461
    Dataset source: https://doi.org/10.5061/dryad.g4q6pv3

    Rights and permissions:
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license (https://creativecommons.org/publicdomain/zero/1.0/)

    No Copyright
    The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights
    to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
    You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

    Vofely2019
    ├── Original
        ├── CellCoordinates (ignored)
        ├── CellImages
            ├── ConfocalImages (ignored)
            ├── 3-03-700x-1-01.tif
            ...
            ├── JH15-010_ad_02.tif
            ├── CellMagnifications.txt
            ├── README.txt
        ├── LeafImages (ignored)
        ├── PhylogeneticResources (ignored)
        ├── README_for_CellCoordinates.txt
        ├── README_for_CellImages.txt
        ├── README_for_LeafImages.txt
        ├── README_for_PhylogeneticResources.txt
        ├── README_for_SampleTable.txt
        ├── SampleTable.csv
        ├── Table S1.xlsx
    ── Processed
        ├── A. aurea
        ...
        ├── V. officinalis
    ├── source.txt
    ├── discard.txt

    MICROSCOPE IMAGES README
    There are two different formats for naming the microscope images
    The first convention has a number of numeric fields separated by hyphens.
    Fields 1 and 2 combined determine the species (please see “SampleTable.csv” file)
    Field 3 contains the magnification of the field.
    Field 4 denotes the side of the leaf represented in the picture (i.e. adaxial or abaxial; please see “SampleTable.csv” as 1 and 2 are not consistent)
    Field 5 is an arbitrary numbering of the images from a leaf side.
    In some cases an optional final pain explains that the image is an overview, rather than the image from which measurements were taken.
    Those samples named “9.**.*.*” use the same convention, with decimal separated fields, but lacks the magnification. Instead, the scale is include on the image.

    The second convention are named either “gp##-###” or “JH##-###”.
    Fields 1 and 2, including “gp” or “JH”, indicate the species (see "SampleTable.csv").
    Field 3 appears as either (1) the picture number, (2) the magnification, or (3) whether the image is from the adaxial (ad) or abaxial (ab) side.
    Magnifications for these images are saved in CellMagnifications.txt or are part of file names.

    1. Rename images
    2. Dsicard unwanted / corrupt images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.vofely2019_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.csv_path = os.path.join(self.input_dir, 'SampleTable.csv')  # CSV file path
        self.images_dir = os.path.join(self.input_dir, 'CellImages')  # images directory
        self.source_name = 'Vofely2019'  # source name
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def get_taxonomy(self) -> dict:
        """Get the image file name and plant species from CSV file"""
        dataframe = pd.read_csv(self.csv_path, encoding='ISO-8859-1')  # load the dataframe from the CSV file
        dataframe['Matrix_name'] = dataframe['Matrix_name'].fillna(dataframe['Name_submitted'])  # fill in the empty matrix names with submitted names
        sample_ids = dataframe['Sample'].tolist()  # get sample, as each sample is uniquely mapped to a matrix name
        species_names = dataframe['Matrix_name'].tolist()  # get full species names
        species_names = [self.abbreviate_species(species_name.replace('_', ' ')) for species_name in species_names]  # abbreviate the species names
        return dict(zip(sample_ids, species_names))

    def rename_images(self) -> None:
        """Copy images and txt files to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['CellImages'], self.processed_dir, image_types, folder_rename=True)  # move files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        species_dictionary = self.get_taxonomy()  # get the dictionary of species names and file names
        new_names, species_names = [], []  # the store new names and species names
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.tif') + get_paths(self.processed_dir, '.png')]
        for file_name in file_names:
            species_name = 'Unknown'  # initialize the default value
            for key in species_dictionary:
                if str(key).replace('.', '-') in file_name or str(key) in file_name:
                    species_name = species_dictionary.get(key, 'Unknown')  # get the corresponding species name
            if species_name == 'A. sp.':
                species_name = 'Adromischus sp'
            if species_name == '0 spp':
                species_name = 'C. wilmottianum'
            new_names.append(f'{species_name} {self.source_name} {file_name}')  # get the image renaming
            species_names.append(species_name)
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename
        self.create_species_folders(self.processed_dir, set(species_names))  # group files by plant species
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        print('Show the progress bar of species folders instead')
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species in tqdm(os.listdir(self.processed_dir), total=len(os.listdir(self.processed_dir))):
            species_folder_dir = os.path.join(self.processed_dir, species)  # the species folder directory
            image_paths = get_paths(species_folder_dir, '.tif') + get_paths(species_folder_dir, '.png')  # get the image paths under the species folder
            for image_path in image_paths:
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.7)  # get the auto labelled masks
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class WangRenninger2023(StomataPyData):
    """
    Wang and Renninger, 2023
    Dataset source: https://doi.org/10.5281/zenodo.8271253

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    WangRenninger2023
    ├── Original
        ├── Labeled Stomatal Images
            ├── STMHD0001.jpg
            ├── STMHD0001.txt
            ...
            ├── STMPP3487.jpg
            ├── STMPP3487.txt
        ├── Labeled Stomatal Images.csv
    ├── Processed
        ├── A. rubrum
        ├── D. palustris
        ├── Fraxinus spp
        ├── I. opaca
        ├── Populus spp
        ├── Q. michauxii
        ├── Q. nigra
        ├── Q. pagoda
        ├── Q. phellos
        ├── Q. shumardii
        ├── Q. stellata
        ├── Q. texana
        ├── U. alata
        ├── U. americana
        ├── V. stamineum
    ├── source.txt
    ├── discard.txt

    The annoation txt files are all in YOLO objection detection format {class, x_center, y_center, width, height} (checked):
        For example:
            0 0.025423 0.029762 0.047612 0.053822
            1 0.325424 0.056191 0.077794 0.043736
    where the class code is {0: 'stomata', 1: 'whole_stomata'} meaning {0: 'outer ledge', 1: 'stomata'}

    1. Rename images and resize every image to 2048 x 1536
    2. Dsicard unwanted / corrupt images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.wangrenninger2023_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.csv_path = os.path.join(self.input_dir, 'Labeled Stomatal Images.csv')  # CSV file path
        self.images_dir = os.path.join(self.input_dir, 'Labeled Stomatal Images')  # images and txt annotations files paths
        self.source_name = 'WangRenninger2023'  # source name
        self.class_code_name = {0: 'outer ledge', 1: 'stomata'}  # class code name mapping
        self.corrupt_files = ['STMPP0369.jpg', 'STMPP0387.jpg', 'STMPP0493.jpg',
                              'STMPP0512.jpg', 'STMPP0604.jpg', 'STMPP0880.jpg',
                              'STMPP1029.jpg', 'STMPP1093.jpg', 'STMPP2055.jpg',
                              'STMPP2981.jpg']  # corrupt files
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def get_taxonomy(self) -> dict:
        """Get the image file name and plant species from CSV file"""
        dataframe = pd.read_csv(self.csv_path)  # load the dataframe from the CSV file
        file_names = dataframe['FileName'].tolist()  # get file names
        species_names = dataframe['ScientificName'].tolist()  # get full species names
        species_names = [self.abbreviate_species(species_name) for species_name in species_names]  # abbreviate the species names
        return dict(zip(file_names, species_names))

    def rename_images(self) -> None:
        """Copy images and txt files to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['Labeled Stomatal Images'], self.processed_dir, image_types + ['.txt'], folder_rename=True)  # move files to 'Processed'
        for corrupt_file in self.corrupt_files:
            os.remove(os.path.join(self.processed_dir, f"Labeled Stomatal Images {corrupt_file}"))  # remov the corrupt image
            os.remove(os.path.join(self.processed_dir, f"Labeled Stomatal Images {corrupt_file}".replace('.jpg', '.txt')))  # and their annotations
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        for txt_path in get_paths(self.processed_dir, '.txt'):
            if not os.path.exists(txt_path.replace('.txt', '.jpg')):
                os.remove(txt_path)
        species_dictionary = self.get_taxonomy()  # get the dictionary of species names and file names
        new_names, species_names = [], []  # the store new names and species names
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.jpg') + get_paths(self.processed_dir, '.txt')]
        for file_name in file_names:
            raw_name = os.path.splitext(file_name)[0].replace('Labeled Stomatal Images ', '')  # get the file name without extension
            species_name = species_dictionary.get(raw_name, 'Unknown')  # get the corresponding species name
            new_names.append(f'{species_name} {self.source_name} {file_name}')  # get the image renaming
            species_names.append(species_name)
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename
        file_paths = get_paths(self.processed_dir, '.jpg')  # to resize the image
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            enlarged_image = cv2.resize(image, (2048, 1536), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
        self.create_species_folders(self.processed_dir, set(species_names))  # group files by plant species
        return None

    def get_annotations(self, species_list: list, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species in species_list:
            species_folder_dir = os.path.join(self.processed_dir, species)  # the species folder directory
            image_paths = get_paths(species_folder_dir, '.jpg')  # get the image paths under the species folder
            for image_path in tqdm(image_paths, total=len(image_paths)):
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                    if bbbox_prompt and os.path.exists(image_path.replace('.jpg', '.txt')):
                        isat_bboxes = []  # to store bboxes
                        with open(image_path.replace('.jpg', '.txt'), 'r', encoding='utf-8') as file:
                            for line in file:
                                parts = line.strip().split()  # split the line
                                isat_bboxes.append(UtilsISAT.bbox_convert(tuple(map(float, parts[1:5])), 'YOLO2ISAT', image.shape))  # to [x_min, y_min, x_max, y_max]
                        if len(isat_bboxes) > 0:
                            prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                            masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                    else:
                        masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class Xie2021(StomataPyData):
    """
    Xie et al, 2021  https://doi.org/10.1093/plphys/kiab299
    Dataset source: https://doi.org/10.13012/B2IDB-8275554_V1

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Xie2021
    ├── Original
        ├── 2016RIL_all_detection_result
            ├── 451_leaf1_1.tif.png
            ...
            ├── 650_leaf4_5.tif.png
        ├── 2016RIL_all_mns (ignored)
        ├── 2016RIL_all_TIF
            ├── 451_leaf1_1.tif
            ...
            ├── 650_leaf4_5.tif
        ├── 2017RIL_all_detection_result
            ├── 17RIL_001_L1_1.tif.png
            ...
            ├── 17RIL_408_L2_4.tif.png
        ├── 2017RIL_all_mns (ignored)
        ├── 2017RIL_all_TIF
            ├── 17RIL_001_L1_1.tif
            ...
            ├── 17RIL_408_L2_4.tif
        ├── training data (ignored)
        ├── dataset_info.txt (ignored)
    ├── Processed
        ├── Z. mays
    ├── source.txt
    ├── discard.txt

    Note: the original images are in '.tif' while the model outputs are in '.png'

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Convert the prediction results to ISAT masks
    4. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.xie2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Xie2021'  # source name
        self.species_name = 'Z. mays'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.001, 'max_mask_ratio': 0.04}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['2016RIL_all_TIF', '2017RIL_all_TIF'], self.processed_dir, image_types)  # move image files to 'Processed'
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.tif')]  # get file basenames
        new_names = [f'{self.species_name} {self.source_name} {file_name}' for file_name in file_names]  # get renamings
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        for image_path in get_paths(self.species_folder_dir, '.tif'):
            gray_image = Image.open(image_path).convert('L')  # open the 8 bit image
            rgb_image = Image.merge("RGB", (gray_image, gray_image, gray_image))  # merge 3 times to RGB so SAM-HQ can work
            rgb_image.save(image_path)  # replace the 8 bit image with synthetic RGB
        file_paths = get_paths(self.species_folder_dir, '.tif')  # get the image paths
        for file_path in file_paths:
            image = imread_rgb(file_path)  # load the image for resizing
            enlarged_image = cv2.resize(image, (image.shape[0] * 4, image.shape[1] * 4), interpolation=cv2.INTER_LANCZOS4)  # resize the image
            cv2.imwrite(file_path, cv2.cvtColor(enlarged_image, cv2.COLOR_RGB2BGR))  # save the resized image in position
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        print('Show the progress bar of species folders instead')
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.tif')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.9)  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Yang2021(StomataPyData):
    """
    Yang et al., 2021  https://doi.org/10.1109/TCBB.2021.3137810
    Dataset source: https://github.com/zjxi/stomata-auto-detector/tree/master/sample%20images

    Rights and permissions: Open Access
    GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.


    Yang2021
    ├── Original
        ├── maize_20x
            ├── U-103-1.tif
            ...
            ├── U-103-6.tif
        ├── wheat_10x
            ├── 1.tif
            ...
            ├── 5.tif
    ├── Processed
        ├── T. aestivum
        ├── Z. mays
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.yang2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.source_name = 'Yang2021'  # source name
        self.species_names = ['T. aestivum', 'Z. mays']  # plant species names
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['maize_20x', 'wheat_10x'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        new_names = []  # to store new names
        for image_path in get_paths(self.processed_dir, '.tif'):
            if 'maize' in image_path:
                new_names.append(f"{'Z. mays'} {self.source_name} {os.path.basename(image_path)}")
            elif 'wheat' in image_path:
                new_names.append(f"{'T. aestivums'} {self.source_name} {os.path.basename(image_path)}")
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.tif')]  # get file basenames
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set(self.species_names))  # create species folders
        print(f'Selected {len(new_names)} images!')  # print out the number of selected images
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        for species_name in self.species_names:
            image_paths = get_paths(os.path.join(self.processed_dir, species_name), '.tif')  # get the image paths under the species folder
            for image_path in tqdm(image_paths, total=len(image_paths)):
                image, masks = imread_rgb(image_path), []  # load the image in RGB scale
                try:
                    auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                    if visualize:
                        visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                        SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                    if len(masks) > 0:
                        Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
                except ValueError:
                    pass
        return None


class Yates2018(StomataPyData):
    """
    Yates et al., 2018  https://doi.org/10.1101/490029
    Dataset source: provided by the authors

    Rights and permissions:
    Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/)

    You are free to:
    Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    Adapt — remix, transform, and build upon the material for any purpose, even commercially.
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    Yates2018
    ├── Original
        ├── ZENODO_pictures_2014
            ├── ACHAT_2014_plot391_rep1.jpg
            ...
            ├── ZOBEL_2014_plot128_rep4.jpg
        ├── ZENODO_text_2014
            ├── ACHAT_2014_plot391_rep1
            ...
            ZOBEL_2014_plot128_rep4
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.yates_dir  # original dataset directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.species_name = 'T. aestivum'  # plant species
        self.source_name = 'Yates2018'  # source name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (32,), 'min_mask_ratio': 0.0002, 'max_mask_ratio': 0.001}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.input_dir, ['ZENODO_pictures_2014'], self.processed_dir, image_types, folder_rename=True)  # move image files to 'Processed'
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names, new_names = [], []  # to store the old and new names
        for image_path in get_paths(self.processed_dir, '.jpg'):
            image_basename = os.path.basename(image_path)  # get the basename
            file_names.append(image_basename)  # populate the file_names
            new_names.append(f'{self.species_name} {self.source_name} {image_basename}')  # populate the new names
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label()  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Zhu2021(StomataPyData):
    """
    Zhu et al., 2021   https://doi.org/10.3389/fpls.2021.716784
    Dataset source: https://github.com/WeizhenLiuBioinform/stomatal_index/releases/tag/wheat1.0

    Rights and permissions:
    BSD 3-Clause "New" or "Revised" License (https://github.com/WeizhenLiuBioinform/stomatal_index/blob/master/LICENSE).

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    Zhu2021
    ├── Original
        ├── wheat10x
            ├── images
                ├── 10026.jpg
                ...
                ├── 12783.jpg
        ├── wheat20x (ignored, as they are augmented form the the 10x images)
    ├── Processed
        ├── T. aestivum
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Generate segmentation masks with SAM-HQ, mannually adjust them
    4. Train custom models for auto labeling
    5. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.zhu2021_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.data_dir = os.path.join(self.input_dir, 'wheat10x')  # data directory
        self.source_name = 'Zhu2021'  # source name
        self.species_name = 'T. aestivum'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (24,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.02}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(self.data_dir, ['images'], self.processed_dir, ['.jpg'], folder_rename=True)  # move image files to a temporary subfolder
        # self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.jpg')]  # get file basenames
        new_names = [f'{self.species_name} {self.source_name} wheat10x {file_name}' for file_name in file_names]  # get renamings
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def get_annotations(self, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.2)  # get the auto labelled masks
                masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None


class Liang2022(StomataPyData):
    """
    Liang et al., 2022   https://doi.org/10.1111/pbi.13741
    Dataset source: http://plantphenomics.hzau.edu.cn/download_checkiflogin_en.action

    Rights and permissions:
    Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License (https://creativecommons.org/licenses/by-nc-nd/4.0/).

    You are free to:
    Share — copy and redistribute the material in any medium or format
    The licensor cannot revoke these freedoms as long as you follow the license terms.

    Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made.
    You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    NonCommercial — You may not use the material for commercial purposes.
    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

    2022 Liang et al
    ├── Original
        ├── Labeled images of leaf stomata
            ├── Proscope_maize_stomata_dataset
                ├── Annotations
                    ├── 000001.xml
                    ...
                    ├── 002000.xml
                ├── JPEGImages
                    ├── 000001.jpg
                    ...
                    ├── 002000.jpg
            ├── Proscope_plant_stomata_image (ignored)
            ├── Tipscope_maize_stomata_dataset
                ├── Annotations
                    ├── 000001.xml
                    ...
                    ├── 000499.xml
                ├── JPEGImages
                    ├── 000001.jpg
                    ...
                    ├── 000499.jpg
        ├── Trained model (ignored)
        ├── User guideline (ignored)
    ├── Processed
        ├── Z. mays
    ├── source.txt
    ├── discard.txt

    1. Rename images
    2. Dsicard unwanted images and their annotations
    3. Load the YOLO format bbox annotations as SAM-HQ prompt inputs
    4. Generate segmentation masks with SAM-HQ, mannually adjust them
    5. Train custom models for auto labeling
    6. Check every annotation
    """
    def __init__(self):
        super().__init__()
        self.input_dir = self.liang2022_dir  # input directory
        self.processed_dir = self.input_dir.replace('Original', 'Processed')  # output directory
        self.data_dir = os.path.join(self.input_dir, 'Labeled images of leaf stomata')  # data directory
        self.source_name = 'Liang2022'  # source name
        self.species_name = 'Z. mays'  # plant species name
        self.species_folder_dir = os.path.join(self.processed_dir, self.species_name)  # get the path of the species folder
        self.samhq_configs = {'points_per_side': (12,), 'min_mask_ratio': 0.0005, 'max_mask_ratio': 0.005}  # SAM-HQ auto label configuration

    def rename_images(self) -> None:
        """Copy images to 'Processed' and rename them"""
        self.ensemble_files(os.path.join(self.data_dir, 'Proscope_maize_stomata_dataset'), ['JPEGImages', 'Annotations'], os.path.join(self.processed_dir, 'Proscope_maize_stomata_dataset'), ['.jpg', '.xml'])  # move image files to a temporary subfolder
        self.ensemble_files(os.path.join(self.data_dir, 'Tipscope_maize_stomata_dataset'), ['JPEGImages', 'Annotations'], os.path.join(self.processed_dir, 'Tipscope_maize_stomata_dataset'), ['.jpg', '.xml'])  # move image files to a temporary subfolder
        self.ensemble_files(self.processed_dir, ['Proscope_maize_stomata_dataset', 'Tipscope_maize_stomata_dataset'], self.processed_dir, ['.jpg', '.xml'], folder_rename=True)  # move image files to 'Processed'
        shutil.rmtree(os.path.join(self.processed_dir, 'Proscope_maize_stomata_dataset')); shutil.rmtree(os.path.join(self.processed_dir, 'Tipscope_maize_stomata_dataset'))  # noqa: remove the temporary subfolders
        self.discard_files(os.path.join(self.input_dir.replace('//Original', ''), 'discard.txt'), self.processed_dir)  # remove unwanted images
        for xml_path in get_paths(self.processed_dir, '.xml'):
            if not os.path.exists(xml_path.replace('.xml', '.jpg')):
                os.remove(xml_path)  # remove unwanted annotations
        file_names = [os.path.basename(path) for path in get_paths(self.processed_dir, '.jpg') + get_paths(self.processed_dir, '.xml')]  # get file basenames
        new_names = [f'{self.species_name} {self.source_name} {file_name}' for file_name in file_names]  # get renamings
        self.batch_rename(self.processed_dir, file_names, new_names)  # rename all images
        self.create_species_folders(self.processed_dir, set([self.species_name]))  # create species folder
        return None

    def load_xml_bbox(self, xml_file_path: str) -> np.ndarray:
        """Load bbox of stomata from xml annotation files"""
        root, bboxes = ET.parse(xml_file_path).getroot(), []  # to store information from xml annotations
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')  # get each bbox
            xmin = int(bbox.find('xmin').text)  # find the xmin
            ymin = int(bbox.find('ymin').text)  # find the ymin
            xmax = int(bbox.find('xmax').text)  # find the xmax
            ymax = int(bbox.find('ymax').text)  # find the ymax
            bboxes.append((xmin, ymin, xmax, ymax))  # collect bbox to the list
        return np.array(bboxes)

    def get_annotations(self, bbbox_prompt: bool = True, catergory: str = 'stoma', visualize: bool = False, random_color: bool = True) -> None:
        """Generate ISAT annotations json files"""
        points_per_side, min_mask_ratio, max_mask_ratio = self.samhq_configs['points_per_side'], self.samhq_configs['min_mask_ratio'], self.samhq_configs['max_mask_ratio']  # get SAN-HQ auto mask configs
        image_paths = get_paths(self.species_folder_dir, '.jpg')  # get the image paths under the species folder
        for image_path in tqdm(image_paths, total=len(image_paths)):
            image, masks = imread_rgb(image_path), []  # load the image in RGB scale
            xml_path = image_path.replace('.jpg', '.xml')  # the corresponding xml annotation file path
            try:
                auto_masks = SAMHQ(image_path=image_path, points_per_side=points_per_side, min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio).auto_label(ellipse_threshold=0.7)  # get the auto labelled masks
                if bbbox_prompt and os.path.exists(xml_path):
                    isat_bboxes = self.load_xml_bbox(xml_path)  # try to get the bboxes
                    if len(isat_bboxes) > 0:
                        prompt_masks = SAMHQ(image_path=image_path).prompt_label(input_box=isat_bboxes, mode='multiple')  # get the bbox prompt masks
                        masks = SAMHQ.isolate_masks(prompt_masks + auto_masks)  # filter redundant masks
                else:
                    masks = SAMHQ.isolate_masks(auto_masks)  # filter redundant masks
                if visualize:
                    visual_masks = [mask['segmentation'] for mask in masks]  # get only bool masks
                    SAMHQ.show_masks(image, visual_masks, random_color=random_color)  # visualize bool masks
                if len(masks) > 0:
                    Anything2ISAT.from_samhq(masks, image, image_path, catergory=catergory)  # export the ISAT json file
            except ValueError:
                pass
        return None
