"""Module providing functions interacting with ISAT annotation files"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, no-name-in-module, too-many-function-args, cell-var-from-loop
import os  # interact with the operating system
import shutil  # for copy files
import json  # manipulate json files
import random  # generate random numbers
from typing import Literal, List, Tuple  # to support type hints
import numpy as np  # NumPy
import cv2  # OpenCV
from tqdm import tqdm  # for progress bar
from PIL import Image  # Pillow image processing
from skimage.draw import polygon  # for cellpose outlines
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure  # for image erosion
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection  # to use shapely.geometry
from shapely.validation import explain_validity, make_valid  # explain invalid shapely polygons
from .core import image_types, get_paths  # noqa: import core functions


class UtilsISAT:
    """Groupping utility functions for ISAT annotation files"""
    def __init__(self, annotations_dir: str = None):
        self.annotations_dir = annotations_dir  # annotations directory

    # ############################### For model training ################################
    @staticmethod
    def data_split(images_dir: str, output_dir: str, r_train: float = 0.7, r_test: float = 0.1, seed: int = 42) -> None:
        """
        Split dataset into train and val with defined ratio

        Args:
        - images_dir (str): the input directory containing the original dataset (image and JSON files)
        - output_dir (str): the output directory where 'train' and 'val' subdirectories will be created for the split dataset
        - r_train (float): the ratio of the dataset to be allocated to the training set (default 0.7)
        - r_test (float): the ratio of the dataset to be allocated to the test set (default 0.1)

        Returns:
        - None: Files are copied to 'train', 'val' and 'test' subdirectories under 'output_dir'
        """
        random.seed(seed); np.random.seed(seed)  # noqa: set seed
        file_names = sorted(os.listdir(images_dir), key=str.casefold)
        file_names = [name for name in file_names if any(name.endswith(file_type) for file_type in image_types)]  # image files only
        train_size = int(len(file_names) * r_train)  # training size
        test_size = int(len(file_names) * r_test)  # test size
        validation_size = len(file_names) - train_size - test_size  # validation size
        file_names_shuffle = file_names.copy()  # prevent changing in place
        random.shuffle(file_names_shuffle)  # random shuffle file names
        train_names = file_names_shuffle[:train_size]  # file names for training
        val_names = file_names_shuffle[train_size:train_size + validation_size]
        test_names = file_names_shuffle[train_size + validation_size:]
        print(f'train size={train_size}, validation size={validation_size}, test_size={test_size}')
        destination_train = os.path.join(output_dir, 'train'); os.makedirs(destination_train, exist_ok=True)  # noqa
        destination_val = os.path.join(output_dir, 'val'); os.makedirs(destination_val, exist_ok=True)  # noqa
        destination_test = os.path.join(output_dir, 'test'); os.makedirs(destination_test, exist_ok=True)  # noqa

        for _, file_names, destination in [
            ('train', train_names, destination_train),
            ('val', val_names, destination_val),
            ('test', test_names, destination_test)
        ]:
            for name in file_names:
                source_image = os.path.join(images_dir, name)
                dest_image = os.path.join(destination, name)
                shutil.copy2(source_image, dest_image)  # copy image

                name_json = os.path.splitext(name)[0] + '.json'
                source_json = os.path.join(images_dir, name_json)
                dest_json = os.path.join(destination, name_json)
                if os.path.exists(source_json):
                    shutil.copy2(source_json, dest_json)  # copy JSON files if exist
        return None

    @staticmethod
    def copy_folder(source_folder: str, destination_folder: str) -> None:
        """
        Make a copy of a given folder

        Args:
        - source_folder (str): the directory path of the folder to be copied
        - destination_folder (str): the directory path where the contents of the source folder will be copied to

        Returns:
        - None: this function does not return any value but copies files and directories from source to destination

        Note:
        - This function will overwrite the destination folder if it already exists, ensuring the destination has exactly the same content as the source
        """
        try:
            os.makedirs(destination_folder, exist_ok=True)  # create destination
            shutil.rmtree(destination_folder)  # remove destination if exists
        except shutil.Error as _:  # noqa
            pass
        shutil.copytree(source_folder, destination_folder)  # make the copy
        return None

    @staticmethod
    def sort_group(json_dir: str, if2rgb: bool = False) -> None:
        """
        Sorts and groups objects in JSON files by a predefined category order and group ids.
        Optionally converts associated grayscale images to RGB format.

        Args:
        - json_dir (str): the directory containing JSON files to be processed
        - if2rgb (bool): if True, converts associated grayscale images to RGB. Default is False

        Returns:
        - None: modifies JSON files in place and optionally updates associated image files

        Note:
        - This function updates both the 'objects' array and the 'info' section of each JSON file
        """
        custom_order = ['stomatal complex', 'stoma', 'guard cell', 'outer ledge', 'pore', 'pavement cell']  # the category order
        order_dict = {category: idx for idx, category in enumerate(custom_order)}  # enumerate the order into a dictionary

        def category_sort_order(obj: dict) -> dict:
            """Return the sort order index for the category"""
            return order_dict.get(obj['category'], len(custom_order))

        json_paths = get_paths(json_dir, '.json')  # get the paths of all json files
        for json_path in json_paths:
            with open(json_path, encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            data['objects'].sort(key=category_sort_order)  # cutom sorting group catergories

            # group objects by their original group id
            grouped_objects = {}
            for obj in data['objects']:
                group = obj['group']
                if group not in grouped_objects:
                    grouped_objects[group] = []
                grouped_objects[group].append(obj)

            group_counter, final_sorted_objects, group_mapping = 1, [], {}  # initialized values for sorting
            all_objects = [obj for group in sorted(grouped_objects.keys()) for obj in grouped_objects[group]]  # flatten grouped objects into a list, preserving order

            # remap group ids globally
            for category in custom_order:
                for obj in all_objects:
                    if obj['category'] == category:
                        original_group = obj['group']
                        if original_group not in group_mapping:
                            group_mapping[original_group] = group_counter
                            group_counter += 1
                        obj['group'] = group_mapping[original_group]
                        final_sorted_objects.append(obj)

            data['objects'] = final_sorted_objects  # update sorted objects in the data
            data['objects'].sort(key=lambda obj: (obj['group'], category_sort_order(obj)))  # Sort all objects again by new group id and within each group by category
            layer_counter, current_group = 1, None  # remap layer values
            for obj in data['objects']:
                if obj['group'] != current_group:
                    current_group = obj['group']  # point to the current group
                obj['layer'] = float(layer_counter)  # convert int to float
                layer_counter += 1  # increment layer counter
            data['info']['folder'] = os.path.normpath(os.path.dirname(json_path)).replace('\\', '//')  # use relative directory component of a pathname
            data['info']['name'] = os.path.basename(json_path).replace('.json', os.path.splitext(data['info']['name'])[1])  # in case file name changed
            if if2rgb:
                image_path = json_path.replace('.json', os.path.splitext(data['info']['name'])[1])  # get the image file path
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # load the image
                if len(image.shape) == 2:
                    print(f'converted {image_path} to RGB')
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # to RGB
                    cv2.imwrite(image_path, rgb_image)  # save the converted image
                data['info']['depth'] = 3  # RGB image depth
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)  # save the sorted json data
        return None

    @staticmethod
    def select_class(json_dir: str = None,
                     category: str = 'pore',
                     action: Literal['remove', 'select', 'remove groups', 'rename class'] = 'remove',
                     source_class: str = 'stoma',
                     destination_class: str = 'stomatal complex') -> None:
        """
        Modifies object categories in JSON files based on the specified action

        Args:
        - json_dir (str): the directory containing JSON files to be processed
        - category (str): the category to act upon. Default is 'pore'.
        - action (Literal['remove', 'select', 'remove groups', 'rename class']): The action to perform. Default is 'remove'
            -'remove': remove all the annotations of a given category
            -'select': select all the annotations of a given category
            -'remove groups': remove all the group a given annotation belongs to
            -'rename class': rename a given category to a new name. e.g. 'stoma' -> 'stomatal complex'
        'rename class' args:
            - source_class (str): the original class to be renamed if action is 'rename class'. Default is 'stoma'
            - destination_class (str): the new class name if action is 'rename class'. Default is 'stomatal complex'

        Returns:
        - None: The function writes modifications directly to the original JSON files.
        """
        json_paths = get_paths(json_dir, '.json')  # get the paths of all json files
        for json_path in json_paths:
            with open(json_path, encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            if action == 'remove':
                data['objects'] = [obj for obj in data['objects'] if obj['category'] != category]  # remove the category
            elif action == 'select':
                data['objects'] = [obj for obj in data['objects'] if obj['category'] == category]  # select the category
            elif action == 'remove groups':
                groups = []
                for obj in data['objects']:
                    if obj['category'] != category:
                        groups.append(obj['group'])  # get the group of the current object
                data['objects'] = [obj for obj in data['objects'] if obj['group'] in groups]  # select objects within the groups
            elif action == 'rename class':
                destination_groups = {obj['group'] for obj in data['objects'] if obj['category'] == destination_class}  # to track destination class presence in groups
                new_objects = []  # to store the final objects
                for obj in data['objects']:
                    if obj['group'] not in destination_groups and obj['category'] == source_class:
                        obj['category'] = destination_class
                    new_objects.append(obj)
                data['objects'] = new_objects  # Updated list after processing
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)  # save the sorted json data
        return None

    @staticmethod
    def group_annotated(json_dir: str) -> None:
        """
        Moves annotated JSON files and their associated images to 'Annotated' or 'Discard' subdirectories based on annotation content
        The function checks the 'note' field in each JSON. Files with 'note' marked as 'discard' are moved to 'Discard', others with note to 'Annotated'

        Args:
        - json_dir (str): the directory containing JSON files to be processed

        Returns:
        - None: moves files within the directory structure based on their annotations
        """
        json_paths = get_paths(json_dir, '.json')  # get the paths of all json files
        for json_path in json_paths:
            with open(json_path, encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            if data['info']['note'] == '':
                continue  # only move checked ISAT json files, whose 'note' should not be empty
            elif data['info']['note'].lower() == 'discard' or len(data['objects']) == 0:
                annotated_dir = os.path.join(json_dir, 'Discard')  # directory containing discarded files
            else:
                annotated_dir = os.path.join(json_dir, 'Annotated')  # directory containing annotated files
            if not os.path.exists(annotated_dir):
                os.makedirs(annotated_dir)  # create annotated_dir if not exist
            image_source_path = os.path.join(json_dir, data['info']['name'])  # image source path
            image_destination_path = os.path.join(annotated_dir, data['info']['name'])  # image destination path
            shutil.move(image_source_path, image_destination_path)  # move the image to annotated directory
            shutil.move(json_path, os.path.join(annotated_dir, os.path.basename(json_path)))  # move the json file to annotated directory
        return None

    @staticmethod
    def resize_isat(input_dir: str, new_width: int, new_height: int, if_keep_ratio: bool = True) -> None:
        """
        Resize ISAT json files and images in a given folder.
        This function maintains aspect ratio, resizes the long edge to fit within the specified dimensions, and pads the shorter edge to meet exactly the new_width and new_height.

        Args:
        - input_dir (str): the directory containing images and JSON files.
        - new_width (int): the target new width for the images.
        - new_height (int): the target new height for the images.
        - if_keep_ratio (bool): If True, maintains the aspect ratio, resizes the long edge, and pads the other dimension.
        """

        json_paths = get_paths(input_dir, '.json')  # get the paths of all json files
        print('Resizing ISAT json files...')
        for json_path in tqdm(json_paths, total=len(json_paths)):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            image_path = json_path.replace('.json', os.path.splitext(data['info']['name'])[1])  # get the image path
            image = Image.open(image_path)  # open the image
            original_width, original_height = image.size  # the original width and height

            # calculate new dimensions and padding
            if if_keep_ratio:
                ratio = min(new_width / original_width, new_height / original_height)  # get the closest ratio to destination dimension
                resize_width, resize_height = int(original_width * ratio), int(original_height * ratio)  # resize width and height according to the ratio
            else:
                resize_width, resize_height = new_width, new_height  # resize width and height to destination dimensions

            image = image.resize((resize_width, resize_height), Image.LANCZOS)  # resize the image
            new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # create a black image in target dimension
            padding_horizontal, padding_vertical = (new_width - resize_width) // 2, (new_height - resize_height) // 2  # calculate padding values
            new_image.paste(image, (padding_horizontal, padding_vertical))  # pad the image to center
            new_image.save(image_path)  # save the resized and padded image

            data['info']['width'], data['info']['height'] = new_width, new_height  # update json info
            width_ratio, height_ratio = resize_width / original_width, resize_height / original_height  # the resize ratio

            for obj in data['objects']:
                for point in obj['segmentation']:
                    point[0] = point[0] * width_ratio + padding_horizontal  # update x coordinate
                    point[1] = point[1] * height_ratio + padding_vertical  # update y coordinate

                # update the bbox
                obj['bbox'][0] = int(obj['bbox'][0] * width_ratio) + padding_horizontal
                obj['bbox'][1] = int(obj['bbox'][1] * height_ratio) + padding_vertical
                obj['bbox'][2] = int(obj['bbox'][2] * width_ratio) + padding_horizontal
                obj['bbox'][3] = int(obj['bbox'][3] * height_ratio) + padding_vertical

            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)  # save the updated json file
        return None

    @staticmethod
    def shapely_valid_transform(input_dir: str = None, if_explain_invalid: bool = False) -> None:
        """
        Transforms the polygons in JSON files to be valid for further processing with libraries such as Sahi and Shapely

        Args:
        - input_dir (str): the directory containing JSON files with polygon data for processing; defaults to None
        - if_explain_invalid (bool): if True, prints explanations for why polygons are invalid; defaults to False

        Returns:
        - None: modifies JSON files in place, updating polygons to valid forms
        """
        json_paths = get_paths(input_dir, '.json')  # get all json file paths
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data

            for obj in data['objects']:
                corrected_segmentation = []  # to collect valid segmentation points
                min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')  # ensure updates will be replaced by real numbers

                for point in obj['segmentation']:
                    x, y = point  # get the ISAT segmentation coordinates
                    if int(x) == 0:
                        x += 1  # if the point locates on the left side, shift 1 pixel inward
                    elif int(x) == data['info']['width'] - 1:
                        x -= 1  # if the point locates on the right side, shift 1 pixel inward
                    if int(y) == 0:
                        y += 1  # if the point locates on the bottom, shift 1 pixel inward
                    elif int(y) == data['info']['height'] - 1:
                        y -= 1  # if the points locates on the top, shift 1 pixel inward
                    corrected_segmentation.append([x, y])  # collect the corrected segmentation coordinates
                    min_x, max_x = min(min_x, x), max(max_x, x)  # get the min/max x to update bbox
                    min_y, max_y = min(min_y, y), max(max_y, y)  # same for y

                polygon_shapely = Polygon(corrected_segmentation)  # noqa: convert the ISAT segmentation points to shapely format
                if not polygon_shapely.is_valid:
                    if if_explain_invalid:
                        invalid_reason = explain_validity(polygon_shapely)  # the reason for invalidity
                        print(f"{obj['category']} of group {{obj['group']}} {os.path.basename(json_path)} is invalid: {invalid_reason}")  # print the reason for invalidity
                    corrected_geometry = make_valid(polygon_shapely)
                    if isinstance(corrected_geometry, MultiPolygon):
                        corrected_geometry = max(corrected_geometry.geoms, key=lambda polygon: polygon.area)  # selecting the largest polygon
                    elif isinstance(corrected_geometry, GeometryCollection):
                        polygons = [geom for geom in corrected_geometry.geoms if isinstance(geom, Polygon)]  # filter out only polygonal components
                        if polygons:
                            corrected_geometry = max(polygons, key=lambda polygon: polygon.area)  # selecting the largest polygon
                        else:
                            continue  # if there are no polygons, use the original corrected_segmentation
                    corrected_segmentation = [[pt[0], pt[1]] for pt in corrected_geometry.exterior.coords[:-1]]  # update segmentation from the largest or corrected polygon

                obj['segmentation'] = corrected_segmentation  # update segmentation coordinates
                obj['bbox'] = [min_x, min_y, max_x, max_y]  # update the bbox
                obj['iscrowd'] = False  # sahi does not support iscrowd=True

            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)  # save the updated json data
        return None

    # ############################### For data engine ################################
    @staticmethod
    def quality_check(input_dir: str = None, remove_small: bool = True) -> None:
        """
        Checks JSON files for redundancy in object annotations within the same groups
        Specifically highlighting 'background' classes and multiple instances of non-'epidermal cell' categories within the same group

        Args:
        - input_dir (str): the directory containing JSON files to be checked for redundancy; defaults to None
        - remove_small (bool): if True, removes entire groups containing objects with area < 40; defaults to True

        Returns:
        - None: outputs to console potential issues with object categorization within groups
        """
        json_paths = get_paths(input_dir, '.json')  # get all json file paths
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            objects = data.get('objects', [])  # get the objects information
            grouped_objects = {}  # to store the grouped objects
            small_groups = set()  # to store groups with small objects
            for obj in objects:
                area = obj.get('area', None)  # get the area
                group = obj.get('group', 0)  # get the group of the object, default to 0
                if group not in grouped_objects:
                    grouped_objects[group] = []  # to store the group
                grouped_objects[group].append(obj)  # add group to the group list
                if area < 40:
                    print(f'object area < 40 in group {group} in {json_path}')  # if small area
                    if remove_small:
                        small_groups.add(group)  # collect groups with small objects

            for group, objs in grouped_objects.items():
                categories = set()  # to store the categories
                for obj in objs:
                    category = obj.get('category')  # get the catergory of the given object
                    if 'background' in category:
                        print(f'background class found in {json_path}')  # if background is labeled
                    elif category != 'epidermal cell' and category in categories:
                        print(f"Redundant category '{category}' in group {group} in {json_path}")  # if there is redundant non-background objects
                    categories.add(category)  # to count number of catergory apperance

            for obj in [obj for obj in objects if obj['category'] == 'epidermal cell']:
                groups = set()  # to store the groups
                group = obj.get('group', 0)  # get the epidermal cell group
                if group in groups:
                    print(f"Redundant epidermal cell in group {group} in {json_path}")  # ff there is redundant epidermal cells
                groups.add(group)  # to find redundant epidermal cells

            if remove_small and small_groups:
                data['objects'] = [obj for obj in objects if obj.get('group', 0) not in small_groups]
                with open(json_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)
        return None

    @staticmethod
    def mask2segmentation(mask_bool: np.ndarray, use_polydp: bool = False, epsilon_factor: float = 0.002) -> List[List[float]]:
        """
        Converts a boolean mask to ISAT-compatible segmentation points
        This function takes a boolean mask, finds contours using OpenCV, and optionally applies contour smoothing using polynomial approximation
        The largest contour is converted into a list of segmentation points suitable for ISAT format

        Args:
        - mask_bool (np.ndarray): a boolean mask array where true values represent the object
        - use_polydp (bool): whether to use polynomial approximation to smooth the contour; defaults to False
        - epsilon_factor (float): a factor used to determine the approximation accuracy; smaller values lead to contours closer to the original; defaults to 0.002

        Returns:
        - segmentation (List[List[float]]): a list of segmentation points, where each inner list contains two floats [[x1, y1], [x2, y2], ..., [xn, yn]]
        """
        mask_uint8 = mask_bool.astype(np.uint8) * 255  # bool mask to uint8
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # find mask contours. Optional: using cv2.CHAIN_APPROX_SIMPLE for smoothing
        largest_contour = max(contours, key=cv2.contourArea)  # get the largest contour only
        if use_polydp:  # approximates the contour for smoothing effect
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)  # samller value leads to a contour closer to the original
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)  # approximates the shape of the contour.
            coordinates = approx_contour.squeeze().tolist()  # smoothed contour coorinates
        else:
            coordinates = largest_contour.squeeze().tolist()  # contour coorinates
        segmentation = [[float(point[0]), float(point[1])] for point in coordinates]  # get the ISAT format segmentation
        return segmentation

    @staticmethod
    def segmentation2mask(segmentation: List[List[float]], image_shape: tuple) -> np.ndarray:
        """
        Converts ISAT-compatible segmentation points back into a boolean mask
        This function takes a list of segmentation points and image dimensions, draws the polygon on a blank mask, and returns it

        Args:
        - segmentation (List[List[float]]): A list of segmentation points, where each inner list contains two floats representing x and y coordinates
        - image_shape (tuple): The shape (height, width) of the image to which the mask will apply

        Returns:
        - mask (np.ndarray): A boolean mask array where True values represent the object
        """
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)  # create a blank mask with the given image dimensions
        pts = np.array(segmentation, np.int32).reshape((-1, 1, 2))  # convert segmentation points into a format suitable for drawing the polygon in OpenCV
        cv2.fillPoly(mask, [pts], 255)  # draw the polygon on the mask
        return mask.astype(bool)

    @staticmethod
    def coco_mask2isat_mask(coco_segmentation: list) -> list:
        """Convert COCO segmentation to ISAT segmentation"""
        return [[float(coco_segmentation[idx]), float(coco_segmentation[idx + 1])] for idx in range(0, len(coco_segmentation), 2)]

    @staticmethod
    def mask2bbox(mask_bool: np.ndarray) -> List[int]:
        """
        Calculates the bounding box (bbox) for a boolean mask as per ISAT format
        This function identifies all true (or '1') values within a boolean mask array to determine the minimum and maximum x and y coordinates, which define the bounding box of the mask

        Args:
        - mask_bool (np.ndarray): the boolean mask array where true values represent the object

        Returns:
        - bbox (List[int]): a list containing four integers [xmin, ymin, xmax, ymax] representing the bounding box coordinates
        """
        y_indices, x_indices = np.where(mask_bool)  # find the indices of all true values in the mask
        xmin, xmax, ymin, ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)  # calculate xmin, ymin, xmax, ymax
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    @staticmethod
    def pad_bbox(bbox: np.ndarray, padding: int, max_width: int, max_height: int, allow_negative_crop: bool = False) -> np.ndarray:
        """
        Expand the bbox by a specified padding while ensuring it stays within the image boundaries unless negative cropping is allowed.

        Args:
        - bbox (np.ndarray): the original bbox as a 1D array [x_min, y_min, x_max, y_max]
        - padding (int): the amount of padding to add to each side of the bbox
        - max_width (int): the maximum allowable width (image width)
        - max_height (int): the maximum allowable height (image height)
        - allow_negative_crop (bool): whether to allow cropping outside of the image boundaries

        Returns:
        - padded_bbox (np.ndarray): the padded bbox as a 1D array [x_min_padded, y_min_padded, x_max_padded, y_max_padded]
        """
        if allow_negative_crop:  # crop outside of the image boundaries, for object dimensions
            x_min_padded, x_max_padded = bbox[0] - padding, bbox[2] + padding
            y_min_padded, y_max_padded = bbox[1] - padding, bbox[3] + padding
        else:  # to restore back the back to the original image
            x_min_padded, x_max_padded = max(bbox[0] - padding, 0), min(bbox[2] + padding, max_width)
            y_min_padded, y_max_padded = max(bbox[1] - padding, 0), min(bbox[3] + padding, max_height)
        return np.array([x_min_padded, y_min_padded, x_max_padded, y_max_padded], dtype=np.int32)

    @staticmethod
    def crop_image_with_padding(image: np.ndarray, bbox: np.ndarray, padding: int, allow_negative_crop: bool = False, pad_value: int = 0) -> np.ndarray:
        """
        Crop image patches based on provided bounding boxes with added padding. This function can conditionally allow for negative cropping.

        Args:
        - image (np.ndarray): the image from which patches are to be cropped (height, width, channels)
        - bbox (np.ndarray): a 2D array of bbox in [x_min, y_min, x_max, y_max]
        - padding (int): the amount of padding to add around the bbox
        - allow_negative_crop (bool): whether to allow cropping outside of the original image bounds
        - pad_value (int): the value to use for padding outside the original image bounds

        Returns:
        - crop (np.ndarray): a cropped image patch as a NumPy array
        """
        max_height, max_width = image.shape[:2]  # get the image dimensions
        padded_bbox = UtilsISAT.pad_bbox(bbox, padding, max_width, max_height, allow_negative_crop)  # get the padded bbox

        if allow_negative_crop:
            pad_width = int(max(padding, -min(padded_bbox[0], padded_bbox[1], 0)))  # get the padding width
            padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'constant', constant_values=pad_value)  # pad the image to ensure we can crop outside the original bounds
            padded_bbox = padded_bbox + pad_width  # adjust the bounding box coordinates for the added padding
        else:
            padded_image = image  # nomarl mode
        crop = padded_image[padded_bbox[1]:padded_bbox[3], padded_bbox[0]:padded_bbox[2]]  # crop the image using the adjusted (and potentially padded) bounding box
        return crop

    @staticmethod
    def bbox_convert(bbox: np.ndarray, flag: Literal['COCO2ISAT', 'ISAT2COCO', 'YOLO2ISAT'] = 'COCO2ISAT', image_shape: tuple = None) -> np.ndarray:
        """
        Converts bounding box coordinates between different formats (COCO, ISAT, and YOLO)
        This function adjusts bounding box dimensions based on the format specified by the 'flag'. It can convert from MSCOCO to ISAT, ISAT to MSCOCO, and YOLO to ISAT, accommodating the peculiarities of each format

        Args:
        - bbox (np.ndarray): the bounding box coordinates to be converted
        - flag (Literal['COCO2ISAT', 'ISAT2COCO', 'YOLO2ISAT']): specifies the conversion type; defaults to 'COCO2ISAT'
        - image_shape (tuple): dimensions of the image (height, width), required for YOLO to ISAT conversion; defaults to None

        Returns:
        - bbox (np.ndarray): an array of converted bounding box coordinates
        """
        if flag == 'COCO2ISAT':
            x_min, y_min, width, height = bbox  # bbox in MSCOCO format
            x_max, y_max = x_min + width, y_min + height  # get xmax and ymax
            return np.array([x_min, y_min, x_max, y_max])
        elif flag == 'ISAT2COCO':
            x_min, y_min, x_max, y_max = bbox  # bbox in ISAT format
            width, height = x_max - x_min, y_max - y_min  # get the width and height
            return np.array([x_min, y_min, width, height])
        elif flag == 'YOLO2ISAT':
            image_width, image_height = image_shape[1], image_shape[0]  # get image dimensions as YOLO format is normalized
            x_center, y_center, width, height = bbox  # bbox in YOLO format (normalized)
            x_center, y_center = x_center * image_width, y_center * image_height
            width, height = width * image_width, height * image_height  # de-normalize the width and height
            x_min, y_min = max(0, x_center - (width / 2)), max(0, y_center - (height / 2))  # get the xmin and ymin
            x_max, y_max = min(image_width, x_center + (width / 2)), min(image_height, y_center + (height / 2))  # get the xmax and ymax
            return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])

    @staticmethod
    def bbox_within(bbox_1: np.ndarray, bbox_2: np.ndarray) -> bool:
        """
        Checks if the first bounding box (bbox_1) is completely within the second bounding box (bbox_2) based on ISAT format coordinates
        This function compares two bounding boxes to determine if bbox_1 is entirely contained within bbox_2. Both bounding boxes should be in the format [xmin, ymin, xmax, ymax]

        Args:
        - bbox_1 (np.ndarray): the first bounding box array in ISAT format
        - bbox_2 (np.ndarray): the second bounding box array in ISAT format

        Returns:
        - bool: returns True if bbox_1 is within bbox_2, otherwise False
        """
        return all(bbox_1[idx] >= bbox_2[idx] for idx in [0, 1]) and all(bbox_1[idx] <= bbox_2[idx] for idx in [2, 3])

    @staticmethod
    def bbox_intersection(bbox_1: list, bbox_2: list, threshold: float = 0.2) -> bool:
        """
        Checks if the intersection between two bounding boxes is at least a certain threshold percentage of the smaller box's area
        This function calculates the intersection of two bounding boxes and returns True if the intersection area is at least [threshold] times the area of the smaller bounding box.

        Args:
        - bbox_1 (list): the first bounding box as a list of coordinates [xmin, ymin, xmax, ymax]
        - bbox_2 (list): the second bounding box as a list of coordinates [xmin, ymin, xmax, ymax]
        - threshold (float): the minimum fraction of the smaller bounding box's area that must be covered by the intersection; defaults to 0.2

        Returns:
        - bool: returns True if the intersection meets the threshold requirement, otherwise False
        """
        bbox_1, bbox_2 = np.array(bbox_1), np.array(bbox_2)  # convert bounding boxes to np.arrays
        x_left, x_right = np.maximum(bbox_1[0], bbox_2[0]), np.minimum(bbox_1[2], bbox_2[2])  # get x coordinates
        y_top, y_bottom = np.maximum(bbox_1[1], bbox_2[1]), np.minimum(bbox_1[3], bbox_2[3])  # get y coordinates
        if x_right < x_left or y_bottom < y_top:
            return False  # check for overlap
        intersection_area = (x_right - x_left) * (y_bottom - y_top)  # intersection area
        bbox_1_area, bbox_2_area = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1]), (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])  # bboxes areas
        smaller_box_area = np.minimum(bbox_1_area, bbox_2_area)  # determine the smaller box area
        return (intersection_area / smaller_box_area) >= threshold if smaller_box_area > 0 else False

    @staticmethod
    def isat_bboxes_filter(bboxes: list, threshold: float = 0.5) -> list:
        """
        Filters out smaller ISAT format bounding boxes that overlap significantly with larger ones based on a specified overlap threshold
        This function evaluates a list of bounding boxes and removes those smaller bounding boxes that overlap by more than the specified threshold with a larger bounding box
        Overlap is calculated based on the area of intersection divided by the area of the smaller box

        Args:
        - bboxes (list): a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        - threshold (float): the minimum fraction of the smaller bounding box's area that must be overlapped for it to be removed; defaults to 0.5

        Returns:
        - final_bboxes (list): a list of the remaining bounding boxes after filtering
        """
        bboxes = np.array(bboxes)  # convert list to np.array
        if bboxes.ndim == 2 and bboxes.shape[1] == 4:
            to_remove = set()  # initialize the set of bboxes to be removed
            widths = bboxes[:, 2] - bboxes[:, 0]  # calculate the width
            heights = bboxes[:, 3] - bboxes[:, 1]  # calculate the height
            areas = widths * heights  # calculate area for all bounding boxes
            for idx, _ in enumerate(bboxes):
                if idx in to_remove:
                    continue
                for idx_j in range(idx + 1, len(bboxes)):
                    if idx_j in to_remove:
                        continue
                    bbox_1 = bboxes[idx]  # the first bbox
                    bbox_2 = bboxes[idx_j]  # the second bbox
                    x_overlap_start = max(bbox_1[0], bbox_2[0])  # xmax
                    x_overlap_end = min(bbox_1[2], bbox_2[2])  # xmin
                    y_overlap_start = max(bbox_1[1], bbox_2[1])  # ymax
                    y_overlap_end = min(bbox_1[3], bbox_2[3])  # ymin
                    overlap_width = x_overlap_end - x_overlap_start  # overlapping width
                    overlap_height = y_overlap_end - y_overlap_start  # overlapping height
                    intersection = max(0, overlap_width) * max(0, overlap_height)
                    area1, area2 = areas[idx], areas[idx_j]  # get areas of both bounding boxes
                    if intersection > threshold * min(area1, area2):
                        smaller_bbox_idx = idx if area1 < area2 else idx_j  # if the intersection is larger than threshold * smaller bbox
                        to_remove.add(smaller_bbox_idx)  # remove the smaller bbox
            final_bboxes = np.delete(bboxes, list(to_remove), axis=0).tolist()  # final bboxes list
            return final_bboxes
        else:
            return []

    @staticmethod
    def shrink_bbox(bbox: np.ndarray, shrink: float) -> np.ndarray:
        """
        Shrinks the ISAT bounding box by a specified factor while keeping the center of the box fixed
        This function takes a bounding box and reduces its dimensions (width and height) by a given shrink factor but ensures that the center of the box remains unchanged

        Args:
        - bbox (np.ndarray): the original bounding box as an array with coordinates [xmin, ymin, xmax, ymax]
        - shrink (float): the factor by which the bounding box dimensions should be reduced; must be less than 1 for shrinking

        Returns:
        - shrinked_bbox (np.ndarray): an array with the new bounding box coordinates after shrinking
        """
        x_min, y_min, x_max, y_max = bbox  # flatten the bbox
        center_x = (x_max + x_min) / 2.0  # calculate the center of x
        center_y = (y_max + y_min) / 2.0  # calculate the center of y
        new_width = (x_max - x_min) * shrink  # calculate the new width after shrinking
        new_height = (y_max - y_min) * shrink  # same for the new height
        new_x_min = center_x - (new_width / 2)  # new xmin after shrinking
        new_y_min = center_y - (new_height / 2)  # new ymin
        new_x_max = center_x + (new_width / 2)  # new xmax
        new_y_max = center_y + (new_height / 2)  # new ymax
        return np.array([new_x_min, new_y_min, new_x_max, new_y_max])

    @staticmethod
    def boolmask2bbox(bool_mask: np.ndarray) -> np.ndarray:
        """
        Calculates the bounding box coordinates (xmin, ymin, xmax, ymax) for a given boolean mask
        This function determines the outermost true values in a boolean mask to identify the rectangular bounding box that encompasses all the true values.

        Args:
        - bool_mask (np.ndarray): a 2D numpy array where True values indicate the presence of the object

        Returns:
        - bbox (np.ndarray): an array containing the coordinates of the bounding box [xmin, ymin, xmax, ymax]
        """
        rows = np.any(bool_mask, axis=1)  # find rows where the mask is True
        cols = np.any(bool_mask, axis=0)  # find columns where the mask is True
        ymin, ymax = np.where(rows)[0][[0, -1]]  # find the min and max of these rows
        xmin, xmax = np.where(cols)[0][[0, -1]]  # find the min and max of thesecolumns
        return np.array((xmin, ymin, xmax, ymax))

    @staticmethod
    def ellipse_filter(mask: np.ndarray, threshold: float = 0.9) -> Tuple[bool, float]:
        """
        Fits an ellipse to the largest contour of a mask and filters the mask based on the Intersection Over Union (IOU) with the fitted ellipse
        This function takes a binary mask, identifies the largest contour within the mask, fits an ellipse to this contour, and evaluates the IOU between the mask of the contour and the mask of the ellipse
        If the IOU is below a given threshold, the function indicates that the ellipse is not a good fit

        Args:
        - mask (np.ndarray): a binary mask array where the object is represented by True (1) values
        - threshold (float): the minimum IOU for the ellipse to be considered a good fit; defaults to 0.9

        Returns:
        - if_elipses (bool): indicates if the ellipse is a good fit
        - elipse_score (float): represents the calculated IOU value to the ellipse approximation
        """
        mask_uint8 = mask.astype(np.uint8) * 255  # bool mask to uint8
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # find mask contours
        largest_contour = max(contours, key=cv2.contourArea)  # get the largest contour only
        if len(largest_contour) < 5:
            return False, 0  # there should be at least 5 points to fit the ellipse
        else:
            ellipse = cv2.fitEllipse(largest_contour)  # fit ellipse
            major_axis, minor_axis = ellipse[1]  # get the major axis and minor axis of the ellipse
            if major_axis > minor_axis * 3:
                return False, 0  # the ellipe is too narrow
            mask_ellipse, mask_contour = np.zeros(mask_uint8.shape, np.uint8), np.zeros(mask_uint8.shape, np.uint8)  # create empty balck masks
            cv2.drawContours(mask_contour, [largest_contour], -1, 255, -1)  # contour mask
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)  # ellipse mask
            intersection = np.logical_and(mask_contour, mask_ellipse)  # calculate ellipse and contour intersections
            union = np.logical_or(mask_contour, mask_ellipse)  # calculate the union
            iou = np.sum(intersection) / np.sum(union)  # calculate iou
            if iou < threshold:
                return False, 0
            else:
                return True, iou

    @staticmethod
    def objects_filter(json_dir: str, ellipse_threshold: float = 0.9, area_threshold: float = 200) -> None:
        """
        Filters out objects in JSON files based on elliptical fit and area size criteria
        This function reads JSON files containing object data, applies an ellipse fit filter, and checks if the object's area exceeds a specified threshold
        Objects that do not meet the ellipse fit or area criteria are removed from the JSON file

        Args:
        - json_dir (str): the directory containing the JSON files
        - ellipse_threshold (float): the minimum Intersection Over Union (IOU) with the fitted ellipse to consider an object as elliptical enough; defaults to 0.9
        - area_threshold (float): the minimum area required for an object to be retained; defaults to 200

        Returns:
        - None: the function modifies the JSON files in place and does not return any value
        """
        json_paths = get_paths(json_dir)  # get all json file paths under the directory
        for json_path in tqdm(json_paths, total=len(json_paths)):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            new_objects = []  # to store the new objects
            for obj in data['objects']:
                mask = np.zeros((data['info'].get('height'), data['info'].get('width')), dtype=np.uint8)  # create an empty mask with the same shape as the original image
                contour = np.array(obj['segmentation'], dtype=np.int32).reshape((-1, 1, 2))  # convert segmentation to a format suitable for cv2.fillPoly
                cv2.fillPoly(mask, [contour], 255)  # fill the contour to create the mask
                mask_bool = mask.astype(bool)  # convert to boolean
                if UtilsISAT.ellipse_filter(mask_bool, threshold=ellipse_threshold)[0] and obj['area'] > area_threshold:
                    new_objects.append(obj)  # add the object fullfill the requirements
            data['objects'] = new_objects  # update objects list
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)  # save the updated json file
        return None


class Anything2ISAT:
    """
    Convert any annotation format to ISAT json format

    The ISAT (Image Segmentation Annotation Tool) format provides a structured approach for representing image annotations
    File Naming: Each image has a corresponding .json file named after the image file (without the image extension)

    ['info']: Contains metadata about the dataset and image
        ['description']: Always 'ISAT'
        ['folder']: The directory where the images are stored
        ['name']: The name of the image file
        ['width'], ['height'], ['depth']: The dimensions of the image; depth is assumed to be 3 for RGB images
        ['note']: An optional field for any additional notes related to the image

   ['objects']: Lists all the annotated objects in the image
        ['category']: The class label of the object. If the category_id from MSCOCO does not have a corresponding entry, 'unknown' is used
        ['group']: An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number. Group numbering starts at 1
        ['segmentation']: A list of [x, y] coordinates forming the polygon around the object, e.g. [[x1, y1], [x2, y2], ..., [xn, yn]]
        ['area']: The area covered by the object in pixels
        ['layer']: A float indicating the sequence of the object. It increments within the same group, starting at 1.0
        ['bbox']: The bounding box coordinates in the format [x_min, y_min, x_max, y_max]
        ['iscrowd']: A boolean value indicating if the object is part of a crowd

    Required input:
    - image_dir: a folder that stores all images
    - annotations_dir: a folder that stores the annotations file(s)

    Output:
    ISAT format json files in under images_dir
    """
    def __init__(self,
                 images_dir: str = None,
                 annotations_dir: str = None,
                 output_dir: str = None):
        self.images_dir = images_dir  # images directory
        self.annotations_dir = annotations_dir  # annotations directory
        self.output_dir = output_dir  # ISAT json output directory
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)  # create output directory

    def isat_area(self, coordinates: List[Tuple[float, float]]) -> float:
        """
        Calculate the area of a polygon given its vertices
        The function uses the Shoelace formula (Gauss's area formula) to compute the area of a polygon whose vertices are provided as a list of [x, y] coordinate pairs
        The vertices must be listed in sequential order, either clockwise or counterclockwise

        Args:
        - coordinates (List[Tuple[float, float]]): a list where each element is a tuple of two floats representing the x and y coordinates of a vertex

        Returns:
        - area (float): the area of the polygon
        """
        coordinates = np.array(coordinates)  # convert coordinates to np.array if not already
        x, y = coordinates[:, 0], coordinates[:, 1]  # get the x, y coordinates
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def seg2isat(info_dict: dict, objects_list: list, output_filename: str) -> None:
        """
        Converts segmentation results into ISAT JSON format
        This function takes a dictionary containing image information, a list of objects each described in a dictionary format, and an output filename

        Args:
        - info_dict (dict): dictionary containing metadata about the image such as description, width, height, etc
        - objects_list (list): list of dictionaries, each representing an object with details like category, bbox, etc
        - output_filename (str): the filename where the JSON data will be saved

        Returns:
        - None: the function writes data to a JSON file and does not return any value
        """
        data = {
            "info": info_dict,
            "objects": objects_list
        }
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        return None

    @staticmethod
    def create_empty_json(image_path: str) -> None:
        """
        Create empty file in ISAT JSON format

        Args:
        - images_path (str): the image path

        Returns:
        - None: the function writes data to a JSON file and does not return any value
        """
        json_path = f'{os.path.splitext(image_path)[0]}.json'  # get the ISAT json file path
        image = cv2.imread(image_path)  # load the image
        isat_info = {
            'description': 'ISAT',  # must be exactly 'ISAT'
            'folder': os.path.dirname(image_path),  # image parent directory
            'name': os.path.basename(image_path),  # image base name
            'width': image.shape[1],  # image width
            'height': image.shape[0],  # image height
            'depth': image.shape[2],  # image depth
            'note': ''
        }
        isat_data = {
            'info': isat_info,  # information section regarding the image
            'objects': []  # objects section regarding segmentation masks
        }
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(isat_data, file, indent=4)  # save the json file
        return None

    def from_yolo_seg(self, class_dictionary: dict = None) -> None:
        """
        Get the objects information form YOLO segmentation txt file
            Key differences:
            1. 'segmentation'
                YOLO: [x1,  y1, x2, y2, ..., xn, yn]
                ISAT: [[x1, y1], [x2, y2], ..., [xn, yn]]
            3. layer
            4. group

        Args:
        - class_dictionary (dict): the dictionary mapping class indices to class names, used for labeling objects in the ISAT format

        Returns:
        - None: writes the output JSON files to the specified directory, does not return any value
        """
        def yolo2isat_segmentation(yolo_seg: list, img_width: int, img_height: int) -> list:
            """Convert YOLO segmentation format to ISAT segmentation format"""
            return [[round(x * img_width), round(y * img_height)] for x, y in zip(yolo_seg[::2], yolo_seg[1::2])]

        def get_isat_bbox(segmentation: list) -> list:
            """Calculate the bbox from the ISAT segmentation"""
            xs = [point[0] for point in segmentation]  # x-coordinates
            ys = [point[1] for point in segmentation]  # y-coordinates
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        image_names = [name for name in os.listdir(self.images_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]  # get all image names
        image_paths = [os.path.join(self.images_dir, name) for name in image_names]  # get all image paths
        annotation_names = [name for name in os.listdir(self.annotations_dir) if any(name.lower().endswith(file_type) for file_type in ['txt'])]  # get annotation file names
        annotation_paths = [os.path.join(self.annotations_dir, name) for name in annotation_names]  # get annotation paths
        for idx, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)  # load the image in BRG scale
            image_width, image_height = image.shape[1], image.shape[0]  # get the image dimensions
            isat_objects = []  # to store objects
            group_bboxes, layer = [], 1.0  # initialize layer as a floating point number
            with open(annotation_paths[idx], 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.split()  # split each line
                    class_index = int(parts[0])  # get the class index
                    yolo_segmentation = list(map(float, parts[1:]))  # get the yolo_segmentation
                    isat_segmentation = yolo2isat_segmentation(yolo_segmentation, image_width, image_height)  # convert yolo_segmentation to isat_segmentation
                    bbox = get_isat_bbox(isat_segmentation)  # calculate the bbox from segmentation
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # roughly calculate the bbox area as segmentation area, it will be replaced anyway
                    group = next((idx for idx, group_bbox in enumerate(group_bboxes, 1) if UtilsISAT.bbox_within(bbox, group_bbox)), len(group_bboxes) + 1)  # get the group idex
                    if group == len(group_bboxes) + 1:
                        group_bboxes.append(bbox)  # populate the group bboxes
                    isat_objects.append({
                        'category': class_dictionary.get(class_index, 'unknown'),  # for example {0: 'class 0', 1: 'class 1'}
                        'group': group,  # group increases if the bbox is not within another
                        'segmentation': isat_segmentation,
                        'area': area,  # segmentation area
                        'layer': layer,  # increment layer for each object
                        'bbox': bbox,  # bbox
                        'iscrowd': False,
                        'note': ''
                    })
                    layer += 1.0  # increment the layer
            isat_info = {
                'description': 'ISAT',  # must be exactly 'ISAT'
                'folder': self.images_dir,  # image parent directory
                'name': os.path.basename(image_path),  # image base name
                'width': image_width,  # image width
                'height': image_height,  # image height
                'depth': image.shape[2],  # image depth
                'note': ''
            }
            isat_data = {
                'info': isat_info,  # information section regarding the image
                'objects': isat_objects  # objects section regarding segmentation masks
            }
            isat_filename = os.path.splitext(os.path.basename(image_path))[0] + '.json'  # output json file name
            isat_file_path = os.path.join(self.output_dir, isat_filename)  # output COCO file path
            with open(isat_file_path, 'w', encoding='utf-8') as file:
                json.dump(isat_data, file, indent=4)  # save the json file
        return None

    def from_coco(self) -> None:
        """
        Get the objects information form COCO json file
            Key differences:
            1. 'segmentation'
                COCO: [x1, y1, x2, y2, ..., xn, yn]
                ISAT: [[x1, y1], [x2, y2], ..., [xn, yn]]
            2. 'bbox'
                COCO: [xmin, ymin, width, height]
                ISAT: [xmin, ymin, xmax, ymax]]
            3. layer
            4. group
        """
        def coco2isat_segmentation(coco_segmentation: list) -> list:
            """Convert COCO segmentation to ISAT segmentation"""
            return [[float(coco_segmentation[idx]), float(coco_segmentation[idx + 1])] for idx in range(0, len(coco_segmentation), 2)]

        def coco2isat(coco_data: dict) -> list:
            category_mapping = {category['id']: category['name'] for category in coco_data['categories']}  # map category id to category name
            isat_data_list = []  # to collect information for all images
            for image_info in coco_data['images']:
                isat_data = {
                    'info': {
                        'description': 'ISAT',  # must be exactly 'ISAT'
                        'folder': self.images_dir,  # image parent directory
                        'name': os.path.splitext(image_info['file_name'])[0],  # image path without extension
                        'width': image_info['width'],  # image width
                        'height': image_info['height'],  # image height
                        'depth': 3,  # assuming RGB
                        'note': ''},
                    'objects': []}
                annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_info['id']]  # check only the given image
                annotations.sort(key=lambda x: -x['area'])  # larger objects first
                group_counter = 1  # group starts from 1
                for annotation in annotations:
                    bbox = [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]]  # xmin, ymin, xmax, ymax
                    group_id = next((obj['group'] for obj in isat_data['objects'] if UtilsISAT.bbox_within(bbox, obj['bbox'])), group_counter)  # group id increases if bbox not within another
                    if group_id == group_counter:
                        group_counter += 1  # new group found
                    layer = sum(obj['group'] == group_id for obj in isat_data['objects']) + 1  # layer inrease by 1
                    isat_object = {
                        'category': category_mapping.get(annotation['category_id'], 'unknown'),  # get the corresponding category name
                        'group': group_id,
                        'segmentation': coco2isat_segmentation(annotation['segmentation'][0]),  # segmentation in [[x1, y1], [x2, y2], ...]
                        'area': annotation['area'],
                        'layer': float(layer),  # 1.0, 2.0, 3.0, ...
                        'bbox': [int(coord) for coord in bbox],  # to integer
                        'iscrowd': annotation['iscrowd'],  # same as in MSCOCO
                        'note': annotation.get('note', '')  # same as in MSCOCO
                    }
                    isat_data['objects'].append(isat_object)  # collect all objects within the given image
                isat_data_list.append(isat_data)  # collect all objects of all images
            return isat_data_list

        json_name = [name for name in os.listdir(self.annotations_dir) if any(name.endswith(file_type) for file_type in ['.json'])][0]  # assuming only one json file
        coco_json_path = os.path.join(self.annotations_dir, json_name)  # get the COCO json file path
        with open(coco_json_path, 'r', encoding='utf-8') as file:
            coco_data = json.load(file)  # load the COCO json file
            isat_datasets = coco2isat(coco_data)  # get ISAT format inforation
            for isat_data in isat_datasets:
                isat_filename = isat_data['info']['name'] + '.json'  # output json file name
                isat_file_path = os.path.join(self.output_dir, isat_filename)  # output COCO file path
                with open(isat_file_path, 'w', encoding='utf-8') as file:
                    json.dump(isat_data, file, indent=4)  # save the json file
        return None

    @staticmethod
    def from_samhq(masks: list, image: np.ndarray, image_path: str, catergory: str = 'stoma', if_remove_overlapping_masks: bool = True) -> None:
        """
        Converts a list of SAM-HQ segmentation masks to ISAT JSON format
        The function processes each mask, determining its group based on overlap with existing groups, and formats each mask's data into ISAT's JSON structure

        Args:
        - masks (list): a list of dictionaries, each containing a mask's data including its segmentation and bounding box
        - image (np.ndarray): the image array corresponding to the masks
        - image_path (str): the path to the image file, used to derive the output file path and other metadata
        - category (str): the category label for all objects; defaults to 'stoma'

        Returns:
        - None: saves a JSON file in ISAT format containing the image and objects information
        """
        def remove_overlapping_masks(masks: List[dict]) -> List[dict]:
            """
            Eliminates overlapping masks, keeping only the largest mask in cases of overlap

            Args:
            - masks (List[Dict[str, Any]]): List of masks containing segmentation data and other properties

            Returns:
            - List[Dict[str, Any]]: List of filtered masks with overlapping smaller masks removed
            """
            def check_overlap(poly1: Polygon, poly2: Polygon) -> bool:
                """Check if two polygons overlap"""
                return poly1.intersects(poly2)

            polygons, keep = [], []  # to collect polygons
            for idx, mask in enumerate(masks):
                poly = Polygon(mask['segmentation'])  # convert the segmentation coordinates to shapley polygon
                if not poly.is_valid:
                    poly = make_valid(poly)  # attempt to fix the polygon
                polygons.append((poly, mask['area'], idx))  # collect the corrected polygons
            polygons.sort(key=lambda x: x[1], reverse=True)  # sort polygons by area in descending order

            for poly_1, area_1, idx_1 in polygons:
                overlap = False  # initialize not overlapping as value
                for poly_2, _, _ in keep:
                    if check_overlap(poly_1, poly_2):
                        overlap = True; break  # noqa: if two polygon overlap
                if not overlap:
                    keep.append((poly_1, area_1, idx_1))  # collect the polygons to be kept

            keep_indices = [idx for _, _, idx in keep]  # get the indices of the polygons to keep
            filtered_masks = [masks[idx] for idx in keep_indices]  # filter the masks to keep only the largest non-overlapping ones
            return filtered_masks

        objects, group_bboxes, layer = [], [], 1.0  # to store information, initialize layer as a floating point number
        for mask in masks:
            group = next((idx for idx, group_bbox in enumerate(group_bboxes, 1) if UtilsISAT.bbox_within(mask['bbox'], group_bbox)), len(group_bboxes) + 1)  # iterate through all existing groups and their corresponding bboxes
            if group == len(group_bboxes) + 1:
                group_bboxes.append(mask['bbox'])  # populate with bbox
            objects.append({
                'category': catergory,  # for example {0: 'class 0', 1: 'class 1'}
                'group': group,  # group increases if the bbox is not within another
                'segmentation': UtilsISAT.mask2segmentation(mask['segmentation']),  # bool mask to ISAT segmentation
                'area': int(mask['area']),
                'layer': layer,  # increment layer for each object
                'bbox': mask['bbox'].tolist(),  # from np.array to list
                'iscrowd': False,
                'note': 'Auto'})
            if if_remove_overlapping_masks:
                objects = remove_overlapping_masks(objects)  # remove the overlapping polygons (inlcusion)
            layer += 1.0  # increment the layer
            info = {
                'description': 'ISAT',
                'folder': os.path.dirname(image_path),  # output directory
                'name': os.path.basename(image_path),  # image basename
                'width': image.shape[1],  # image width
                'height': image.shape[0],  # image height
                'depth': image.shape[2],  # image depth
                'note': ''}
            with open(f'{os.path.splitext(image_path)[0]}.json', 'w', encoding='utf-8') as file:
                json.dump({'info': info, 'objects': objects}, file, indent=4)
        return None

    def from_cellpose(self, category: str = 'stoma', use_polydp: bool = False, epsilon_factor: float = 0.002) -> None:
        """
        Incorporates a Cellpose npy file segmentation into a category of an ISAT json file
        This method processes Cellpose-generated npy files, extracting segmentation masks, and integrating them into corresponding ISAT-formatted JSON files

        Args:
        - category (str): the category label for all objects; defaults to 'stoma'
        - use_polydp (bool): specifies whether to use polynomial approximation to smooth the segmentation; defaults to False
        - epsilon_factor (float): a factor used to determine the approximation accuracy when use_polydp is True; defaults to 0.002

        Returns:
        - None: updates ISAT formatted JSON files with segmentation data from Cellpose npy files
        """
        def populate_objects(data: dict, mask_bool: np.ndarray) -> dict:
            """Populate the existing ISAT json file with Cellpose label. Assign the object to an existing group if bboxes intersect"""
            if len(data['objects']) > 0:
                max_group = max(obj['group'] for obj in data['objects'])  # get the max group
                max_layer = max(obj['layer'] for obj in data['objects'])  # get the max layer
            else:
                max_group, max_layer = 0, 0.0  # if there is no object at all
            segmentation = UtilsISAT.mask2segmentation(mask_bool, use_polydp, epsilon_factor)  # get the ISAT segmentation
            bbox = UtilsISAT.mask2bbox(mask_bool)  # get the ISAT bbox ndarray
            group_assigned = False   # whether the group is assigned
            for obj in data['objects']:
                if UtilsISAT.bbox_intersection(bbox, obj['bbox'], threshold=0.5):
                    group_assigned = True  # assign the same group as the intersect object
                    if obj['category'] == category:  # same object: just replace the segmentation
                        obj['segmentation'] = segmentation  # update the segmentation of the existing object
                        obj['area'] = int(np.sum(mask_bool))  # update the area based on boolean mask
                        obj['layer'] = max_layer + 1.0  # increase layer by 1.0
                        obj['bbox'] = bbox  # update the bbox
                        obj['note'] = 'Auto'  # add Cellpose tag
                    else:
                        new_object_same_group = {
                            'category': category,  # category as user input
                            'group': obj['group'],  # assign to the same group
                            'area': int(np.sum(mask_bool)),  # get the area based on boolean mask
                            'segmentation': segmentation,  # the ISAT segmentation
                            'layer': max_layer + 1.0,  # increase layer by 1.0
                            'bbox': bbox,  # the ISAT bbox
                            'iscrowd': False,
                            'note': 'Auto'}
                        data['objects'].append(new_object_same_group)  # update the objects section
            if not group_assigned:
                new_object = {
                    'category': category,  # category as user input
                    'group': max_group + 1,  # assign to new group
                    'area': int(np.sum(mask_bool)),  # get the area based on boolean mask
                    'segmentation': segmentation,  # the ISAT segmentation
                    'layer': max_layer + 1.0,  # increase layer by 1.0
                    'bbox': bbox,  # the ISAT bbox
                    'iscrowd': False,
                    'note': 'Auto'}
                data['objects'].append(new_object)  # update the objects section
            return data

        cellpose_npy_paths = get_paths(self.annotations_dir, '.npy')  # get the paths of all Cellpose npy files
        for cellpose_npy_path in cellpose_npy_paths:
            json_path = cellpose_npy_path.replace('_seg.npy', '.json')  # get the ISAT json file path
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            if data['info']['note'] != '':
                continue  # only convert checked ISAT json files
            cellpose_npy = np.load(cellpose_npy_path, allow_pickle=True)  # load the Cellpose npy file
            cellpose_masks = cellpose_npy.item()['masks']   # get the Cellpose segmentation masks
            unique_instances = np.unique(cellpose_masks)  # get unique instances
            unique_instances = unique_instances[unique_instances > 0]   # remove the background
            for instance_id in unique_instances:
                mask_bool = cellpose_masks == instance_id  # select the unique instance
                data = populate_objects(data, mask_bool)  # populate the objects setction
            #  print(json_path)
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file)  # save updated json file
        return None

    def from_openmmlab(self, valid_predictions: List[dict]) -> None:
        """
        Convert the mmdetection or mmsegmentation results to an ISAT json file

        Args:
        - valid_predictions (list): a list of dictionaries, each containing a prediction's data including its segmentation and bounding box

        Returns:
        - None: updates ISAT formatted JSON files with segmentation data from Cellpose npy files
        """
        def populate_objects(data: dict, valid_prediction: dict, idx: int) -> dict:
            """
            Populate the existing ISAT json file with valid_prediction
            Assign the object to an existing group if bboxes intersect
            """
            if len(data['objects']) > 0:
                max_group = max(obj['group'] for obj in data['objects'])  # get the max group
                max_layer = max(obj['layer'] for obj in data['objects'])  # get the max layer
            else:
                max_group, max_layer = 0, 0.0  # if there is no object at all
            segmentation = valid_prediction['masks'][idx]  # get the ISAT segmentation
            bbox = valid_prediction['bboxes'][idx].tolist()  # get the ISAT bbox ndarray
            group_assigned = False   # whether the group is assigned
            image_shape = cv2.imread(valid_prediction['image_path']).shape
            for obj in data['objects']:
                if UtilsISAT.bbox_intersection(bbox, obj['bbox'], threshold=0.5):
                    group_assigned = True  # assign the same group as the intersect object
                    if obj['category'] == valid_prediction['category_name'][idx]:  # same object: just replace the segmentation
                        obj_ellipse = UtilsISAT.ellipse_filter(UtilsISAT.segmentation2mask(obj['segmentation'], image_shape), 0.5)[1]  # get the ellipse value of original mask
                        mask_ellipse = UtilsISAT.ellipse_filter(UtilsISAT.segmentation2mask(segmentation, image_shape), 0.5)[1]  # get the value for new mask
                        if obj_ellipse > mask_ellipse:
                            continue
                        obj['segmentation'] = segmentation  # update the segmentation of the existing object
                        obj['area'] = int(self.isat_area(segmentation))  # update the area based on boolean mask
                        obj['layer'] = max_layer + 1.0  # increase layer by 1.0
                        obj['bbox'] = bbox  # update the bbox
                        obj['note'] = 'Auto'  # add Cellpose tag
                    else:
                        new_object_same_group = {
                            'category': valid_prediction['category_name'][idx],  # category as user input
                            'group': obj['group'],  # assign to the same group
                            'area': int(self.isat_area(segmentation)),  # get the area based on boolean mask
                            'segmentation': segmentation,  # the ISAT segmentation
                            'layer': max_layer + 1.0,  # increase layer by 1.0
                            'bbox': bbox,  # the ISAT bbox
                            'iscrowd': False,
                            'note': 'Auto'}
                        data['objects'].append(new_object_same_group)  # update the objects section
            if not group_assigned:
                new_object = {
                    'category': valid_prediction['category_name'][idx],  # category as user input
                    'group': max_group + 1,  # assign to new group
                    'area': int(self.isat_area(segmentation)),  # get the area based on boolean mask
                    'segmentation': segmentation,  # the ISAT segmentation
                    'layer': max_layer + 1.0,  # increase layer by 1.0
                    'bbox': bbox,  # the ISAT bbox
                    'iscrowd': False,
                    'note': 'Auto'}
                data['objects'].append(new_object)  # update the objects section
            return data

        print('Converting predictions to ISAT JSON files...')
        for valid_prediction in tqdm(valid_predictions, total=len(valid_predictions)):
            image_path = valid_prediction['image_path']  # get the image path
            json_path = f'{os.path.splitext(image_path)[0]}.json'  # get the ISAT json file path
            if not os.path.exists(json_path):
                Anything2ISAT.create_empty_json(image_path)  # create an empty ISAT json file if not exists
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            if data['info']['note'] != '' or valid_prediction['masks'] is None:
                continue  # only convert unchecked ISAT json files
            for idx, _ in enumerate(valid_prediction['masks']):
                data = populate_objects(data, valid_prediction, idx)  # populate the objects setction
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file)  # save updated json file
        return None

    @staticmethod
    def from_isat(json_dir_1: str, json_dir_2: str, json_dir_merged: str) -> None:
        """
        Combines two ISAT json files from different models and removes redundant objects
        This method merges JSON files from two directories into a third directory, and ensures objects are not duplicated based on overlapping bboxes

        Args:
        - json_dir_1 (str): the directory containing the first set of ISAT JSON files
        - json_dir_2 (str): the directory containing the second set of ISAT JSON files
        - json_dir_merged (str): the directory where the merged JSON files will be saved

        Returns:
        - None: writes the merged ISAT JSON files into the specified output directory
        """
        def populate_objects(data: dict, obj_json_2: dict) -> dict:
            """
            Populate the existing ISAT json file. Assign the object to an existing group if bboxes intersect
            Input: data: dict is the data from json 1. obj_json_2 is an object in json 2
            The obj_json_2 is comparied with all objects of json 1 to see if there is a bbox overlap
            """
            if len(data['objects']) > 0:
                max_group = max(obj['group'] for obj in data['objects'])  # get the max group
                max_layer = max(obj['layer'] for obj in data['objects'])  # get the max layer
            else:
                max_group, max_layer = 0, 0.0  # if there is no object at all
            group_assigned = False   # whether the group is assigned
            for obj in data['objects']:
                if UtilsISAT.bbox_intersection(obj_json_2['bbox'], obj['bbox'], threshold=0.5):
                    group_assigned = True  # assign the same group as the intersect object
                    if obj['category'] == obj_json_2['category']:  # same object: just replace the segmentation
                        # compare which segmentation is more eliptical:
                        obj['segmentation'] = obj_json_2['segmentation']  # update the segmentation of the existing object
                        obj['area'] = obj_json_2['area']  # update the area based on boolean mask
                        obj['layer'] = max_layer + 1.0  # increase layer by 1.0
                        obj['bbox'] = obj_json_2['bbox']  # update the bbox
                        obj['note'] = obj_json_2['note']  # update the note
                    else:
                        new_object_same_group = {
                            'category': obj_json_2['category'],  # the same group but other categories
                            'group': obj['group'],  # assign to the same group
                            'area': obj_json_2['area'],  # get the area based on boolean mask
                            'segmentation': obj_json_2['segmentation'],  # the ISAT segmentation
                            'layer': max_layer + 1.0,  # increase layer by 1.0
                            'bbox': obj_json_2['bbox'],  # the ISAT bbox
                            'iscrowd': obj_json_2['iscrowd'],
                            'note': obj_json_2['note']}
                        data['objects'].append(new_object_same_group)  # update the objects section
            if not group_assigned:
                new_object = {
                    'category': obj_json_2['category'],  # new group and other categories
                    'group': max_group + 1,  # assign to new group
                    'area': obj_json_2['area'],  # get the area based on boolean mask
                    'segmentation': obj_json_2['segmentation'],  # the ISAT segmentation
                    'layer': max_layer + 1.0,  # increase layer by 1.0
                    'bbox': obj_json_2['bbox'],  # the ISAT bbox
                    'iscrowd': obj_json_2['iscrowd'],
                    'note': obj_json_2['note']}
                data['objects'].append(new_object)  # update the objects section
            return data

        json_paths = get_paths(json_dir_1, 'json')  # get the json paths in the directory 1
        for json_path in tqdm(json_paths, total=len(json_paths)):
            json_base_name = os.path.basename(json_path)  # get the file base name
            json_path_2 = os.path.join(json_dir_2, json_base_name)  # get the json path in the directory 2 (the same base name)
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            with open(json_path_2, 'r', encoding='utf-8') as file:
                data_2 = json.load(file)  # load the json data 2
                for obj_json_2 in data_2['objects']:
                    data = populate_objects(data, obj_json_2)  # populate the objects setction
            json_path_merged = os.path.join(json_dir_merged, json_base_name)  # merged json path
            with open(json_path_merged, 'w', encoding='utf-8') as file:
                json.dump(data, file)  # save the merged json file
        return None

    @staticmethod
    def from_legacy_starch_annotation(images_dir: str):
        """
        Combines two ISAT json files from different models and removes redundant objects
        This method merges JSON files from two directories into a third directory, and ensures objects are not duplicated based on overlapping bboxes

        Args:
        - json_dir_1 (str): the directory containing the first set of ISAT JSON files
        - json_dir_2 (str): the directory containing the second set of ISAT JSON files
        - json_dir_merged (str): the directory where the merged JSON files will be saved

        Returns:
        - None: writes the merged ISAT JSON files into the specified output directory
        """
        guard_cell_mask_dir = images_dir.replace('images', 'Masks_GC')  # the guard cell mask directory
        starch_mask_dir = images_dir.replace('images', 'Masks_Starch')  # the starch mask directory
        return None


class ISAT2Anything:
    """Convert ISAT forma to other formats"""
    def __init__(self,
                 images_dir: str = None,
                 annotations_dir: str = None,
                 output_dir: str = None):
        self.images_dir = images_dir  # images directory
        self.annotations_dir = annotations_dir  # annotations directory
        self.output_dir = output_dir  # new annotation output directory

    @staticmethod
    def to_coco(annotations_dir, output_dir) -> None:
        """
        Converts ISAT format to MSCOCO format
        Warning: MAKE COPIES BEFORE DO IT! IT WILL REMOVE ALL ISAT JSON FILES!

        Args:
        - annotations_dir (str): the directory containing the ISAT JSON files
        - output_dir (str): the file path where the converted MSCOCO JSON file will be saved

        Returns:
        - None: outputs a MSCOCO formatted JSON file and deletes the original ISAT JSON files
        """
        coco_annotation = {}  # to collect COCO annotations
        coco_annotation['info'] = {}  # image information section
        coco_annotation['info']['description'] = 'StomataPy'
        coco_annotation['info']['version'] = None  # no sepcific version
        coco_annotation['info']['year'] = 2024  # year of publication
        coco_annotation['info']['contributor'] = 'Hongyuan Zhang'  # author name
        coco_annotation['info']['date_created'] = None  # not specific
        coco_annotation['licenses'] = []  # licenses information
        datat_license = {}  # license dictionary
        datat_license['url'] = None  # not specific
        datat_license['id'] = 0  # not specific
        datat_license['name'] = 'Apache 2.0'  # not specific
        coco_annotation['licenses'].append(datat_license)  # fill in liscenses information
        coco_annotation['images'] = []  # to collect image information
        coco_annotation['annotations'] = []  # to collect annotation information
        coco_annotation['categories'] = []  # to collect categories information
        categories_dict = {}
        json_paths = get_paths(annotations_dir, '.json')  # get all json file paths
        for idx, json_path in enumerate(json_paths):
            with open(json_path, encoding='utf-8') as file:
                dataset = json.load(file)  # load the individual json file
                info = dataset.get('info', {})  # get information setction
                description = info.get('description', '')  # read the description field
                if not description.startswith('ISAT'):
                    continue  # if the json file was not made wih ISAT, skip
                image_name = info.get('name', '')  # get the image name
                width = info.get('width', None)  # get the image width
                height = info.get('height', None)  # get the image height
                objects = dataset.get('objects', [])  # get the objects information
                coco_image_info = {}  # to store image information in a dictionary
                coco_image_info['license'] = 'Apache 2.0'  # not specific
                coco_image_info['url'] = None  # not specific
                coco_image_info['file_name'] = image_name  # fill in the image name
                coco_image_info['height'] = height  # fill in the image height
                coco_image_info['width'] = width  # fill in the image width
                coco_image_info['date_captured'] = None  # not specific
                coco_image_info['id'] = idx  # fill in the image idex
                coco_annotation['images'].append(coco_image_info)  # collect information for all images
                objects_groups = [obj.get('group', 0) for obj in objects]  # get object groups
                objects_groups.sort()  # sort object groups
                objects_groups = set(objects_groups)  # avoid redundancy
                for group in objects_groups:
                    objs_with_group = [obj for obj in objects if obj.get('group', 0) == group]  # filter out none group
                    catergories = set([obj.get('category', 'unknow') for obj in objs_with_group])  # avoid redundancy
                    for catergory in catergories:
                        if catergory not in categories_dict:
                            categories_dict[catergory] = len(categories_dict)  # add new category if not added
                        category_index = categories_dict.get(catergory)  # add new category index if not added
                        objs_with_cat = [obj for obj in objs_with_group if obj.get('category', 0) == catergory]
                        crowds = set([obj.get('iscrowd', 'unknown') for obj in objs_with_group])  # iscrowd
                        for crowd in crowds:
                            objs_with_crowd = [obj for obj in objs_with_cat if obj.get('iscrowd', 0) == crowd]  # all iscrowd=False objects
                            coco_annotation_info = {}  # to store annotation information of these objects
                            coco_annotation_info['iscrowd'] = crowd  # iscrowd
                            coco_annotation_info['image_id'] = idx  # image index
                            coco_annotation_info['image_name'] = image_name  # image name
                            coco_annotation_info['category_id'] = category_index  # catergory index
                            coco_annotation_info['id'] = len(coco_annotation['annotations'])  # object index
                            coco_annotation_info['segmentation'] = []  # to segmentation coordinates
                            coco_annotation_info['area'] = 0.  # initialize area with 0
                            coco_annotation_info['bbox'] = []  # to store MSCOCO bbox
                            for obj in objs_with_crowd:
                                segmentation = obj.get('segmentation', [])  # get object segmentation list
                                area = obj.get('area', 0)  # get object segmentation area
                                bbox = obj.get('bbox', [])  # get the object ISAT bbox
                                if bbox is None:
                                    segmentation_np = np.array(segmentation)  # convert to np.array for calculation
                                    bbox = [min(segmentation_np[:, 0]), min(segmentation_np[:, 1]), max(segmentation_np[:, 0]), max(segmentation_np[:, 1])]  # ISAT bbox to MSCOCO bbox
                                    del segmentation_np  # delete the np.array to save memory
                                segmentation = [e for p in segmentation for e in p]
                                if bbox != []:
                                    if coco_annotation_info['bbox'] == []:
                                        coco_annotation_info['bbox'] = bbox  # fill in bbox information
                                    else:
                                        bbox_tmp = coco_annotation_info['bbox']  # if bbox already exists
                                        bbox_tmp = [min(bbox_tmp[0], bbox[0]), min(bbox_tmp[1], bbox[1]), max(bbox_tmp[2], bbox[2]), max(bbox_tmp[3], bbox[3])]  # get (xmin, ymin, xmax, ymax)
                                        coco_annotation_info['bbox'] = bbox_tmp  # fill in bbox information
                                coco_annotation_info['segmentation'].append(segmentation)  # collect all segmentation
                                if area is not None:
                                    coco_annotation_info['area'] += float(area)  # add the area from 0
                            bbox_tmp = coco_annotation_info['bbox']  # get the ISAT format bbox
                            coco_annotation_info['bbox'] = [bbox_tmp[0], bbox_tmp[1], bbox_tmp[2] - bbox_tmp[0], bbox_tmp[3] - bbox_tmp[1]]  # (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height)
                            coco_annotation['annotations'].append(coco_annotation_info)  # collect all annotations
            os.remove(json_path)  # remove all json files
        categories_dict = sorted(categories_dict.items(), key=lambda x: x[0])  # sort categories by keys
        new_category_ids = {category: index + 1 for index, (category, old_id) in enumerate(categories_dict)}  # create a new dictionary mapping class names to new ids, staring from 1
        id_to_category_name = {old_id: category_name for category_name, old_id in categories_dict}  # create a reverse mapping from old category IDs to category names
        for annotation in coco_annotation['annotations']:
            old_category_id = annotation['category_id']  # to reoder the catergory id
            category_name = id_to_category_name.get(old_category_id)
            if category_name:
                new_category_id = new_category_ids.get(category_name)
                if new_category_id is not None:
                    annotation['category_id'] = new_category_id
        coco_annotation['categories'] = [{'name': name, 'id': id, 'supercategory': None} for name, id in new_category_ids.items()]  # reoder the catergory id
        with open(output_dir, 'w', encoding='utf-8') as file:
            json.dump(coco_annotation, file)  # save MSCOCO json file
        return None

    def to_cellpose(self, category: str = 'stoma') -> None:
        """
        Converts ISAT format JSON files to Cellpose .npy files for a specific category
        This method processes each JSON file, extracts segmentations for the specified category, and converts them into a format usable by Cellpose

        Args:
        - category (str): the category of objects to convert, defaults to 'stoma'

        Returns:
        - None: saves .npy files compatible with Cellpose in the specified annotations directory
        """
        def masks_to_outlines(masks: np.ndarray) -> np.ndarray:
            """Converts a mask where objects are labeled with integers into a binary outline image"""
            outlines = np.zeros(masks.shape, np.uint8)  # initializes an array of zerostructural element
            for idx in range(1, masks.max() + 1):
                mask_n = masks == idx  # get the nth masks (starting form 1)
                outline = np.logical_xor(mask_n, binary_erosion(mask_n, structure=generate_binary_structure(2, 1)))  # a logical XOR is performed between the original mask and the eroded mask, resulting in the outline of the object
                outlines[outline] = 255  # fill in the outline to be white
            return outlines

        json_paths = get_paths(self.annotations_dir, '.json')  # get the paths of all json files
        for json_path in tqdm(json_paths, total=len(json_paths)):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # load the json data
            # if data['info']['note'] == '':
            #    continue  # only convert checked ISAT json files
            image_width, image_height = data['info']['width'], data['info']['height']  # image width and height
            segmentations = [obj['segmentation'] for obj in data['objects'] if obj['category'].lower() == category]
            converted_mask = np.zeros((image_height, image_width), dtype=np.int32)  # update dimensions if needed
            for idx, segmentation in enumerate(segmentations, 1):
                rows, columns = polygon([point[1] for point in segmentation], [point[0] for point in segmentation], converted_mask.shape)  # get the y and x coordinates for each object
                converted_mask[rows, columns] = idx  # draw each object's segmentation onto the mask array
            cellpose_npy_data = {'outlines': masks_to_outlines(converted_mask), 'masks': converted_mask, 'filename': data['info']['name']}  # combine cellpose data
            cellpose_npy_path = os.path.join(self.annotations_dir, f"{os.path.splitext(data['info']['name'])[0]}_seg.npy")  # get cellpose seg npy file path
            np.save(cellpose_npy_path, cellpose_npy_data)  # save the cellpose npy file
        return None
