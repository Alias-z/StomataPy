"""Module providing functions inference starch images"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import time  # record time
import copy  # for deepcopy
import shutil  # copy and paste files
import cv2  # OpenCV
import numpy as np  # NumPy
import torch  # PyTorch
from tqdm import tqdm  # progress bar
import tifffile   # read metadata from .tif files
import pandas as pd  # for Excel sheet
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from mmseg.utils import register_all_modules as mmseg_utils_register_all_modules  # register mmseg modules
from mmseg.apis import init_model as mmseg_apis_init_model  # initialize mmseg model
from mmseg.apis import inference_model as mmseg_apis_inference_model  # mmseg inference segmentor
from ..core.core import device, image_types, GC_Colors, Starch_Colors, imread_rgb, unique_color, color_select, get_contour, lab_logo  # import core functions
from ..utils.stoma_dimension import GetDiameter  # import core functions for stomatal aperature


class StarchSeeker:
    """Inference starch images"""
    def __init__(self,
                 input_dir: str,
                 output_name: str = 'Results',
                 batch_size: int = 100,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 detector_threshold: float = 0.9,
                 segmentor_config_path: str = None,
                 segmentor_weight_path: str = None,
                 concatenate_excels: bool = True,
                 empty_dataframe: pd.DataFrame = pd.DataFrame(columns=['image name', 'image height', 'image width', 'scale (pixels/\u03BCm)', 'stomata lenghth (\u03BCm)', 'stomata width (\u03BCm)',
                                                                       'guard cell area (\u03BCm\N{SUPERSCRIPT TWO})', 'starch area (\u03BCm\N{SUPERSCRIPT TWO})', 'starch guard cell area ratio (%)'])
                 ):
        self.input_dir = os.path.normpath(input_dir)  # input directory
        self.output_name = output_name  # output folder name
        self.batch_size = batch_size  # inference batch size
        self.detector_config_path = detector_config_path  # instance segmentation config path
        self.detector_weight_path = detector_weight_path  # instance segmentation weight path
        self.detector_threshold = detector_threshold  # instance segmentation threshold
        self.segmentor_config_path = segmentor_config_path  # semantic segmentation config path
        self.segmentor_weight_path = segmentor_weight_path  # semantic segmentation weight path
        self.concatenate_excels = concatenate_excels  # whether to concatenate all Excel sheets
        self.empty_dataframe = empty_dataframe  # template dataframe

        lab_logo()  # print logo

    def get_images(self, subfolder_path):
        """Load images in a subfolder_path into a list, and get the properties of each image"""
        images, image_sizes, image_scales = [], [], []  # to store values
        file_names = sorted(os.listdir(subfolder_path), key=str.casefold)  # sort file names
        file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        print('\n \x1b[31m 1. loading all input images')
        for name in tqdm(file_names, total=len(file_names)):
            scale = None  # initialize default scale value (pixels per micrometer)
            if name.lower().endswith(('.tif', '.tiff')):
                try:
                    with tifffile.TiffFile(os.path.join(subfolder_path, name)) as tif:
                        metadata = tif.pages[0].tags  # try to get the metadata in the .tif file
                        if 'XResolution' in metadata:
                            scale = metadata['XResolution'].value[0] / metadata['XResolution'].value[1]
                            resolution_unit = [tag.value for tag in metadata.values() if tag.name == 'ResolutionUnit'][0]
                            if resolution_unit.name == 'CENTIMETER':
                                scale *= 1e-4  # convert centimeter to micrometer
                except ValueError:
                    pass
            image = imread_rgb(os.path.join(subfolder_path, name))  # load the image in RGB scale
            images.append(image); image_sizes.append(image.shape[:2]); image_scales.append(scale)    # noqa: collect image arrays, scales, and dimensions
        return file_names, images, image_sizes, image_scales

    def batch_images(self, subfolder_path):
        """Split images into batches to avoid CUDA out of memory."""
        output_dir = os.path.join(self.output_name, *os.path.normpath(subfolder_path).split(os.sep)[1:])  # create output dir
        os.makedirs(output_dir, exist_ok=True)  # create the output folder
        file_names, images, image_sizes, image_scales = self.get_images(subfolder_path)  # get the file names, images, sizes, and scales
        batches = []  # to store information of each batch
        for i in range(0, len(file_names), self.batch_size):
            file_names_batches = file_names[i: i + self.batch_size]  # split the names into batches
            images_batches = images[i: i + self.batch_size]  # split the images accordingly
            image_sizes_batches = image_sizes[i: i + self.batch_size]  # same for the image sizes
            image_scales_batches = image_scales[i: i + self.batch_size]  # image scales
            batches.append([subfolder_path, output_dir, file_names_batches, images_batches, image_sizes_batches, image_scales_batches])  # summary
        return batches

    def gc_seeker(self, images):
        """Instance segmentation for two guard cells"""
        originals = copy.deepcopy(images)  # prevent change in position
        good_ids, detector_results, area_gcs = [], [], []  # to collect the id of images that can be detected, whose guard cells regions are filled with colors
        mmdet_utils_register_all_modules(init_default_scope=False)  # initialize mmdet scope
        detector = mmdet_apis_init_detector(self.detector_config_path, self.detector_weight_path, device=device)  # initialize a detector from config file
        predictions = mmdet_apis_inference_detector(detector, images)  # inference image(s) with the detector.
        for idx, prediction in tqdm(enumerate(predictions), total=len(predictions)):
            image = originals[idx]  # get target image by indexing
            top_indices = torch.topk(prediction.pred_instances.scores, k=2).indices  # choose the top 2 guard cells instances
            scores = [prediction.pred_instances.scores[i] for i in top_indices]  # get their prediction scores (condifidence levels)
            if all(score > self.detector_threshold for score in scores):
                gc_1, gc_2 = [prediction.pred_instances.masks[i].cpu().numpy() for i in top_indices]  # load bool mask for both guard cells
                a_gc_1, a_gc_2 = np.count_nonzero(gc_1), np.count_nonzero(gc_2)  # calculate the number of non-background pixels
                a_gc = a_gc_1 + a_gc_2  # the total area of the 2 guard cells
                a_image = image.shape[0] * image.shape[1]  # the total area of the image
                if np.logical_and(gc_1, gc_2).any() and a_gc_1 / a_gc > 0.4 and a_gc_2 / a_gc > 0.4 and a_gc / a_image >= 0.05:
                    # threshold must be meet that the 2 guard cell should roughly occupy half of the total stomata, and they have intersections
                    gc_1, gc_2 = gc_1.astype(np.uint8), gc_2.astype(np.uint8)  # convert the bool mask to unit8
                    image[gc_1 == 1] = np.array(GC_Colors[1].mask_rgb)  # fill in color to the image on guard cell 1 region
                    image[gc_2 == 1] = np.array(GC_Colors[2].mask_rgb)  # same as in the guard cell 2 region
                    color = unique_color(image).tolist()  # get unique color values of the current images
                    if GC_Colors[1].mask_rgb in color and GC_Colors[2].mask_rgb in color:
                        good_ids.append(idx); detector_results.append(image); area_gcs.append([a_gc_1, a_gc_2])  # noqa: if the colors of both guard cells exist, we consider this as a good image (detected)
        return good_ids, detector_results, area_gcs

    def gc_split(self, images, masks, names):
        """Split starch granules based on the guard cells regions"""
        new_names, new_images = [], []  # to collect new image names and the splited guard cell images
        for idx, name in tqdm(enumerate(names), total=len(names)):
            image, mask_gc = images[idx], masks[idx]    # load a image and its mask
            if GC_Colors[1].mask_rgb in unique_color(mask_gc):
                new_name = os.path.splitext(name)[0] + f' GC_1{os.path.splitext(name)[1]}'  # file for guard cell 1
                image_gc1 = color_select(image, mask_gc, GC_Colors[1].mask_rgb)  # fill in guard cell 1 region with guard cell 1 color
                new_names.append(new_name); new_images.append(image_gc1)  # noqa: collect names and images for guard cell 1
            if GC_Colors[2].mask_rgb in unique_color(mask_gc):
                new_name = os.path.splitext(name)[0] + f' GC_2{os.path.splitext(name)[1]}'  # same as above for guard cell 2
                image_gc2 = color_select(image, mask_gc, GC_Colors[2].mask_rgb)  # same as guard cell 1 above
                new_names.append(new_name); new_images.append(image_gc2) # noqa: same as guard cell 1 above
        return new_names, new_images

    def starch_seeker(self, images):
        """Semantic segmentation for starch granules"""
        segmentor_results = []  # to collect segmentation results (filled with prediction colors)
        # mmseg_utils_register_all_modules(init_default_scope=False)  # initialize mmseg scope
        segmentor = mmseg_apis_init_model(self.segmentor_config_path, self.segmentor_weight_path, device=device)   # initialize a segmentor from config file
        for image in tqdm(images, total=len(images)):
            prediction = mmseg_apis_inference_model(segmentor, image)  # inference image with the segmentor (batch inference is not supported)
            prediction = prediction.pred_sem_seg.data.cpu().numpy()[0]  # move prediction from GPU to CPU
            empty = np.zeros([prediction.shape[0], prediction.shape[1], 3], dtype=np.uint8)  # create a black image
            for idx in np.unique(prediction):
                color = Starch_Colors[idx].mask_rgb  # pick the color for starch
                mask = prediction == idx  # pick the prediction class
                for idx_j in range(3):
                    empty[:, :, idx_j] = np.where(mask, color[idx_j], empty[:, :, idx_j])  # fill in colors for the predicted class
            segmentor_results.append(empty)  # collect the images that has been filled with prediction colors
        return segmentor_results

    def is_float(self, string):
        """Check if a string can be converted to float"""
        try:
            float(string)
            return True
        except ValueError:
            return False

    def results2d(self, image, stomata_trait, image_name, scale, area_gcs):
        """Output all 2D results"""
        a_gc_1, a_gc_2 = area_gcs  # guard cell 1 area and guard cell 2 area in pixels
        a_starch_1 = len(np.where(np.all(image == GC_Colors[1].mask_rgb, axis=-1))[0])  # guard cell 1 starch area
        a_starch_2 = len(np.where(np.all(image == GC_Colors[2].mask_rgb, axis=-1))[0])  # guard cell 2 starch area
        if a_starch_1 > a_starch_2:
            a_starch_gc1 = a_starch_1; a_starch_gc2 = a_starch_2  # noqa: rank the area of guard cells
        else:
            a_starch_gc2 = a_starch_1; a_starch_gc1 = a_starch_2  # noqa: guard cell 1 area is always larger than guard cell 2
            a_gc_1, a_gc_2 = a_gc_2, a_gc_1  # swap the guard cell 1 area and guard cell 2 area
        a_starch_gc1_absolute = a_starch_gc1 * (1 / scale) ** 2  # the absolute area values of guard cell 1 starch
        a_starch_gc2_absolute = a_starch_gc2 * (1 / scale) ** 2  # same for guard cell 2 starch
        a_gc_1_absolute = a_gc_1 * (1 / scale) ** 2  # the absolute area values of the guard cell 1
        a_gc_2_absolute = a_gc_2 * (1 / scale) ** 2  # the abosulte area values of the guard cell 2

        stomata_lenghth_pixel, stomata_width_pixel = stomata_trait
        stomata_lenghth = stomata_lenghth_pixel * (1 / scale)  # the absolute lenghth of stomata
        stomata_width = stomata_width_pixel * (1 / scale)  # the absolute width of stomata

        results = {'image name': [image_name + ' GC 1', image_name + ' GC 2'],
                   'image height': [image.shape[0], image.shape[0]],
                   'image width': [image.shape[1], image.shape[1]],
                   'scale (pixels/\u03BCm)': [scale, scale],
                   'stomata lenghth (\u03BCm)': [stomata_lenghth, stomata_lenghth],
                   'stomata width (\u03BCm)': [stomata_width, stomata_width],
                   'guard cell area (\u03BCm\N{SUPERSCRIPT TWO})': [a_gc_1_absolute, a_gc_2_absolute],
                   'starch area (\u03BCm\N{SUPERSCRIPT TWO})': [a_starch_gc1_absolute, a_starch_gc2_absolute],
                   'starch guard cell area ratio (%)': [a_starch_gc1 / a_gc_1 * 100, a_starch_gc2 / a_gc_2 * 100]}
        results = pd.DataFrame(data=results)  # collect results in a pd dataframe for exporting to an Excel sheet
        return results

    def gcstarch(self, batch):
        """The overall function that automaticly detect starch for a image batch"""
        subfolder_path, output_dir, file_names_batch, images_batch, image_sizes_batch, image_scales_batch = batch  # load parameters for the batch

        print('\n \x1b[31m 2. detecting guard cells \n')
        good_ids, detector_results, area_gcs = self.gc_seeker(images_batch)  # get the predictions
        good_images = [images_batch[i] for i in good_ids]  # get the good images (guard cells detected)
        good_images_names = [file_names_batch[i] for i in good_ids]  # get good images names
        good_images_scales = [image_scales_batch[i] for i in good_ids]  # get good images scales
        bad_ids = [i for i in range(len(file_names_batch)) if i not in good_ids]  # get the ids of bad images (not detected)
        bad_images_names = [file_names_batch[i] for i in bad_ids]  # get the bad images names
        bad_images_sizes = [image_sizes_batch[i] for i in bad_ids]  # get the bad images sizes
        bad_images_scales = [image_scales_batch[i] for i in bad_ids]  # get the bad images scales
        failure = [file_names_batch[i] for i in bad_ids]; failure_rate = len(failure) / len(file_names_batch) * 100  # noqa: the bad images
        bad_images_dir = os.path.join(output_dir, 'Bad Images'); os.makedirs(bad_images_dir, exist_ok=True)  # noqa: create a folder for bad images
        for name in failure:
            shutil.copy(os.path.join(subfolder_path, name), os.path.join(bad_images_dir, name))  # export the bad ones to a folder
        if len(failure) > 0:
            print(f'\n \x1b[31m fail to detect both guard cells in the following images {failure},\n  which can be used for training \n')
        print(f'\n failure rate = {failure_rate}%\n')

        print('\n \x1b[31m 3. detecting starch \n')
        segmentor_results = self.starch_seeker(good_images)  # segment starch for all good images

        print('\n \x1b[31m 4. splitting guard cells \n')
        new_images = self.gc_split(segmentor_results, detector_results, good_images_names)[1]  # get the file names named with guard cell numbers

        print('\n \x1b[31m 5. generating starch masks and stomata traits images \n')
        output_starch_mask = os.path.join(output_dir, 'Starch Masks')  # where to export the starch masks
        os.makedirs(output_starch_mask, exist_ok=True)  # create a folder for starch masks
        output_stomata = os.path.join(output_dir, 'Stomata Traits')  # where to export the stomata traits
        os.makedirs(output_stomata, exist_ok=True)  # create a folder for stomata traits
        starch, stomata_traits = [], []  # to collect predicted starch results
        for idx, name in tqdm(enumerate(good_images_names), total=len(good_images_names)):
            mask_gc = detector_results[idx]  # the original images with regions of guard cells painted
            black_image = np.zeros((mask_gc.shape[0], mask_gc.shape[1], 3), dtype=np.uint8)  # empty black image
            mask_gc1 = color_select(mask_gc, mask_gc, GC_Colors[1].mask_rgb)  # the region of guard cell 1 mask
            mask_gc2 = color_select(mask_gc, mask_gc, GC_Colors[2].mask_rgb)  # the region of guard cell 2 mask
            black_image = get_contour(mask_gc1, black_image)[0]  # get the contour of guard cell 1
            black_image = get_contour(mask_gc2, black_image)[0]  # get the contour of guard cell 2
            mask_starch_1 = new_images[idx * 2]  # the starch image with guard cell 1 only
            mask_starch_2 = new_images[idx * 2 + 1]  # the starch image with guard cell 2 only
            black_image[np.all(mask_starch_1 == Starch_Colors[1].mask_rgb, axis=-1)] = GC_Colors[1].mask_rgb  # fill in the black image with colors for guard cell 1
            black_image[np.all(mask_starch_2 == Starch_Colors[1].mask_rgb, axis=-1)] = GC_Colors[2].mask_rgb  # same for guard cell 2
            starch.append(black_image)  # collect all starch images that each guard cell region filled with different colors
            cv2.imwrite(os.path.join(output_starch_mask, name), cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR))  # export the image
            dimension = GetDiameter(black_image, shrink_ratio=1.2, line_thickness=3).pca()  # calculate the stomata lenghth and width
            stomata_lenghth_pixel, stomata_width_pixel, rgb_mask = dimension['length'], dimension['width'], dimension['visualization']  # get the results
            stomata_traits.append([stomata_lenghth_pixel, stomata_width_pixel])  # collect stomata percentage result
            cv2.imwrite(os.path.join(output_stomata, name), cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))  # export the image

        print('\n \x1b[31m 6. calculating starch percentage \n')
        with open(os.path.join(subfolder_path, 'scale.txt'), 'r', encoding='utf-8') as file:
            scale = file.readline().strip()  # stripping potential whitespace
            scale = float(scale) if self.is_float(scale) else None  # checking if the scale is a float or not
        results = copy.deepcopy(self.empty_dataframe)  # to store values as a dataframe
        for idx, name in tqdm(enumerate(file_names_batch), total=len(file_names_batch)):
            if name in good_images_names:
                index = good_images_names.index(name)  # get the index of the good image
                scale_original = good_images_scales[index]  # get the scale from image meta info
                if scale_original is None:
                    scale_original = scale  # use the value from 'scale.txt' if meta info not available
                result = self.results2d(starch[index], stomata_traits[index], name, float(scale_original), area_gcs[index])
            elif name in bad_images_names:
                index = bad_images_names.index(name)  # get the index of the bad image
                scale_original = bad_images_scales[index]  # get the scale from image meta info info
                if scale_original is None:
                    scale_original = scale  # use the value from 'scale.txt' if meta info not available
                result = {'image name': [name + ' GC 1', name + ' GC 2'],
                          'image height': [bad_images_sizes[index][0], bad_images_sizes[index][0]],
                          'image width': [bad_images_sizes[index][1], bad_images_sizes[index][1]],
                          'scale (pixels/\u03BCm)': [float(scale_original), float(scale_original)],
                          'stomata lenghth (\u03BCm)': [pd.NA, pd.NA],
                          'stomata width (\u03BCm)': [pd.NA, pd.NA],
                          'guard cell area (\u03BCm\N{SUPERSCRIPT TWO})': [pd.NA, pd.NA],
                          'starch area (\u03BCm\N{SUPERSCRIPT TWO})': [pd.NA, pd.NA],
                          'starch guard cell area ratio (%)': [pd.NA, pd.NA]}
                result = pd.DataFrame(data=result)  # fill in NAs to bad image results
            results = pd.concat([results, result], axis=0)  # concatenate all results
        return len(file_names_batch), len(failure), results

    def batch_predict(self):
        """detect starch for all subfolders that contain 'scale.txt' file"""
        start, images_num, failure, folders = time.time(), 0, 0, []  # initialize default values
        folders = sorted([root for root, _, files in os.walk(self.input_dir) if 'scale.txt' in files], key=str.casefold)  # sort folders by name
        gc_color = [GC_Colors[idx].mask_rgb for idx in range(len(GC_Colors) - 2)][1:]  # map color for the overlay
        dataframes = []  # to collect the results from all sunfolders
        for folder in folders:
            print(f'\n \033[34m processing {folder} \n')
            results = copy.deepcopy(self.empty_dataframe)  # to store values as a dataframe
            split_folder = folder.split(os.sep)[1:]  # get the folder name
            if len(split_folder) > 1:
                output_dir = os.path.join(self.output_name, os.path.join(*split_folder))  # in case input dir has a relative parent dir
            else:
                output_dir = os.path.join(self.output_name, folder)  # in case input dir has no relative parent dir
            os.makedirs(output_dir, exist_ok=True)  # create the output folder
            batches = self.batch_images(folder)  # get subfolder image batches
            for idx, batch in enumerate(batches):
                print(f'\n \033[34m batch {idx + 1} out of {len(batches)} \n')
                n_files, n_failure, result = self.gcstarch(batch)  # run starch detection pipline for each batch
                failure += n_failure  # record number of imagse that the guard cell cannot be detected
                images_num += n_files  # record the total number of images processed
                results = pd.concat([results, result], axis=0)  # noqa: concatenate results from all batches
            results.to_excel(os.path.join(output_dir, 'starch area.xlsx'), index=False)  # export results to Excel
            results.insert(0, 'folder name', os.path.basename(folder)); dataframes.append(results)  # noqa: for summarizing results
            print('\n \x1b[31m 7. creating starch overlay on original images \n')
            result_folder = folder.replace(os.path.normpath(self.input_dir).split(os.sep)[0], self.output_name)  # root of predicted results
            mask_folder = os.path.join(result_folder, 'Starch Masks')  # where masks stored
            visualize_folder = os.path.join(result_folder, 'Starch Overlay'); os.makedirs(visualize_folder, exist_ok=True)  # noqa: to store mask overlay
            file_names = sorted(os.listdir(mask_folder), key=str.casefold)  # noqa: sort file names
            file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types)]  # images files only
            for name in tqdm(file_names, total=len(file_names)):
                image = imread_rgb(os.path.join(folder, name))  # load the original image
                mask = imread_rgb(os.path.join(mask_folder, name))  # load the predicted starch mask
                mask_colors, overlay_colors = np.array(gc_color), np.array(gc_color)  # load colors to be filled in
                mask_overlay = image.copy()  # to prevent change in position
                for mask_color, overlay_color in zip(mask_colors, overlay_colors):
                    mask_locs = np.where((mask == mask_color).all(axis=-1))  # find mask regions
                    mask_overlay[mask_locs] = image[mask_locs] * 0.5 + overlay_color * 0.5  # create starch overlay on the original image
                cv2.imwrite(os.path.join(visualize_folder, name), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))  # save the overlay
        if self.concatenate_excels and images_num != 0:
            print('\n \x1b[31m 8. concatenating all Excel sheets \n')
            dataframes = pd.concat(dataframes, axis=0)  # concatenate all the DataFrames
            dataframes.rename(columns={dataframes .columns[0]: 'folder name'}, inplace=True)  # rename the first column
            dataframes.to_excel(os.path.join(self.output_name, os.path.join(*self.input_dir.split(os.sep)[1:]), 'starch area summary.xlsx'), index=False)  # export the summarized results to Excel
        end = time.time()  # stop the timer
        print('\n \033[34m Done! \n')
        print(f'\033[34m processed {images_num} images in {(end-start)/60} min')
        print(f'\n \033[34mtotal detection failure rate = {failure/images_num*100} % \n')
        if images_num == 0:
            print('\n \n \x1b[31m there is no image provided any use in this run; please check the following possibilities:')
            print('\n 1. check if your input directory is correct')
            print('\n 2. maybe you forgot to put the "scale.txt" file in your folder?')
            print('\n 3. is the image quality good enough?')
            print(f'\n 4. are image formats not supported? currently the program supports \n {image_types}')
        return None

    ################################################################################################
    # functions related to 3D starch recontruction are defined in the following sections:
    ################################################################################################

    def read_stack_tif(self, stack_path):
        """read a z-stack .tif file, get 3D array in ZXY format and get ZXY scales"""
        with tifffile.TiffFile(stack_path) as tif:
            stack_zyxc = tif.asarray()  # in ZYXC format
            stack_zxyc = np.transpose(stack_zyxc, (0, 2, 1, 3))  # to ZXYC
            metadata = tif.pages[0].tags
            image_description = metadata['ImageDescription'].value  # read image description
            image_description = {line.split('=')[0]: line.split('=')[1] for line in image_description.split('\n')}  # extract information
            image_unit, image_nspacing = image_description['unit'], float(image_description['spacing'])  # get unit and z-axis spacing
            print(f'image unit: {image_unit}')
            image_scale_x = metadata['XResolution'].value[1] / metadata['XResolution'].value[0]  # pixel per unit
            image_scale_y = metadata['YResolution'].value[1] / metadata['YResolution'].value[0]  # pixel per unit
            scale_zxy = [image_nspacing, image_scale_x, image_scale_y]  # the z, x, y axis scale for correct 3D view
        return stack_zxyc, scale_zxy

    def find_closest_values(self, list1, list2):
        """for each element in list1, find the element in list2 whose value is the closest"""
        closest_values = [np.array(list2)[np.abs(np.array(list2) - value).argmin()] for value in list1]
        return closest_values

    def threshold_rgb(self, image, threshold=1):
        """threshold a RGB image"""
        if np.mean(image) < threshold:
            image[:] = 0
        return image

    def starch_volume(self, stack_zxy, scale_zxy):
        """calculate the starch volume"""
        voxel_volume = np.prod(scale_zxy)  # calculate the volume of a single voxel
        nonzero_count = np.count_nonzero(stack_zxy)  # count the number of non-zero elements in the array
        total_volume = nonzero_count * voxel_volume  # calculate the total volume of the non-zero regions
        return total_volume

    def gcstarch3d(self, stack_path, n=3):
        """predict starch masks for a confocal z-stack"""
        stack_zxyc, scale_zxy = self.read_stack_tif(stack_path)  # read tif stack and get the scale in z, x, y axis
        stack = [stack_zxyc[idx] for idx in range(0, stack_zxyc.shape[0])]  # flatten the stack into a images list
        good_ids, detector_results, _ = self.gc_seeker(stack)  # find slice whose guard cells can be detected
        slice_ids = list(range(0, stack_zxyc.shape[0]))  # generate the slice IDs from 0 to the end
        gc_mask_ids = self.find_closest_values(slice_ids, good_ids)  # assign the closest mask to those undetectable
        gc_masks, images, starch_masks = [], [], []  # create a empty list to store all guard cell masks
        for idx, image in enumerate(stack):
            gc_mask_id = gc_mask_ids[idx]  # find which mask should assign to the given slice
            index = good_ids.index(gc_mask_id)  # the idex in good masks IDs, which is the same idex for good masks
            gc_mask = detector_results[index]  # find the predicted mask according the idex
            gc1_region = (gc_mask == GC_Colors[1].mask_rgb).all(axis=-1)  # where guard cell 1 occupies
            gc2_region = (gc_mask == GC_Colors[2].mask_rgb).all(axis=-1)  # where guard cell 2 occupies
            (y1, x1), (y2, x2) = np.where(gc1_region), np.where(gc2_region)  # find the x, y coordinates of guard cell 1 and 2
            x1, y1, x2, y2 = x1.mean(), y1.mean(), x2.mean(), y2.mean()  # find the center x, y of guard cell 1 and 2
            if (x1**2 + y1**2) < (x2**2 + y2**2):  # unify the starch color in each guard cell
                gc_mask[gc1_region] = GC_Colors[1].mask_rgb  # guard cell is defined as the one closer to (0, 0)
                gc_mask[gc2_region] = GC_Colors[2].mask_rgb  # in this case, range1 is closer
            else:
                gc_mask[gc1_region] = GC_Colors[2].mask_rgb  # in this case, range1 is not close to (0, 0)
                gc_mask[gc2_region] = GC_Colors[1].mask_rgb
            gc_masks.append(gc_mask)  # collect the assigned mask
            image[np.logical_not(np.logical_or(gc1_region, gc2_region))] = 0  # assign black pixel to the image regions outside guard cells
            images.append(image)  # collect the images
        starch_masks = self.starch_seeker(images)  # predict starch for each slice
        gc1_starch_masks, gc2_starch_masks = [], []  # create a empty list for binary starch labels
        denoised_images, threshold = [], sum([np.mean(images[good_ids[0] - n]), np.mean(images[good_ids[-1] + n])]) / 2  # from the edge n slices, we cut the background
        print(f'threshold = {threshold}')
        for idx, image in enumerate(images):
            if idx in good_ids:
                denoised_images.append(image)  # those do not need to denoise
            else:
                denoised_images.append(self.threshold_rgb(image, threshold))  # apply thresholding to bad slices
        for idx, starch_mask in enumerate(starch_masks):
            if np.all(denoised_images[idx] == 0):
                starch_mask[:] = 0
            starch_mask[np.all(np.logical_and(gc_masks[idx] == GC_Colors[1].mask_rgb, starch_mask == Starch_Colors[1].mask_rgb), axis=-1)] = GC_Colors[1].mask_rgb  # fill starch in GC1 to GC color 1
            starch_mask[np.all(np.logical_and(gc_masks[idx] == GC_Colors[2].mask_rgb, starch_mask == Starch_Colors[1].mask_rgb), axis=-1)] = GC_Colors[2].mask_rgb  # fill starch in GC2 to GC color 2
            gc1_starch_mask = color_select(starch_mask, starch_mask, color=GC_Colors[1].mask_rgb)  # separate starch in guard cell 1
            gc2_starch_mask = color_select(starch_mask, starch_mask, color=GC_Colors[2].mask_rgb)  # separate starch in guard cell 2
            gc1_starch_masks.append(cv2.cvtColor(gc1_starch_mask, cv2.COLOR_RGB2GRAY))  # collect guard cell 1 starch masks
            gc2_starch_masks.append(cv2.cvtColor(gc2_starch_mask, cv2.COLOR_RGB2GRAY))  # collect guard cell 2 starch masks
        stack_zxy = np.stack(denoised_images, axis=0)  # stack (cleaned) images toghter
        gc1_starch_zxy = np.stack(gc1_starch_masks, axis=0)  # stack guard cell 1 starch masks toghter
        gc2_starch_zxy = np.stack(gc2_starch_masks, axis=0)  # stack guard cell 2 starch masks toghter
        gc1_starch_volume = self.starch_volume(gc1_starch_zxy, scale_zxy)  # calculate starch volume in guard cell 1
        gc2_starch_volume = self.starch_volume(gc2_starch_zxy, scale_zxy)  # calculate starch volume in guard cell 2
        return gc1_starch_zxy, gc2_starch_zxy, stack_zxy, scale_zxy, gc1_starch_volume, gc2_starch_volume
