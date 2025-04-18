"""Module checking the statistics of StomataPy datasets"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, wildcard-import, no-member, too-many-function-args, cell-var-from-loop
import os  # interact with the operating system
import json  # manipulate json files
import warnings; warnings.filterwarnings(action='ignore', category=FutureWarning, message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.')  # noqa: suppress pandas warning
import shutil  # for copy files
import pandas as pd  # for Excel sheet and CSV file
from ..core.core import get_paths  # import core functions


class DataStatistics:
    """Checking the statistics of StomataPy datasets"""
    def __init__(self, root_dir: str = 'Datasets'):
        self.root_dir = root_dir  # the root directory of all datasets
        self.reference_table_path = os.path.join(self.root_dir, 'Supplementary tables.xlsx')  # database reference

    def get_processed_dirs(self) -> tuple:
        """Find the paths of all processed datasets"""
        datasets_dirs = []  # to store processed datasets directories
        for dirpath, dirnames, _ in os.walk(self.root_dir):
            if 'Processed' in dirnames:
                datasets_dirs.append(os.path.join(dirpath, 'Processed'))  # collect all 'Processed' folder directories
        datasets_names = sorted([os.path.basename(os.path.dirname(datasets_dir)) for datasets_dir in datasets_dirs], key=str.casefold)  # sorted dataset names
        return list(set(datasets_dirs)), list(set(datasets_names))

    def get_species_names(self) -> tuple:
        """Get the set of species names of processed datasets"""
        datasets_dirs, _ = self.get_processed_dirs()  # get the directories of all processed datasets
        species_names, species_dirs = [], []  # to store the species names
        for datasets_dir in datasets_dirs:
            if os.path.isdir(datasets_dir):
                species_names.extend(os.listdir(datasets_dir))  # populate the species names
                species_dirs.extend(os.path.join(datasets_dir, species_name) for species_name in species_names)  # populate the species dirs
        species_names = list(set(species_names))  # get the unique species names
        return sorted(species_names, key=str.casefold), list(set(species_dirs))

    def select_species_folders(self, pavements_only: bool = False, semantic: bool = False, ensemble_files: bool = False, ensemble_by_modality: bool = True) -> dict:
        """Retuern the summarized information cross all selected species folders"""
        def check_dataset(cell, dataset):
            datasets = [dataset.strip() for dataset in cell.split(';')]
            return dataset in datasets

        if ensemble_files:
            destination_root = os.path.join(self.root_dir, 'Ensemble')  # the directory of ensembled files
            os.makedirs(destination_root, exist_ok=True)  # create the ensemble files directory

        reference = pd.read_excel(self.reference_table_path, sheet_name='S2. Plant species', header=13)

        results = pd.DataFrame()

        category_counts, selected_datasets, selected_jsons, image_modalities = {}, [], {}, []  # to store the maks counts of each category
        n_masks, n_autolabel, adjusted_n_stomata, n_images = 0, 0, 0, 0  # to count the number of masks

        species_names, species_folder_dirs = self.get_species_names()  # get the species folders directories
        for species_folder_dir in species_folder_dirs:
            json_paths = get_paths(species_folder_dir, 'json')  # get the paths of ISAT annotation files
            dataset_name = os.path.normpath(species_folder_dir).split(os.sep)[1]  # get the dataset name
            for json_path in json_paths:
                # print(json_path)
                with open(json_path, encoding='utf-8') as file:
                    data = json.load(file)  # load the json data

                image_name = data['info'].get('name', '')  # get the image name
                image_path = json_path.replace('.json', os.path.splitext(image_name)[1])  # get the image path info

                image_width = data['info'].get('width', '')  # get the image width
                image_height = data['info'].get('height', '')  # get the image height
                note = data['info'].get('note', '')  # get note and convert to lower case
                if '_' not in note:
                    continue

                categories = set([obj['category'] for obj in data['objects']])  # get all categories

                if pavements_only:
                    if 'pavement cell' not in categories:
                        continue
                    if len(categories) == 1:
                        continue

                if semantic:
                    if 'outer ledge' not in categories:
                        continue

                n_images += 1  # to count the number of images
                stomata_type, sampling_method, microscopy, image_quality, image_scale = note.split('_')  # note format
                image_scale = pd.NA if image_scale.strip() == 'NA' else float(image_scale.strip())
                image_modality = f'{sampling_method}_{microscopy}'  # get the image modality
                image_modalities.append(image_modality)  # collect the image modalities

                if ensemble_files:
                    if ensemble_by_modality:
                        destination_dir = os.path.join(destination_root, image_modality)
                        if not os.path.exists(destination_dir):
                            os.makedirs(destination_dir, exist_ok=True)  # create the modality folder if needed
                    else:
                        destination_dir = destination_root  # use default directory if not grouping by image modality
                    shutil.copy2(image_path, os.path.join(destination_dir, os.path.basename(image_path)))  # copy the image to the ensembled files directory
                    shutil.copy2(json_path, os.path.join(destination_dir, os.path.basename(json_path)))  # copy the json file to the ensembled files directory

                species = os.path.basename(species_folder_dir)  # get the species name
                dataset_species = f'{dataset_name} - {species}'
                if species_folder_dir not in selected_datasets:
                    selected_datasets.append(dataset_species)

                matching_rows = reference[reference['Species'] == species]
                matching_rows = matching_rows[matching_rows['Datasets'].apply(lambda x: check_dataset(x, dataset_name))]  # noqa
                lineage = matching_rows.iloc[0]['Lineage'] if not matching_rows.empty else pd.NA
                clade = matching_rows.iloc[0]['Clade'] if not matching_rows.empty else pd.NA
                family = matching_rows.iloc[0]['Family'] if not matching_rows.empty else pd.NA
                genus = matching_rows.iloc[0]['Genus'] if not matching_rows.empty else pd.NA

                n_stomatal_complex, n_stoma, n_outer_ledge, n_pore, n_pavment_cell = 0, 0, 0, 0, 0   # to count number of each catergory
                stomatal_complex_areas, stomatal_areas, pavment_cell_areas = [], [], []

                for obj in data['objects']:
                    category = obj['category']
                    note = obj['note']
                    if note == 'Auto':
                        n_autolabel += 1
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1
                    if category == 'stomatal complex':
                        n_stomatal_complex += 1
                        stomatal_complex_areas.append(float(obj['area']))
                    elif category == 'stoma':
                        n_stoma += 1
                        stomatal_areas.append(float(obj['area']))
                    elif category == 'outer ledge':
                        n_outer_ledge += 1
                    elif category == 'pore':
                        n_pore += 1
                    elif category == 'pavement cell':
                        n_pavment_cell += 1
                        pavment_cell_areas.append(float(obj['area']))

                if n_stoma > 0 and n_stomatal_complex > 0:
                    adjusted_n_stomata = n_stomatal_complex
                elif n_stoma > 0 and n_stomatal_complex == 0:
                    adjusted_n_stomata = n_stoma
                elif n_stoma == 0 and n_stomatal_complex > 0:
                    adjusted_n_stomata = n_stomatal_complex

                mean_stomatal_complex_areas = sum(stomatal_complex_areas) / len(stomatal_complex_areas) * (1 / image_scale) ** 2 if stomatal_complex_areas else pd.NA
                mean_stomatal_area = sum(stomatal_areas) / len(stomatal_areas) * (1 / image_scale) ** 2 if stomatal_areas else pd.NA
                mean_pavment_cell_areas = sum(pavment_cell_areas) / len(pavment_cell_areas) * (1 / image_scale) ** 2 if pavment_cell_areas else pd.NA
                stomatal_index = adjusted_n_stomata / n_pavment_cell if n_pavment_cell != 0 else pd.NA

                result = {
                    'Image name': image_name,
                    'image_path': image_path,
                    'Sampling method': sampling_method,
                    'Microscopy': microscopy,
                    'Image quality': image_quality,
                    'Image scale (pixels/\u03BCm)': image_scale,
                    'Image width': image_width,
                    'Image height': image_height,
                    'Dataset': dataset_name,
                    'Species': species,
                    'Lineage': lineage,
                    'Clade': clade,
                    'Family': family,
                    'Genus': genus,
                    'Stomata type': stomata_type,
                    'Mean stomatal area (\u03BCm\N{SUPERSCRIPT TWO})': mean_stomatal_area,
                    'Mean stomatal complex area (\u03BCm\N{SUPERSCRIPT TWO})': mean_stomatal_complex_areas,
                    'Mean pavement cell area (\u03BCm\N{SUPERSCRIPT TWO})': mean_pavment_cell_areas,
                    'Stomatal density (mm\u207B\u00B2)': adjusted_n_stomata / (image_height * image_width * (1 / image_scale) ** 2) * 1e6,
                    'Stomatal index': stomatal_index,
                    'No. stomatal complex': n_stomatal_complex,
                    'No. stoma': n_stoma,
                    'No. outer ledge': n_outer_ledge,
                    'No. pore': n_pore,
                    'No. pavement cell': n_pavment_cell
                }

                if adjusted_n_stomata > 0:
                    if dataset_species not in selected_jsons:
                        selected_jsons[dataset_species] = [result]
                    else:
                        selected_jsons[dataset_species].append(result)

                results = pd.concat([results, pd.DataFrame([result])], axis=0)  # concatenate all results
                n_masks += len(data['objects'])
        results.to_excel(os.path.join(self.root_dir, 'Dataset_summary.xlsx'), index=False, na_rep='NA')  # save the results in Excel
        print(f'Total images: {n_images}, total plant species: {len(set(species_names))}, total_modalities: {len(set(image_modalities))}')
        print(f'Total masks: {n_masks}, {n_autolabel} ({round(n_autolabel / n_masks * 100, 2)} %) out of which is autolabeled')
        print('Category counts:', category_counts)
        return selected_jsons

    @staticmethod
    def dataset_filter(dataset_root: str = None, pavements_only: bool = False, semantic: bool = False, ensemble_by_modality: bool = True) -> dict:
        """
        Filters and copies files from StomataPy400K dataset based on specified criteria.

        Args:
            dataset_root (str, optional): Root directory of the dataset to filter.
            pavements_only (bool, default=False): If True, only keep files containing pavement cells.
            semantic (bool, default=False): If True, only keep files containing outer ledges.
            ensemble_by_modality (bool, default=True): If True, organize filtered files by modality in separate folders.

        """

        print(f'Filtering dataset: {dataset_root}')
        destination_root = dataset_root + '_filtered'  # the directory of filtered files
        os.makedirs(destination_root, exist_ok=True)  # create the destination directory

        # get all subfolder directories under dataset_root
        subfolder_dirs = []
        for root, dirs, _ in os.walk(dataset_root):
            for dir_name in dirs:
                subfolder_dir = os.path.join(root, dir_name)
                if os.path.isdir(subfolder_dir):
                    subfolder_dirs.append(subfolder_dir)

        for subfolder_dir in subfolder_dirs:
            json_paths = get_paths(subfolder_dir, 'json')  # get the paths of ISAT annotation files
            if not json_paths:  # skip folders with no JSON files
                continue

            for json_path in json_paths:
                # print(json_path)
                with open(json_path, encoding='utf-8') as file:
                    data = json.load(file)  # load the json data

                image_name = data['info'].get('name', '')  # get the image name
                image_path = json_path.replace('.json', os.path.splitext(image_name)[1])  # get the image path info

                note = data['info'].get('note', '')  # get note and convert to lower case
                if '_' not in note:
                    continue

                categories = set([obj['category'] for obj in data['objects']])  # get all categories

                if pavements_only:
                    if 'pavement cell' not in categories:
                        continue

                if semantic:
                    if 'outer ledge' not in categories:
                        continue

                if ensemble_by_modality:
                    destination_dir = os.path.join(destination_root, os.path.basename(subfolder_dir))
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir, exist_ok=True)  # create the modality folder if needed
                else:
                    destination_dir = destination_root  # use default directory if not grouping by image modality
                shutil.copy2(image_path, os.path.join(destination_dir, os.path.basename(image_path)))  # copy the image to the ensembled files directory
                shutil.copy2(json_path, os.path.join(destination_dir, os.path.basename(json_path)))  # copy the json file to the ensembled files directory
        return None
