"""Module providing functions clustering image features"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args, wrong-import-position
import os  # interact with the operating system
import random  # random number generator
from typing import Literal  # to support type hints
import numpy as np  # NumPy
from tqdm import tqdm  # progress bar
import hdbscan  # Hierarchical Density-Based Spatial Clustering of Applications with Noise
from sklearn.cluster import KMeans  # KMeans clustering
import torch  # PyTorch
from PIL import Image  # Pillow image processing
from matplotlib import pyplot as plt  # for image visualization
from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, AutoModel  # import CLIP and DINOv2
from ..core.core import device, image_types, imread_rgb  # import core elements


class FeatureClustering:
    """Group image based on the similarity of their features"""
    def __init__(self,
                 root_dir: str = 'Datasets',
                 n_sample: int = 20):
        self.root_dir = root_dir  # the root directory of datasets
        self.n_sample = n_sample  # number of sampled images from each dataset

    @staticmethod
    def extract_features(image: np.ndarray, model_type: Literal['CLIP', 'DINOv2'] = 'DINOv2') -> float:
        """Extract image features with CLIP (https://github.com/openai/CLIP) or DINOv2 (https://github.com/facebookresearch/dinov2)"""
        if model_type == 'CLIP':
            processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')  # get the CLIP model processor
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)  # get the CLIP model
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                image_inputs = processor(images=image, return_tensors='pt').to(device)  # image inputs
                image_features = model.get_image_features(**image_inputs)  # extract features from the image
        elif model_type == 'DINOv2':
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')  # get the DINOv2 model processor
            model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)  # get the DINOv2 model
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                image_inputs = processor(images=image, return_tensors='pt').to(device)  # image inputs
                image_features = model(**image_inputs).last_hidden_state.mean(dim=1)  # extract features from the image
        return image_features.squeeze().cpu().numpy()  # torch tensor pt (features) ~ csv (image_names) HDBSCRAN

    def get_sample_features(self, save_features: bool = True) -> dict:
        """Get a list of sampled image paths for clutering"""
        def get_processed_paths() -> list:
            """Get the species directory paths of all processed folders"""
            processed_dir_paths = []  # to store the paths
            for dir_path, dir_names, _ in os.walk(self.root_dir):
                if "Processed" in dir_names:
                    dir_path = os.path.join(dir_path, "processed")  # create the full path to the 'processed' directory
                    processed_dir_paths.append(dir_path)  # populate the processed directory paths list
            return processed_dir_paths

        @staticmethod
        def image_sampler(processed_dir_paths: list, n_sample: int = 20) -> dict:
            """Sample images from all processed_paths for clutering"""
            image_samples = {}  # to store the image paths
            for processed_dir_path in processed_dir_paths:
                dir_image_paths = []
                print(processed_dir_path)
                for dirpath, _, filenames in os.walk(processed_dir_path):
                    for filename in filenames:
                        if os.path.splitext(filename)[1].lower() in image_types:
                            image_path = os.path.join(dirpath, filename)  # create the full path to the image file
                            dir_image_paths.append(image_path)  # poplulate the list of image paths under a given directory
                if len(dir_image_paths) >= n_sample:
                    sampled_paths = random.sample(dir_image_paths, n_sample)
                else:
                    sampled_paths = random.choices(dir_image_paths, k=n_sample)  # if there are fewer images than n_sample, return them all or sample with replacement

                path_parts = os.path.normpath(processed_dir_path).split(os.sep)  # extract the specific part of the directory path
                if len(path_parts) > 1:
                    dir_key = path_parts[1]  # the second part of the path
                else:
                    dir_key = path_parts[0]  # fallback to the first part if the path is not as expected
                images = [imread_rgb(image_path) for image_path in sampled_paths]  # load all images
                features = np.array([FeatureClustering.extract_features(image) for image in tqdm(images, total=len(images))])  # get all image features
                image_samples[dir_key] = {'image_paths': sampled_paths, 'features': features}  # add the directory as key to sampled image paths
            return image_samples

        image_samples = image_sampler(get_processed_paths(), self.n_sample)
        if save_features:
            torch.save(image_samples, os.path.join(self.root_dir, 'extracted_features.pth'))  # save image paths and features
        return image_samples

    @staticmethod
    def load_torch_data(features_path: str) -> dict:
        """Load the saved features and image paths from a PyTorch file"""
        return torch.load(features_path)

    @staticmethod
    def cluster_images_hdbscan(image_samples: dict, min_cluster_size: int = 30, min_samples: int = 15, n_show: int = 20) -> float:
        """Cluster images features based on HDBSCAN"""

        def visualize_clusters(image_paths: list, dataset_keys: list, cluster_labels: np.ndarray) -> None:
            """Visualize n random images from each cluster"""
            unique_labels = np.unique(cluster_labels)  # get unique labels
            unique_labels_sorted = np.sort(unique_labels)  # sort the labels to start from -1 (noise)
            for label in unique_labels_sorted:
                cluster_indices = [idx for idx, cluster in enumerate(cluster_labels) if cluster == label]  # filter image paths for the current cluster
                cluster_image_paths = [image_paths[idx] for idx in cluster_indices]  # get the image paths of the current cluster
                cluster_dataset_keys = [dataset_keys[idx] for idx in cluster_indices]  # get the dataset names
                print(f"Cluster {label} ({len(cluster_image_paths)} images):")  # display cluster info
                if len(cluster_image_paths) > n_show:
                    sample_indices = random.sample(range(len(cluster_image_paths)), n_show)  # radomly select a subset of images for visualization
                else:
                    sample_indices = range(len(cluster_image_paths))
                _, axs = plt.subplots(1, len(sample_indices), figsize=(len(sample_indices) * 4, 4))  # create a figure for the current cluster
                if len(sample_indices) == 1:
                    axs = [axs]  # if there's only one image, axs might not be an array, so we put it in a list for consistent handling
                for ax, idx in zip(axs, sample_indices):
                    image = Image.open(cluster_image_paths[idx])
                    ax.imshow(image); ax.axis('off'); ax.set_title(f'{cluster_dataset_keys[idx]}')  # noqa: hide axes ticks
                plt.show()
            return None

        def sample_images(image_paths: list, cluster_labels: np.ndarray) -> list:
            """Sample images from each cluster based on the median cluster size"""
            unique_labels = np.unique(cluster_labels)  # get unique labels
            clusters_image_paths = {label: [] for label in unique_labels}  # dictionary to hold image paths for each cluster
            for path, label in zip(image_paths, cluster_labels):
                clusters_image_paths[label].append(path)  # populate the dictionary with image paths for each cluster
            median_size = np.median([len(paths) for label, paths in clusters_image_paths.items() if label != -1])  # calculate the median cluster size, excluding the noise cluster (-1)
            print(f'median size size = {median_size}')
            sampled_image_paths = []  # to store selected samples
            for label, paths in clusters_image_paths.items():
                if len(paths) < median_size or label == -1:
                    sampled_image_paths.extend(paths)  # include all images if the cluster is smaller than the median or from noise cluster
                else:
                    sampled_image_paths.extend(random.sample(paths, int(median_size)))  # sample to the median size if the cluster is larger than the median
            return sampled_image_paths

        all_features, all_image_paths, dataset_keys = [], [], []  # Keep track of the dataset each image belongs to
        # for dir_key, data in image_samples.items():
        #     all_features.append(data['features'])
        #     all_image_paths.extend(data['image_paths'])
        #     dataset_keys.extend([dir_key] * len(data['image_paths']))  # repeat dir_key for each image
        # all_features = np.vstack(all_features)  # concatenate all features into one array for clustering

        for dataset_species, metadata_list in tqdm(image_samples.items(), total=len(image_samples.items())):
            for metadata in metadata_list:
                all_features.append(metadata['image_features'])
                all_image_paths.append(metadata['image_path'])
                dataset_keys.append(dataset_species)  # repeat dir_key for each image
        all_features = np.vstack(all_features)  # concatenate all features into one array for clustering

        clusterer = hdbscan.HDBSCAN(metric='manhattan', min_cluster_size=min_cluster_size, min_samples=min_samples)  # define the HDBSCAN parameters
        cluster_labels = clusterer.fit_predict(all_features)  # get the cluster labels
        sampled_image_paths = sample_images(all_image_paths, cluster_labels)  # the selected image paths
        visualize_clusters(all_image_paths, dataset_keys, cluster_labels)  # visualize images and number of each cluster
        return sampled_image_paths

    @staticmethod
    def compute_similarity_matrix(features: np.ndarray) -> dict:
        """ Compute the similarity matrix from the extracted features """
        features_norm = np.stack(features) / np.linalg.norm(features, axis=1, keepdims=True)  # normalize the feature vectors to unit length
        similarity_matrix = np.dot(features_norm, features_norm.T)  # compute pairwise cosine similarity
        return similarity_matrix

    def cluster_images_kmeans(self, image_samples: dict, k: int = 10, n_show: int = 20) -> float:

        def cluster_images(self, features: np.ndarray) -> list:
            """Cluster images based on features similarity"""
            features = [self.extract_features(image) for image in images]  # etract features from all images
            similarity_matrix = self.compute_similarity_matrix(features)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(similarity_matrix)  # apply K-Means clustering on the similarity matrix
            image_clusters = {}  # to store final clustering result
            for idx, label in enumerate(kmeans.labels_):
                if label not in image_clusters:
                    image_clusters[label] = []  # populate cluster labels
                image_clusters[label].append(idx)  # group image indices by labels
            return image_clusters
        
        
