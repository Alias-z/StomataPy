"""Module providing core functions"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import, dangerous-default-value
import os  # interact with the operating system
import sys  # to use system standard output
import contextlib  # for mutting prints
import glob  # Unix style pathname pattern expansion
from typing import Tuple, Optional  # support typing hints
import requests  # for downloading checkpoints
import cv2  # OpenCV
import torch  # PyTorch
import numpy as np  # NumPy
from PIL import Image, ImageOps  # Pillow image processing
from more_itertools import sort_together  # sort two list together
from skimage.filters import threshold_isodata, laplace  # for image thresholding

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # use GPU if available

image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']  # supported image types


class SegColors:
    """Define the segmentation class names, colors and their one-hot encoding"""
    def __init__(self,
                 class_name: str = None,
                 mask_rgb: np.ndarray = None,
                 class_encoding: int = None):
        self.class_name = class_name  # catergory name
        self.mask_rgb = mask_rgb  # catergory rgb color
        self.class_encoding = class_encoding  # catergory encoding


GC_Colors = [SegColors('background', [0, 0, 0], 0),
             SegColors('guard cell 1', [255, 0, 255], 1),
             SegColors('guard cell 2', [255, 255, 0], 1),
             SegColors('subsidiary cell 1', [255, 255, 0], 2),
             SegColors('subsidiary cell 2', [255, 255, 0], 2)]

Starch_Colors = [SegColors('background', [0, 0, 0], 0),
                 SegColors('starch', [255, 255, 0], 1)]

Stomata_Colors = [SegColors('background', [0, 0, 0], 0),
                  SegColors('stomata', [255, 0, 255], 1),
                  SegColors('outer ledge', [255, 170, 0], 2),
                  SegColors('pore', [255, 255, 0], 3)]  # legacy

Cell_Colors = [SegColors('background', [0, 0, 0], 0),
               SegColors('stomatal complex', [170, 0, 0], 1),
               SegColors('stoma', [231, 116, 237], 2),
               SegColors('outer ledge', [255, 245, 54], 3),
               SegColors('pore', [234, 234, 235], 4),
               SegColors('pavement cell', [255, 170, 0], 5),
               SegColors('guard cell', [85, 85, 255], 6)]


def get_paths(input_dir: str = None, file_extension='.json') -> list:
    """Get all files paths with a certain file extension Under the input folder"""
    return glob.glob(os.path.join(input_dir, f'*{file_extension}'))


def imread_rgb(image_dir: str) -> np.ndarray:
    """cv2.imread + BRG2RGB"""
    return cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)


def unique_color(image: np.ndarray) -> np.ndarray:
    """Find the unique RGB color of a given image"""
    return np.unique(image.reshape(-1, image.shape[-1]), axis=0)


def color_select(image: np.ndarray, mask: np.ndarray, color: np.ndarray = np.array([255, 0, 255])) -> np.ndarray:
    """
    Isolates and retains areas of an image that match a specified color, setting all other areas to black

    Args:
    - image (np.ndarray): the input image in which color areas are to be selected
    - mask (np.ndarray): the mask image where the color selection is based
    - color (np.ndarray): the RGB color as an array to be isolated in the image; default is magenta ([255, 0, 255])

    Returns:
    - output (np.ndarray): modified image with only the selected color areas retained and other areas set to black
    """
    binary_mask = np.all(mask == np.array(color), axis=-1)  # create a binary mask on a given color region
    non_target_indices = np.where(np.logical_not(binary_mask))  # choose non-target regions
    output = image.copy()  # prevent change in position
    output[non_target_indices] = 0  # fill the non-target regions to be black (background)
    return output


def binary(image_rgb: np.ndarray) -> np.ndarray:
    """Computer threshold isodata method to convert a RGB image to binary"""
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # RGB to gray
    isodata = threshold_isodata(image_gray)  # return threshold value(s) based on ISODATA method.
    image_binary = cv2.threshold(image_gray, isodata, 255, cv2.THRESH_BINARY)[1]  # gray to binary
    return image_binary


def get_contour(image: np.ndarray,
                foreground: Optional[np.ndarray] = None,
                color: list = (255, 255, 255),
                thickness: int = 2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts the largest contour from an image and draws it on a foreground

    Args:
    - image (np.ndarray): the input image from which contours are extracted
    - foreground (np.ndarray): the image on which contours will be drawn; if None, a new one is created
    - color (list): the color of the contour, default is white [255, 255, 255]
    - thickness (int): the thickness of the contour line

    Returns:
    - foreground image (np.ndarray): the modified foreground image, if any
    - largest_contour (np.ndarray): the largest contour
    """
    edge_laplace = laplace(binary(image))  # find the edges of an image using the Laplace operator
    edges = cv2.convertScaleAbs(edge_laplace)  # floating-point array to an 8-bit unsigned integer array
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # get external contours
    contour_area, contour_id = [], []  # to store values
    for idx, contour in enumerate(contours):
        contour_area.append(cv2.contourArea(contour))  # collect contour areas
        contour_id.append(idx)  # collect contour ids
    contour_area, contour_id = sort_together([[int(idx) for idx in contour_area], contour_id])  # sort id according to area
    if foreground is None:
        foreground = np.zeros((*image.shape[:2], 3), dtype=np.uint8)  # create a black image with same dimension
    cv2.drawContours(foreground, contours, contour_id[-1], color=color, thickness=thickness)  # the largest contour
    return foreground, contours[contour_id[-1]]


def resize_and_pad_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """
    Adjusts an image's size to fit precisely within the specified target dimensions
    Initial outward padding of 20% to each edge, then resizing to slightly larger than the target, and finally applying the necessary padding

    Args:
    - image (np.ndarray): the image object to be processed
    - target_size (Tuple[int, int], optional): the dimensions to which the image is resized and padded, defaults to (512, 512)

    Returns:
    - resized_and_padded_image (np.ndarray): the image after adjustments
    - padding_values (Tuple[int, int, int, int]): padding applied on each side
    - scale_factor (float): the scale used to adjust the image size initially
    """
    image = Image.fromarray(image)  # convert np.ndarray to pil image
    initial_padding_amount = (int(image.width * 0.2), int(image.height * 0.2))  # calculate 20% padding
    padded_image = ImageOps.expand(image, border=initial_padding_amount, fill=0)  # apply initial padding

    temp_target_size = (int(target_size[0] * 1.1), int(target_size[1] * 1.1))  # calculate 10% larger than the final target size
    scale_width = temp_target_size[0] / padded_image.width  # calculate the width scaling factor
    scale_height = temp_target_size[1] / padded_image.height  # calculate the height scaling factor
    scale = min(scale_width, scale_height)  # determine the minimum scaling factor
    new_width, new_height = int(padded_image.width * scale), int(padded_image.height * scale)  # apply the scaling factor
    resized_image = padded_image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # resize the image based on current ratio

    delta_w = target_size[0] - new_width  # calculate the width difference
    delta_h = target_size[1] - new_height  # calculate the height difference
    left_pad = delta_w // 2  # calculate left padding
    right_pad = delta_w - left_pad  # calculate right padding
    top_pad = delta_h // 2  # calculate top padding
    bottom_pad = delta_h - top_pad  # calculate bottom padding
    final_padding = (left_pad, top_pad, right_pad, bottom_pad)  # compile final padding dimensions
    final_padded_image = ImageOps.expand(resized_image, final_padding)  # apply final padding

    final_padded_image = np.array(final_padded_image)  # convert the final padded image back to numpy array
    return final_padded_image, final_padding, scale


def restore_original_dimensions(image: Image.Image, padding: Tuple[int, int, int, int], scale: float) -> Image.Image:
    """
    Restore an image to its original dimensions before it was resized and padded
    Reversing an initial outward padding of 20% applied to each edge, and the subsequent padding and scaling
    Args:
    - image (Image.Image): the pil image object to be restored
    - padding (Tuple[int, int, int, int]): the padding value from each side of the image as (left_pad, top_pad, right_pad, bottom_pad)
    - scale (float): the scale factor used when the image was resized, to reverse the resizing effect

    Returns:
    - restored_image (Image.Image): the pil image object after reversing the padding and scaling
    """
    # calculate coordinates for cropping to remove the additional padding first
    initial_padding = int(max(image.width, image.height) * 0.2)  # 20% of the larger dimension for consistent initial padding
    crop_left = padding[0] + initial_padding  # add initial padding to left
    crop_top = padding[1] + initial_padding  # add initial padding to top
    crop_right = image.width - (padding[2] + initial_padding)  # subtract initial padding from right
    crop_bottom = image.height - (padding[3] + initial_padding)  # subtract initial padding from bottom

    # crop the image to remove all padding
    cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    # calculate the original dimensions by reversing the scaling
    original_width = int(cropped_image.width / scale)  # reverse width scaling based on original scale
    original_height = int(cropped_image.height / scale)  # reverse height scaling based on original scale

    # resize the cropped image back to its original dimensions
    restored_image = cropped_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

    return restored_image


def get_checkpoints(download_url: str = None, destination_file_path: str = None) -> bool:
    """
    Downloads a file from a specified URL to a local file path
    Ensures that required model checkpoint files are locally available for subsequent processing tasks

    Args:
    - download_url (str): the URL from where the file should be downloaded
    - destination_file_path (str): the local path where the downloaded file should be saved

    Returns:
    - bool: returns True if the file was downloaded and saved successfully, False otherwise
    """
    response = requests.get(download_url, allow_redirects=True, timeout=200)  # download checkpoints from url
    if response.status_code == 200:
        with open(destination_file_path, 'wb') as file:
            file.write(response.content)  # save the retrieved content
        return True
    else:
        print(f'Failed to download the file. HTTP status code: {response.status_code}')
        return False


@contextlib.contextmanager
def suppress_stdout() -> any:
    """To mute a given function's standard output"""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        previous_stdout = sys.stdout  # save the current stdout to restore it later
        sys.stdout = devnull  # redirect stdout to devnull
        try:
            yield  # allowing the code within the 'with' block to run.
        finally:
            sys.stdout = previous_stdout  # set sys.stdout back to its original value
            sys.stdout.flush()  # minimize the chance of leftover whitespace characters
    return None


def lab_logo() -> None:
    """print the lab logo"""
    logo = """
      _____ ____ ____  ______   ___ _     ____  ____      _      ____ ____  
     / ___//    |    \|      | /  _| |   |    |/    |    | |    /    |    \ 
    (   \_|  o  |  _  |      |/  [_| |    |  ||  o  |    | |   |  o  |  o  )
     \__  |     |  |  |_|  |_|    _| |___ |  ||     |    | |___|     |     |
     /  \ |  _  |  |  | |  | |   [_|     ||  ||  _  |    |     |  _  |  O  |
     \    |  |  |  |  | |  | |     |     ||  ||  |  |    |     |  |  |     |
      \___|__|__|__|__| |__| |_____|_____|____|__|__|    |_____|__|__|_____|
    """
    print(logo)
    return None
