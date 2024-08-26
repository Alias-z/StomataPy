"""Fast fourier transform (fft) augmentation"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
from math import sqrt  # for the square root
from typing import Tuple  # support typing hints
import numpy as np  # NumPy
from matplotlib import pyplot as plt  # show images and plot figures
from .core import color_select, get_contour


class FFT:
    """
    Papers:
    - Semantic-Aware Mixup for Domain Generalization (https://arxiv.org/abs/2304.05675)
    https://github.com/ccxu-ustc/SAM/blob/main/data/data_utils.py

    - A Fourier-based Framework for Domain Generalization (https://arxiv.org/abs/2105.11120)
    https://github.com/MediaBrain-SJTU/FACT/blob/main/data/data_utils.py
    """
    def __init__(self,
                 stomatal_contour_mask: np.ndarray,
                 shrink_ratio: float = 1.2,
                 line_thickness: int = 2,
                 ):
        self.stomatal_contour_mask = stomatal_contour_mask  # RGB stomatal contour mask
        self.shrink_ratio = shrink_ratio  # the shrink ratio for the mini-area bounding box
        self.line_thickness = line_thickness  # the line thickness for drawing lines

    @staticmethod
    def get_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Fourier Transform of an image to determine its frequency spectrum. This function computes both the amplitude and the phase of the Fourier Transform.

        Args:
        - image (np.ndarray): The input image for which the Fourier Transform is to be calculated

        Returns:
        - image_amplitude (np.ndarray): The amplitude spectrum of the Fourier Transform, representing the amplitude of different frequency components.
        - image_phase (np.ndarray): The phase spectrum of the Fourier Transform, indicating the phase shift associated with each frequency component
        """
        image_fft = np.fft.fft2(image)  # compute the 2D Fourier Transform of the image
        image_amplitude = np.abs(image_fft)  # (style) compute the amplitude of the complex numbers from the Fourier Transform
        image_phase = np.angle(image_fft)  # (semantic) compute the phase of the complex numbers from the Fourier Transform
        return image_amplitude, image_phase

    @staticmethod
    def get_centralized_spectrum(img):
        img_fft = np.fft.fft2(img)
        img_fft = np.fft.fftshift(img_fft)
        img_abs = np.abs(img_fft)
        img_pha = np.angle(img_fft)
        return img_abs, img_pha

    @staticmethod
    def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, alpha)

        assert img1.shape == img2.shape
        h, w, c = img1.shape
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(0, 1))
        img2_fft = np.fft.fft2(img2, axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
        img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
        img21 = np.uint8(np.clip(img21, 0, 255))
        img12 = np.uint8(np.clip(img12, 0, 255))

        return img21, img12

