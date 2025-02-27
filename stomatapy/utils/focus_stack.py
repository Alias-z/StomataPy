"""Module providing functions to do focus stacking"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import cv2  # OpenCV
import numpy as np  # NumPy


def focus_stack(stack: np.ndarray, save_output: bool = False):
    """
    Code modified from https://github.com/cmcguinness/focusstack

    Simple Focus Stacker

        Author:     Charles McGuinness (charles@mcguinness.us)
        Copyright:  Copyright 2015 Charles McGuinness
        License:    Apache License 2.0


    This code will take a series of images and merge them so that each
    pixel is taken from the image with the sharpest focus at that location.

    The logic is roughly the following:

    1.  Align the images.  Changing the focus on a lens, even
        if the camera remains fixed, causes a mild zooming on the images.
        We need to correct the images so they line up perfectly on top
        of each other.

    2.  Perform a gaussian blur on all images

    3.  Compute the laplacian on the blurred image to generate a gradient map

    4.  Create a blank output image with the same size as the original input
        images

    4.  For each pixel [x,y] in the output image, copy the pixel [x,y] from
        the input image which has the largest gradient [x,y]

    This algorithm was inspired by the high-level description given at

    http://stackoverflow.com/questions/15911783/what-are-some-common-focus-stacking-algorithms

    """
    def do_lap(image):
        """
        Apply a Gaussian blur and then Laplacian to measure focus.

        Parameters:
        image (np.ndarray): Grayscale image array.

        Returns:
        np.ndarray: Laplacian-filtered image highlighting edges/focus.
        """
        # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
        kernel_size = 5         # Size of the laplacian window
        blur_size = 5           # How big of a kernal to use for the gaussian blur

        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

    if stack is None or len(stack) == 0:
        raise ValueError("Stack is empty or not initialized")

    # Convert images to grayscale and compute Laplacian
    laps = []
    for i in range(len(stack)):
        gray = cv2.cvtColor(stack[i], cv2.COLOR_BGR2GRAY)
        laps.append(do_lap(gray))

    laps = np.asarray(laps)
    output = np.zeros(shape=stack[0].shape, dtype=stack[0].dtype)

    # Find pixels with maximum focus
    abs_laps = np.absolute(laps)
    maxima = abs_laps.max(axis=0)
    bool_mask = abs_laps == maxima
    mask = bool_mask.astype(np.uint8)

    # Create output image using the masks
    for i in range(len(stack)):
        output = cv2.bitwise_not(stack[i], output, mask=mask[i])

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    if save_output:
        cv2.imwrite("focus_stack_result.png", output_rgb)

    return output_rgb
