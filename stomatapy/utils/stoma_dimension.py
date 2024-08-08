"""Module providing functions to get stomatal lenth and width"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import math  # for sqrt
import copy  # for deepcopy
import cv2  # OpenCV
import numpy as np  # NumPy
from matplotlib import pyplot as plt  # show images and plot figures
from ..core.core import color_select, get_contour


class GetDiameter:
    """Get the length and width of a stomatal contour mask (white RGB)"""
    def __init__(self,
                 stomatal_contour_mask: np.ndarray,
                 shrink_ratio: float = 1.2,
                 line_thickness: int = 2,
                 ):
        self.stomatal_contour_mask = stomatal_contour_mask  # RGB stomatal contour mask
        self.shrink_ratio = shrink_ratio  # the shrink ratio for the mini-area bounding box
        self.line_thickness = line_thickness  # the line thickness for drawing lines

    def get_bbox_rotate(self, show_bbox: bool = False, show_bbox_shrink: bool = False) -> tuple:
        """Find the rotated min-area bounding box for a stomatal contour mask"""
        stoma_contour = color_select(self.stomatal_contour_mask, self.stomatal_contour_mask, [255, 255, 255])  # get only white cell wall [255, 255, 255]
        stoma_center = copy.deepcopy(stoma_contour)  # maks a copy of our cell wall image
        white_pixels = np.where(get_contour(stoma_contour, thickness=4)[0][:, :, 0] == 255)  # get the location of cell wall
        stoma_xy_int = np.column_stack((white_pixels[1], white_pixels[0])).reshape(-1, 2)  # int points of the stoma contour
        bbox_rotate = cv2.minAreaRect(get_contour(stoma_contour)[1])  # find the rotated mim-area bounding box
        points = np.intp(cv2.boxPoints(bbox_rotate))  # convert bbox points to int
        cv2.drawContours(self.stomatal_contour_mask, [points], 0, (255, 0, 0), self.line_thickness)  # draw the bbox on the cell wall image
        if show_bbox:
            plt.imshow(self.stomatal_contour_mask); plt.show()  # noqa: for debugging
        (center_x, center_y), (width, height), angle = bbox_rotate  # get bbox parameters
        bbox_rotate_shrink = ((center_x, center_y), (width // self.shrink_ratio, height // self.shrink_ratio), angle)  # shrink the bbox closer to center
        shrink_points = np.intp(cv2.boxPoints(bbox_rotate_shrink))  # get the points of shrink bbox
        cv2.drawContours(stoma_center, [shrink_points], 0, (255, 0, 0), cv2.FILLED)  # draw the shrink bbox on the cell wall image in red
        stoma_center = color_select(stoma_contour, stoma_center, (255, 0, 0))  # select regions within the shrink bbox
        stoma_center = get_contour(stoma_center)[0]  # keep the max contour only. For starch, it will be pore with cell wall; For aperture, it will be only pore
        if show_bbox_shrink:
            plt.imshow(stoma_center); plt.show()  # noqa: for debugging
        if width > height:
            width, height = height, width  # assumes height is longer than width
        return stoma_xy_int, stoma_center, width, height, points

    def pca(self, show_result: bool = False) -> dict:
        """PCA for a given colored RGB mask to find symmetrical lines"""
        stoma_xy_int, stoma_center, width, height, _ = self.get_bbox_rotate()  # get results from shrinked mini-area bbox
        white_pixels = np.where(stoma_center[:, :, 0] == 255)  # extract the cell wall within
        stoma_center_xy = np.column_stack((white_pixels[1], white_pixels[0])).astype(np.float32)  # flatten the contour array and use PCA:
        mean = np.mean(stoma_center_xy, axis=0); stoma_center_xy -= mean  # noqa: subtract the mean to center the data
        cov = np.cov(stoma_center_xy, rowvar=False)  # calculate covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # calculate the eigenvalues and eigenvectors of the covariance matrix
        sorted_indices = np.argsort(eigenvalues)[::-1]  # sort the eigenvalues in descending order and get the indices
        principal_component = eigenvectors[:, sorted_indices[0]]  # get the largest eigenvector (principal component)
        smallest_component = eigenvectors[:, sorted_indices[1]]  # get the smallest eigenvector
        angle_principal_component = np.arctan2(principal_component[1], principal_component[0]) * 180 / np.pi  # the angle of the principal component (in degrees) relative to the x-axis
        lenghth_diagonal = math.sqrt(width ** 2 + height ** 2)  # find the diagonal line lenghth of that bbox
        start_point = tuple(np.round(mean - lenghth_diagonal * principal_component).astype(int))  # initial starting point along the principal_component axis
        end_point = tuple(np.round(mean + lenghth_diagonal * principal_component).astype(int))  # initial ending point along the principal_component axis
        start_point2 = tuple(np.round(mean - lenghth_diagonal * smallest_component).astype(int))  # initial starting point along the smallest_component axis
        end_point2 = tuple(np.round(mean + lenghth_diagonal * smallest_component).astype(int))  # initial ending point along the smallest_component axis

        def common_points(start_point: tuple, end_point: tuple, thickness: int = 2, step: int = 1) -> tuple:
            """Find the intersection points between length and width and the stomatal contour"""
            stoma_set = set(map(tuple, stoma_xy_int))  # convert stoma_xy_int to a set for faster look-up
            x_1, y_1 = start_point; x_2, y_2 = end_point; intersection = []  # noqa: x, y coordinates of the starting and ending points
            d_x = x_2 - x_1; d_y = y_2 - y_1  # noqa: the distance on x and y axis
            line_length = int(np.sqrt(d_x ** 2 + d_y ** 2))  # the Euclidean distance between starting and ending points
            indices = np.arange(0, line_length + 1, step)
            point_x = x_1 + indices * d_x / line_length  # the x coordinate of a point along the line defined by the start and end points
            point_y = y_1 + indices * d_y / line_length  # the y coordinate of a point along the line defined by the start and end points
            j_values = np.arange(-thickness, thickness + 1, step)
            if d_x == 0:  # fix x, move y to create vertical lines
                x_offsets = np.repeat(point_x[:, np.newaxis], len(j_values), axis=1)
                y_offsets = point_y[:, np.newaxis] + j_values
            elif d_y == 0:  # fix y, move x to create horizontal lines
                x_offsets = point_x[:, np.newaxis] + j_values
                y_offsets = np.repeat(point_y[:, np.newaxis], len(j_values), axis=1)
            else:
                angle = np.arctan2(d_y, d_x) - np.pi / 2  # perpendicular to the line
                x_offsets = point_x[:, np.newaxis] + j_values * np.cos(angle)
                y_offsets = point_y[:, np.newaxis] + j_values * np.sin(angle)
            x_offsets = np.round(x_offsets).astype(int)  # the x coordinate of a point that is j units away from a (x, y),along a direction specified by the angle
            y_offsets = np.round(y_offsets).astype(int)  # the y coordinate of a point that is j units away from a (x, y),along a direction specified by the angle
            intersection = [(x, y) for x, y in zip(x_offsets.ravel(), y_offsets.ravel()) if (x, y) in stoma_set]  # collect intersection points
            if len(intersection) >= 2:
                points_array = np.array(intersection)  # create a numpy array with all the intersection points
                distances = np.linalg.norm(points_array[:, np.newaxis] - points_array, axis=-1)  # calculate the Euclidean distance between all the intersection points
                indices = np.unravel_index(np.argmax(distances), distances.shape)  # get the index of the intersection point with the maximum Euclidean distance
                p_1 = np.array(intersection[indices[0]])  # get the intersection point with the maximum Euclidean distance
                p_2 = np.array(intersection[indices[1]])  # get another intersection point with the maximum Euclidean distance
                distance = np.linalg.norm(p_2 - p_1)  # the maximum Euclidean distance between the intersection points
                return distance, p_1, p_2
            else:
                return 0, np.array([0, 0]), np.array([0, 0])  # return the (0, 0)

        stoma_lenghth, p_1, p_2 = common_points(start_point, end_point)  # stoma_lenghth along the principal_component axis
        cv2.line(self.stomatal_contour_mask, p_1, p_2, (0, 255, 0), self.line_thickness)  # draw the length
        stoma_width, p_3, p_4 = common_points(start_point2, end_point2)  # stoma_width along the smallest_component axis
        cv2.line(self.stomatal_contour_mask, p_3, p_4, (0, 0, 255), self.line_thickness)  # draw the width
        if show_result:
            plt.imshow(self.stomatal_contour_mask); plt.show()  # noqa: for debugging
        dimension = {
            'length': stoma_lenghth,
            'length_points': (p_1, p_2),
            'width': stoma_width,
            'width_points': (p_3, p_4),
            'visualization': self.stomatal_contour_mask,
            'angle': angle_principal_component,
        }
        return dimension
