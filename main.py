import os
import argparse
from typing import Literal
from stomatapy.core.core import get_paths
from stomatapy.utils.isat2excel import json2excel
from stomatapy.models.openmmlab import OpenMMlab
from stomatapy.inference_api.starch_inference import StarchSeeker

image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']
video_types = ['.avi', '.mp4', '.mov', '.wmv']


class Inferencer:
    def __init__(self,
                 scale: float = 4.3,
                 density_model: str = 'StomataPy400K_density_betatest_n387',
                 use_sahi: bool = True,
                 stack_input: bool = False,
                 check_straight_edges: bool = False,
                 straight_line_threshold: float = 100,
                 detector_threshold: float = 0.5,
                 starch_batch_size: int = 33,
                 show_prediction: bool = False,
                 output_dir: str = 'Results'
                 ):
        self.scale = scale
        self.density_model = density_model
        self.use_sahi = use_sahi
        self.stack_input = stack_input
        self.check_straight_edges = check_straight_edges
        self.straight_line_threshold = straight_line_threshold
        self.detector_threshold = detector_threshold
        self.starch_batch_size = starch_batch_size
        self.show_prediction = show_prediction
        self.output_dir = output_dir

    def infer(self, aim: Literal['Density', 'Aperture', 'Starch', 'To_Excel'] = 'Density', input_dir: str = None):
        """
        Infer images and return the result
        Args:
            aim: Literal['Density', 'Aperture', 'Starch', 'To_Excel'] = 'Density'
            input_dir: str = None
        Returns:
            None
        """

        image_paths = [path for ext in image_types + video_types for path in get_paths(input_dir, ext)]
        json_paths = [os.path.splitext(image_path)[0] + '.json' for image_path in image_paths]

        if aim == 'Starch':
            StarchSeeker(
                input_dir=input_dir,
                output_dir=self.output_dir,
                batch_size=self.starch_batch_size,
                detector_config_path='/content/drive/MyDrive/Applications/Configs/INSTANCE_mask2former_swin-s.py',
                detector_weight_path='/content/drive/MyDrive/Applications/Weights/INSTANCE_BOTH_mask2former_swin-s_2023.05.26.pth',
                detector_threshold=self.detector_threshold,
                segmentor_config_path='/content/drive/MyDrive/Applications/Configs/SEMANTIC_mask2former_swin-I.py',
                segmentor_weight_path='/content/drive/MyDrive/Applications/Weights/SEMANTIC_BOTH_mask2former_swin-I_2023.05.27.pth',
                concatenate_excels=True).batch_predict()
            return None

        else:
            models = OpenMMlab(
                detector_config_path='train/config/det_rein_dinov2_mask2former.py',
                detector_weight_path=f'Checkpoints/{self.density_model}/dinov2_detector.pth',
                detector_threshold=self.detector_threshold,
                segmentor_config_path='Checkpoints/StomataPy400K_aperture_512/seg_rein_dinov2_mask2former.py',
                segmentor_weight_path='Checkpoints/StomataPy400K_aperture_512/dinov2_segmentor.pth',
                use_sahi=self.use_sahi,
                stack_input=self.stack_input,
                check_straight_edges=self.check_straight_edges,
                straight_line_threshold=self.straight_line_threshold
            )
            if aim == 'Density':
                _ = models.detect_cell(image_paths, if_visualize=False, if_auto_label=True)
            elif aim == 'Aperture':
                _ = models.segment_cell(json_paths, if_visualize=False, if_auto_label=True, resize_to=(2048, 2048))
            elif aim == 'To_Excel':
                output_dir = os.path.join(input_dir, 'predictions')
                _ = json2excel(input_dir, output_dir, scale=self.scale, show_prediction=self.show_prediction)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim', type=str, default='Density')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='Results')
    parser.add_argument('--scale', type=float, default=4.3)
    parser.add_argument('--density_model', type=str, default='StomataPy400K_density_betatest_n387')
    parser.add_argument('--use_sahi', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--stack_input', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--check_straight_edges', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--show_prediction', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--straight_line_threshold', type=float, default=100)
    args = parser.parse_args()
    print(args)
    inferencer = Inferencer(
        scale=args.scale,
        density_model=args.density_model,
        use_sahi=args.use_sahi == 'True',
        stack_input=args.stack_input == 'True',
        check_straight_edges=args.check_straight_edges == 'True',
        straight_line_threshold=args.straight_line_threshold,
        show_prediction=args.show_prediction == 'True',
        output_dir=args.output_dir
    )
    inferencer.infer(aim=args.aim, input_dir=args.input_dir)
