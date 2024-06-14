"""Module providing functions autolabeling images with Cellpose"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args
from typing import Literal  # to support type hints
import numpy as np  # NumPy
from cellpose import io, models, train  # noqa: import cellpose functions
from ..core.isat import UtilsISAT, ISAT2Anything, Anything2ISAT  # to interact with ISAT jason files


class CellPose:
    """
    Automatic mask generation with Cellpose https://github.com/MouseLand/cellpose
    """
    def __init__(self,
                 train_dir: str = None,
                 inference_dir: str = None,
                 test_dir: str = None,
                 model_name: str = None,
                 initial_model: str = Literal['general'],
                 n_epochs: int = 500,
                 nimg_per_epoch: int = 8,
                 channel_one: int = Literal[0, 1, 2, 3],
                 channel_two: int = Literal[0, 1, 2, 3],
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.0001,
                 flow_threshold: float = 0.4,
                 cellprob_threshold: float = 0,
                 inference_batch_size: int = 40,
                 annnotation_category: str = 'stoma'):
        self.train_dir = train_dir  # the directory of the training images
        self.inference_dir = inference_dir  # the directory of the inference images
        self.test_dir = test_dir  # the directory of the test images during training
        self.model_name = model_name  # the model name to be saved
        self.initial_model = initial_model  # the initial model the transfering learning will be based on
        self.n_epochs = n_epochs  # number of epochs during training
        self.nimg_per_epoch = nimg_per_epoch  # number of images per epoch during training
        self.channel_one = channel_one  # only the gray channels  {'None':Grayscale, 'Red': 1, 'Green':2, 'Blue':3}
        self.channel_two = channel_two  # optional second channels {'None':0, 'Red': 1, 'Green':2, 'Blue':3}
        self.channels = [self.channel_one, self.channel_two]  # summary the two channels
        self.learning_rate = learning_rate  # the training learning rate
        self.weight_decay = weight_decay  # the training weight decay
        self.flow_threshold = flow_threshold  # threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded)
        self.cellprob_threshold = cellprob_threshold  # threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6))
        self.inference_batch_size = inference_batch_size  # batch size for inference
        self.annnotation_category = annnotation_category  # the annotation category in ISAT json file

    def training(self, isat2cellpose: bool = True) -> str:
        """Traiing models based on the CellPose 'general' model"""
        if isat2cellpose:
            print('converting ISAT .json files to CellPose .npy files')
            ISAT2Anything(annotations_dir=self.train_dir).to_cellpose(category=self.annnotation_category)  # convert ISAT json files to CellPose npy file

        io.logger_setup()  # initialize the logger during training
        model = models.CellposeModel(gpu=True, model_type=self.initial_model)  # load the model into GPU
        train_data, train_labels, _, test_data, test_labels, _ = io.load_train_test_data(self.train_dir, self.test_dir, mask_filter='_seg.npy')  # load files
        model_path = train.train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            channels=self.channels,
            save_path=self.train_dir,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            SGD=True,
            nimg_per_epoch=self.nimg_per_epoch,
            model_name=self.model_name)
        model.net.diam_labels.item()  # diameter of labels in training images
        return model_path

    def inference(self, model_path, cellpose2isat: bool = True) -> None:
        """Predict labels using a pretained CellPose model"""
        files = io.get_image_files(self.inference_dir, mask_filter='_seg.npy')  # filter out '_seg.npy' files
        images = [io.imread(file) for file in files]  # load these images
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)  # load the pretrained model
        diameter = model.diam_labels  # set to zero to use diameter from training set
        io.logger_setup()  # initialize the logger during inference
        masks, flows, _ = model.eval(
            images,
            batch_size=self.inference_batch_size,
            channels=[self.channel_one, self.channel_two],
            diameter=diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold)  # run model on inference images
        for idx, mask in enumerate(masks):
            io.masks_flows_to_seg(images[idx], mask, flows[idx], files[idx], channels=self.channels, diams=diameter * np.ones(len(mask)))  # save output to *_seg.npy

        if cellpose2isat:
            print('converting CellPose .npy files to ISAT .json files')
            Anything2ISAT(annotations_dir=self.inference_dir).from_cellpose(category=self.annnotation_category, use_polydp=False, epsilon_factor=0.001)  # convert CellPose npy to ISAT json
            UtilsISAT.sort_group(json_dir=self.inference_dir)  # tidy up the ISAT json file
        return None
