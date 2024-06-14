import os
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.utils import register_all_modules

classes = ('background', 'Vegetation')
palette = [[0, 0, 0], [0, 255, 0]]

@DATASETS.register_module()
class CellDataset(BaseSegDataset):
  METAINFO = dict(classes=classes, palette=palette)
  def __init__(self, 
    img_suffix='.png',
    seg_map_suffix='.png',
    reduce_zero_label=False,
    **kwargs):
    super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)