import os
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.utils import register_all_modules

# classes = ('background', 'stomatal complex', 'stoma', 'outer ledge', 'pore', 'pavement cell', 'guard cell')
# palette = [[0, 0, 0], [170, 0, 0], [231, 116, 237], [255, 245, 54], [234, 234, 235], [255, 170, 0], [85, 85, 255]]

# classes = ('background', 'stomatal complex', 'stoma')
# palette = [[0, 0, 0], [170, 0, 0], [231, 116, 237]]

classes = ('background', 'stomatal complex')
palette = [[0, 0, 0], [170, 0, 0]]


@DATASETS.register_module()
class StomataDataset(BaseSegDataset):
    """Stomata dataset"""
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, 
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
