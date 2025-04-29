from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CFADataset(BaseSegDataset):
    # Classes and corresponding RGB color
    METAINFO = {
        'classes':['background', 'red', 'green', 'white', 'seed-black', 'seed-white'],
        'palette':[[127,127,127], [200,0,0], [0,200,0], [144,238,144], [30,30,30], [251,189,8]]
    }
    
    # Specify image extension and annotation extension
    def __init__(self,
                 seg_map_suffix='.png',   # Format of the annotation mask image
                 reduce_zero_label=False, # Whether to remove the class with ID 0
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)