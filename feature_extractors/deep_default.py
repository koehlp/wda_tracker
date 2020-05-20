from feature_extractors.deep.feature_extractor import Extractor
'''
The originally in deep_sort_pytorch contained feature extractor.
'''
class Deep_default:

    def __init__(self,cfg):
        self.extractor = Extractor(cfg.feature_extractor.checkpoint_file, cfg)

    def extract(self,img_crops):
        return self.extractor(img_crops)