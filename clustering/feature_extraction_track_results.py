from feature_extractors.reid_strong_extractor import Reid_strong_extractor
from feature_extractors.abd_net_extractor import Abd_net_extractor

import numpy as np
import cv2
import os

class Feature_extraction:

    def __init__(self,cfg):



        if cfg.feature_extractor_name == "reid_strong_extractor":
            self.feature_extractor = Reid_strong_extractor(cfg)

        if cfg.feature_extractor_name == "abd_net_extractor":
            self.feature_extractor = Abd_net_extractor(cfg.abd_net_extractor)


    def get_features(self, bboxes_xyxy, ori_img, debug=False):
        im_crops = []
        for bbox in bboxes_xyxy:

            bbox = map(int,bbox)
            xtl, ytl, xbr, ybr = bbox
            im = ori_img[ytl:ybr, xtl:xbr]

            if debug:
                cv2.imshow('image', im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            im_crops.append(im)
        if im_crops:
            features = self.feature_extractor.extract(im_crops)
        else:
            features = np.array([])
        return features