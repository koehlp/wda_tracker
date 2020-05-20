import argparse
import mmcv
import importlib
from trackers.deep_sort import DeepSort
from util import draw_bboxes
from tqdm import tqdm
from utilities.helper import xtylwh_to_xyxy
import os
import warnings
import cv2
import logging
import json
from utilities.track_result_statistics import count_tracks

from utilities.non_daemonic_pool import NonDeamonicPool

from trackers.utilities import *

import pandas as pd

from feature_extractors.reid_strong_baseline.utils.logger import setup_logger

from utilities.helper import TqdmToLogger




class Run_tracker:
    def __init__(self,args):


        self.cfg = mmcv.Config.fromfile(args.config).root

        self.cfg.general.config_basename = os.path.basename(args.config).replace(".py","")

        #mmdetection does not put everything to the device that is being set in its function calls
        #E.g. With torch.cuda.set_device(4) it will run without errors. But still using GPU 0
        #With os.environ['CUDA_VISIBLE_DEVICES'] = '4' the visibility will be restricted to only the named GPUS starting internatlly from zero
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg.general.cuda_visible_devices

        append_to_pythonpath(self.cfg.general.source_root_paths,__file__)

        print(sys.path)

        #Loads the detector module
        #E.g. detectors.faster_rcnn_resnet_50
        detector_module = importlib.import_module(self.cfg.detector.module_name)

        #Initializes the detector class by calling the constructor and creating the object
        self.detector = getattr(detector_module,self.cfg.detector.class_name)(self.cfg)

        # Loads the dataset module
        self.dataset_module = importlib.import_module(self.cfg.data.module_name)

        # Initializes the dataset class by calling a function
        self.cam_image_iterators = self.dataset_module.get_cam_iterators(self.cfg
                                                                                  , self.cfg.data.source.base_folder
                                                                                     , self.cfg.data.source.cam_ids)
        self.set_tracker_config_run_path()

        self.deep_sort = DeepSort(self.cfg)



        #Set up the logger
        logger = setup_logger("wda_tracker", self.config_run_path, 0)

        logger.info(args)
        logger.info(json.dumps(self.cfg,sort_keys=True, indent=4))

    def load_detections(self,cam_id):

        detections_path_folder = os.path.join(self.cfg.detector.detections_path
                                              , self.cfg.general.config_basename)
        os.makedirs(detections_path_folder, exist_ok=True)

        self.detections_path = os.path.join(detections_path_folder, "detections_cam_{}.csv".format(cam_id))
        self.detections_loaded = pd.DataFrame([])
        self.detections_to_store = pd.DataFrame([])
        if os.path.exists(self.detections_path):
            self.detections_loaded = pd.read_csv(self.detections_path)
        else:
            self.detections_to_store = pd.DataFrame(
                {"frame_no_cam": [], "id": [], "x": [], "y": [], "w": [], "h": [], "score": []})


    def set_tracker_config_run_path(self):
        #self.cfg.config_run_path
        # Build the path where results and logging files etc. should be stored
        self.config_run_path = os.path.join(self.cfg.general.work_dirs, "tracker", "config_runs",self.cfg.general.config_basename)
        os.makedirs(self.config_run_path, exist_ok=True)


    def save_detections(self):
        if len(self.detections_to_store) > 0:
            self.detections_to_store = self.detections_to_store.astype({"frame_no_cam": int
                                                                           , "id": int
                                                                           , "x": int
                                                                           , "y": int
                                                                           , "w": int
                                                                           , "h": int
                                                                           , "score": float})
            self.detections_to_store.to_csv(self.detections_path, index=False)

    def store_detections_one_frame(self, frame_no_cam, xywh_bboxes, scores):
        '''
        "frame_no_cam,id,x,y,w,h,score"
        :param xywh_bboxes:
        :return:
        '''


        for index, xywh_bbox_score in enumerate(zip(xywh_bboxes,scores)):
            xywh_bbox, score = xywh_bbox_score

            self.detections_to_store = self.detections_to_store.append({ "frame_no_cam" : frame_no_cam
                                        , "id" : index
                                        , "x" : xywh_bbox[0]
                                        , "y" : xywh_bbox[1]
                                        , "w" : xywh_bbox[2]
                                        , "h" : xywh_bbox[3]
                                        , "score" : score },ignore_index=True)



    def img_callback(self,dataset_img):


        if len(self.detections_loaded) > 0:
            detections_current_frame =  self.detections_loaded[self.detections_loaded["frame_no_cam"] == dataset_img.frame_no_cam]
            scores = detections_current_frame["score"].tolist()
            bboxes_xtlytlwh = list(zip(detections_current_frame["x"], detections_current_frame["y"],detections_current_frame["w"],detections_current_frame["h"]))
        else:
            bboxes_xtlytlwh, scores = self.detector.detect(dataset_img.img)
            self.store_detections_one_frame(dataset_img.frame_no_cam, bboxes_xtlytlwh, scores)


        draw_img = dataset_img.img
        if bboxes_xtlytlwh is not None:

            outputs = self.deep_sort.update(bboxes_xtlytlwh, scores, dataset_img)

            if len(outputs) > 0:

                bboxes_xtylwh = outputs[:, :4]
                bboxes_xyxy = [ xtylwh_to_xyxy(bbox_xtylwh,dataset_img.img_dims) for bbox_xtylwh in bboxes_xtylwh ]
                identities = outputs[:, -2]
                detection_idxs = outputs[:, -1]
                draw_img = draw_bboxes(dataset_img.img, bboxes_xyxy, identities)


                for detection_idx, person_id, bbox in zip(detection_idxs,identities,bboxes_xyxy):
                    print('%d,%d,%d,%d,%d,%d,%d,%d' % (
                            dataset_img.frame_no_cam, dataset_img.cam_id, person_id, detection_idx,int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), file=self.track_results_file)

        if self.cfg.general.display_viewer:
            cv2.imshow("Annotation Viewer", draw_img)
            cv2.waitKey(1)


    def run_on_cam_images(self,cam_iterator):
        for image in cam_iterator:
            self.pbar_tracker.update()
            self.img_callback(image)



    def run_on_dataset(self):
        logger = logging.getLogger("deep_sort_mc")
        logger.info("Starting tracking on dataset.")

        for cam_iterator in self.cam_image_iterators:
            logger.info("Processing cam {}".format(cam_iterator.cam_id))

            self.track_results_path = os.path.join(self.config_run_path, "track_results_{}.txt".format(cam_iterator.cam_id))
            logger.info(self.track_results_path)
            self.track_results_file = open(self.track_results_path, 'w')
            print("frame_no_cam,cam_id,person_id,detection_idx,xtl,ytl,xbr,ybr", file=self.track_results_file)

            self.load_detections(cam_iterator.cam_id)

            self.run_on_cam_images(cam_iterator)


            self.track_results_file.close()

            self.save_detections()

            track_count_string = count_tracks(self.track_results_path)
            logger.info(track_count_string)



    def run(self):

        self.run_on_dataset()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    return parser.parse_args()


if __name__=="__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    args = parse_args()

    run_tracker = Run_tracker(args)

    run_tracker.run()