

import detectors.evaluation.object_det_metrics._init_paths


from detectors.evaluation.object_det_metrics.lib.BoundingBox import BoundingBox
from detectors.evaluation.object_det_metrics.lib.BoundingBoxes import BoundingBoxes
from detectors.evaluation.object_det_metrics.lib.Evaluator import Evaluator
from detectors.evaluation.object_det_metrics.lib.utils_object_det_metrics import *

from evaluation.multicam_evaluation import load_ground_truth_dataframes
from utilities.helper import constrain_bbox_to_img_dims

import os

import pandas as pd

def load_detected_cam_dataframes(detection_folder,cam_ids):

    cam_ids.sort()
    detections_cams = []
    for cam_id in cam_ids:
        detections_cam_filename = "detections_cam_{}.csv".format(cam_id)

        detections_cam_path = os.path.join(detection_folder,detections_cam_filename)
        detections_cam = pd.read_csv(filepath_or_buffer=detections_cam_path)
        detections_cams.append(detections_cam)


    return detections_cams


class Detector_evaluation:

    def __init__(self,groundtruth_folder,detection_results_folder,working_dir,cam_ids,image_dims=(1920,1080)):
        self.groundtruth_folder = groundtruth_folder
        self.detection_results_folder = detection_results_folder
        self.working_dir = working_dir
        self.cam_ids = cam_ids
        self.image_dims = image_dims

    def load_gt_bboxes(self):


        gt_cam_dataframes = load_ground_truth_dataframes(dataset_folder=self.groundtruth_folder
                                     ,working_dir=self.working_dir
                                    ,cam_ids=self.cam_ids)

        bboxes_cams = []
        for gt_df_cam in gt_cam_dataframes:
            bboxes_one_cam = []
            for idx, df_row in gt_df_cam.iterrows():
                frame_no_cam = int(df_row["frame_no_cam"])
                xtl = df_row["x_top_left_BB"]
                ytl = df_row["y_top_left_BB"]
                xbr = df_row["x_bottom_right_BB"]
                ybr = df_row["y_bottom_right_BB"]

                xtl,ytl,xbr,ybr = constrain_bbox_to_img_dims(xyxy_bbox=(xtl,ytl,xbr,ybr))


                bbox = BoundingBox(imageName=str(frame_no_cam), classId='person', x=xtl, y=ytl,
                            w=xbr, h=ybr, typeCoordinates=CoordinatesType.Absolute,
                            bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, imgSize=self.image_dims)

                bboxes_one_cam.append(bbox)
            bboxes_cams.append(bboxes_one_cam)
        return bboxes_cams



    def load_detected_bboxes(self):
        detected_cam_dataframes = load_detected_cam_dataframes(detection_folder=self.detection_results_folder
                                     ,cam_ids=self.cam_ids)

        detections_all_cams = []
        for detected_cam_df in detected_cam_dataframes:
            detections_one_cam = []
            for idx,detection_row in detected_cam_df.iterrows():
                x = detection_row["x"]
                y = detection_row["y"]
                w = detection_row["w"]
                h = detection_row["h"]
                score = detection_row["score"]
                frame_no_cam = int(detection_row["frame_no_cam"])

                bbox = BoundingBox(imageName=str(frame_no_cam), classId='person', x=x, y=y,
                                   w=w, h=h, typeCoordinates=CoordinatesType.Absolute,
                                   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=self.image_dims
                                   ,classConfidence=score)
                detections_one_cam.append(bbox)

            detections_all_cams.append(detections_one_cam)

        return detections_all_cams


    def bbox_list_to_bboxes_obj(self,bboxes_list_cams,bboxes_obj_cams:BoundingBoxes=None):

        #If bboxes_obj_cams has not been initialized. This has to be done at first.
        if bboxes_obj_cams is None:
            bboxes_obj_cams = []

            for _ in bboxes_list_cams:
                bboxes_obj = BoundingBoxes()

                bboxes_obj_cams.append(bboxes_obj)

        assert len(bboxes_list_cams) == len(bboxes_obj_cams)


        for bboxes_list, bboxes_obj in zip(bboxes_list_cams,bboxes_obj_cams):

            for bbox in bboxes_list:
                bboxes_obj.addBoundingBox(bbox)

        return bboxes_obj_cams


    def get_all_gt_and_dt_bboxes(self):
        detected_bboxes_cams = self.load_detected_bboxes()
        gt_bboxes_cams = self.load_gt_bboxes()

        bboxes_obj_cams = self.bbox_list_to_bboxes_obj(bboxes_list_cams=detected_bboxes_cams
                                                       ,bboxes_obj_cams=None)

        bboxes_obj_cams = self.bbox_list_to_bboxes_obj(bboxes_list_cams=gt_bboxes_cams
                                                             ,bboxes_obj_cams=bboxes_obj_cams)

        return bboxes_obj_cams

    def evaluate(self):

        evaluator = Evaluator()
        all_bboxes_cams = self.get_all_gt_and_dt_bboxes()
        gt_bboxes = []
        for cam_id, bboxes_cam in enumerate(all_bboxes_cams):

            for bb in bboxes_cam.getBoundingBoxes():
                if bb.getBBType() == BBType.Detected:
                    gt_bboxes.append(bb)

            eval_results = evaluator.GetPascalVOCMetrics(boundingboxes=bboxes_cam)
            print("cam_id : {}".format(cam_id))
            print("AP: {}".format(eval_results[0]["AP"]))




if __name__ == "__main__":
    de = Detector_evaluation(groundtruth_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                             ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                             ,detection_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/detector/detections/faster_rcnn_r50_coco_trained_strong_reid_GtaExtShort_test"
                             ,cam_ids=list(range(6)))

    de.evaluate()