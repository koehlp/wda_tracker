

import detectors.evaluation.object_det_metrics._init_paths

from utilities.helper import drawBoundingBox
from detectors.evaluation.object_det_metrics.lib.BoundingBox import BoundingBox
from detectors.evaluation.object_det_metrics.lib.BoundingBoxes import BoundingBoxes
from detectors.evaluation.object_det_metrics.lib.Evaluator import Evaluator
from detectors.evaluation.object_det_metrics.lib.utils_object_det_metrics import *

from evaluation.multicam_evaluation import load_ground_truth_dataframes
from utilities.helper import constrain_bbox_to_img_dims

import os
from tqdm import tqdm
from collections import defaultdict

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

    def __init__(self,groundtruth_folder,detection_results_folder,output_folder,working_dir,cam_ids,image_dims=(1920,1080)):
        self.groundtruth_folder = groundtruth_folder
        self.detection_results_folder = detection_results_folder
        self.working_dir = working_dir
        self.cam_ids = cam_ids
        self.image_dims = image_dims
        self.output_folder = output_folder

    def load_gt_bboxes(self):


        gt_cam_dataframes = load_ground_truth_dataframes(dataset_folder=self.groundtruth_folder
                                     ,working_dir=self.working_dir
                                    ,cam_ids=self.cam_ids)

        bboxes_cams = []
        for gt_df_cam in gt_cam_dataframes:
            frame_no_to_bboxes = defaultdict(list)
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

                frame_no_to_bboxes[frame_no_cam].append(bbox)
            bboxes_cams.append(frame_no_to_bboxes)
        return bboxes_cams



    def load_detected_bboxes(self):
        detected_cam_dataframes = load_detected_cam_dataframes(detection_folder=self.detection_results_folder
                                     ,cam_ids=self.cam_ids)

        detections_all_cams = []
        for detected_cam_df in detected_cam_dataframes:
            frame_no_to_bboxes = defaultdict(list)
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
                frame_no_to_bboxes[frame_no_cam].append(bbox)

            detections_all_cams.append(frame_no_to_bboxes)

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



    def get_all_bboxes_frame_wise(self):

        def create_bboxes_object(bboxes):
            bboxes_obj = BoundingBoxes()

            for bbox in bboxes:
                bboxes_obj.addBoundingBox(bbox)
            return bboxes_obj


        detected_bboxes_cams = self.load_detected_bboxes()
        gt_bboxes_cams = self.load_gt_bboxes()


        assert len(detected_bboxes_cams) == len(gt_bboxes_cams)

        bboxes_obj_per_cam = []
        for bboxes_gt_cam, det_gt_cam in zip(gt_bboxes_cams,detected_bboxes_cams):
            frame_no_to_bboxes_obj = {}
            for frame_no_cam, gt_bboxes in bboxes_gt_cam.items():
                dt_bboxes = det_gt_cam[frame_no_cam]

                all_bboxes = dt_bboxes + gt_bboxes

                frame_no_to_bboxes_obj[frame_no_cam] = create_bboxes_object(all_bboxes)

            bboxes_obj_per_cam.append(frame_no_to_bboxes_obj)

        return bboxes_obj_per_cam


    def draw_all_bboxes(self,cam_id_frame_no_list):

        def get_all_bboxes_with_type(bbox_type,bboxes):

            return [bbox for bbox in bboxes if bbox.getBBType() == bbox_type ]

        def draw_bboxes(cam_id,bboxes):

            #To group together the gt bboxes at first and then the detected ones to draw the gt bboxes thicker
            gt_bboxes = get_all_bboxes_with_type(bbox_type=BBType.GroundTruth,bboxes=bboxes)

            dt_bboxes = get_all_bboxes_with_type(bbox_type=BBType.Detected, bboxes=bboxes)

            bboxes = gt_bboxes + dt_bboxes

            frame_no_cam = bboxes[0].getImageName()

            image_name = "image_{}_{}.jpg".format(frame_no_cam, cam_id)
            image_path = os.path.join(self.groundtruth_folder, "cam_{}".format(cam_id), image_name)

            img = cv2.imread(image_path)
            for i,bbox_obj in enumerate(bboxes):

                bbox = bbox_obj.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

                bbox = list(map(int,bbox))



                if bbox_obj.getBBType() == BBType.GroundTruth:
                    bbox_color = (0,255,0)



                    drawBoundingBox(image=img, xyxy_bbox=bbox, color=bbox_color,thickness=3)
                    cv2.putText(img=img, text=str(i), org=(bbox[2], bbox[3]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 255, 255), thickness=1)
                else:
                    bbox_color = (0, 0, 255)

                    drawBoundingBox(image=img,xyxy_bbox=bbox,color=bbox_color,thickness=2)

                image_output_path = os.path.join(self.output_folder,image_name)

            print(image_output_path)

            cv2.imwrite(image_output_path,img)


        all_bboxes_cams = self.get_all_bboxes_frame_wise()

        for cam_id, frame_no in cam_id_frame_no_list:

            bboxes_cam = all_bboxes_cams[cam_id]

            bboxes_obj = bboxes_cam[frame_no]

            bboxes = bboxes_obj.getBoundingBoxes()

            draw_bboxes(cam_id,bboxes)



    def evaluate(self):
        framewise_results = pd.DataFrame({ "frame_no_cam" : [] , "cam_id" : [] , "bbox_count" : [],"AP" : []})
        evaluator = Evaluator()
        all_bboxes_cams = self.get_all_bboxes_frame_wise()
        for cam_id, bboxes_cam in enumerate(all_bboxes_cams):

            for frame_no_cam,bboxes_obj in tqdm(bboxes_cam.items()):


                eval_results = evaluator.GetPascalVOCMetrics(boundingboxes=bboxes_obj)

                framewise_results = framewise_results.append({ "frame_no_cam" : frame_no_cam
                                                                 , "cam_id" : cam_id
                                                                 , "bbox_count" : bboxes_obj.count()
                                                                 ,"AP" : eval_results[0]["AP"]
                                                               ,"gt_positives" : eval_results[0]["total positives"]
                                                               ,"TP" : eval_results[0]["total TP"]
                                                               ,"FP" : eval_results[0]["total FP"]}
                                                             , ignore_index=True)


        output_path = os.path.join(self.output_folder,"framwise_eval.csv")
        framewise_results.to_csv(output_path)
        print("Result written to: {}".format(output_path))


if __name__ == "__main__":
    de = Detector_evaluation(groundtruth_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                             ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                             ,detection_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/detector/detections/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
                             , output_folder="/media/philipp/philippkoehl_ssd/work_dirs/evaluation/framewise_detector_evaluation"
                             ,cam_ids=[0,1,2,3,4,5])

    #de.evaluate()

    de.draw_all_bboxes([(1,185951),(1,190308),(5,189565),(2,188103)])
