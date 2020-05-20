

import numpy as np
from utilities.pandas_loader import load_csv
from utilities.helper import adjustCoordsTypes
from collections import defaultdict
from utilities.helper import constrain_bbox_to_img_dims
from tqdm import tqdm
import pickle
import os
from collections import Counter
import pandas as pd
import networkx as nx
import math
import sys
import cv2
from scipy import spatial
from evaluation.motmetrics_evaluation import Motmetrics_evaluation, Motmetrics_distance

from utilities.helper import *
from utilities.dataset_statistics import *
from clustering.clustering_utils import *

from shapely.geometry import Point
import copy
import time
import logging
import json
from clustering.velocity_calculation import Velocity_calculation



class Single_cam_clustering:

    def __init__(self,work_dirs,train_track_results_folder,test_track_results_folder,train_dataset_folder,test_dataset_folder
                 ,cam_count,person_identifier):


        self.train_track_results_folder = train_track_results_folder
        self.test_track_results_folder = test_track_results_folder
        self.work_dirs = work_dirs
        self.train_dataset_folder = train_dataset_folder
        self.test_dataset_folder = test_dataset_folder
        self.cam_count = cam_count
        self.person_identifier = person_identifier

        self.logger = logging.getLogger("clustering_logger")


    def get_predicted_position(self,track1,track2,n_tail=40):


        #We need track1[-1] < track2[0]
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        last_n_positons = track1[-n_tail:]


        first_pos_track1 = get_bbox_middle_pos(last_n_positons[0]["bbox"])
        second_pos_track1 = get_bbox_middle_pos(last_n_positons[-1]["bbox"])
        first_pos_frame_no_track1 = last_n_positons[0]["frame_no_cam"]
        second_pos_frame_no_track1 = last_n_positons[-1]["frame_no_cam"]

        if len(track1) == 1:
            return np.array(first_pos_track1)

        velocity = (np.array(second_pos_track1) - np.array(first_pos_track1)) / (second_pos_frame_no_track1 - first_pos_frame_no_track1)


        start_pos_frame_no_track2 = track2[0]["frame_no_cam"]

        passed_time = start_pos_frame_no_track2 - second_pos_frame_no_track1

        predicted_position = np.array(second_pos_track1) + (velocity * passed_time)


        return predicted_position


    def get_track_pred_pos_start_distance(self, track1, track2):

        # We need track1[-1] < track2[0]
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        pred_position = self.get_predicted_position(track1, track2)

        track2_start = get_bbox_middle_pos(track2[0]["bbox"])
        bbox_height = get_bbox_height(track1[-1]["bbox"])
        pred_pos_start_distance = np.linalg.norm(np.array(pred_position)-np.array(track2_start)) / bbox_height

        if pred_pos_start_distance > self.maximum_link_pred_distance:
            return np.Inf
        else:
            return pred_pos_start_distance / self.maximum_link_pred_distance

    def are_tracks_disjunct(self,track1,track2):

        track1_frame_nos = [track_pos["frame_no_cam"] for track_pos in track1]
        track2_frame_nos = [track_pos["frame_no_cam"] for track_pos in track2]
        return set(track1_frame_nos).isdisjoint(track2_frame_nos)

    def combine_all_cams(self,cam_ids,dataset_type):
        '''
        Will combine tracks for multiple cameras.
        :return:
        '''

        result = []
        for cam_id in cam_ids:
            combined_one_cam = self.cluster_via_hierarchical(cam_id=cam_id)
            result.append(combined_one_cam)

        return result

    def combine_tracks(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        return track1 + track2

    def calc_dist_clustered_single(self, track1_indices, track2_indices):
        min_dist = sys.maxsize
        for track1_idx in track1_indices:
            for track2_idx in track2_indices:
                pair_dist = self.track_pair_to_dist[frozenset([track1_idx, track2_idx])]

                min_dist = min(min_dist, pair_dist)
                # if there is a hard constraint it should not be clustered
                if pair_dist > 8000:
                    return 10000

        return min_dist

    def calculate_dist_clustered_tracks(self,track1_indices, track2_indices, all_tracks):


        track1 = map_idx_to_tracks(track1_indices,all_tracks)
        track2 = map_idx_to_tracks(track2_indices,all_tracks)

        track1 = flatten_list(track1)
        track2 = flatten_list(track2)

        for track1_idx in track1_indices:
            for track2_idx in track2_indices:
                pair_dist = self.track_pair_to_dist[frozenset([track1_idx, track2_idx])]

                # if there is a hard constraint it should not be clustered
                if pair_dist == np.Inf:
                    return np.Inf

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1_indices
            track1_indices = track2_indices
            track2_indices = track_tmp

        pair_dist = self.track_pair_to_dist[frozenset([track1_indices[-1], track2_indices[0]])]

        return pair_dist

    def get_track_cosine_distance(self,track1,track2,n_tail=40):
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp


        track1_tail = track1[-n_tail:]
        track2_head = track2[:n_tail]

        track1_direction = np.array(get_bbox_middle_pos(track1_tail[-1]["bbox"])) - np.array(get_bbox_middle_pos(track1_tail[0]["bbox"]))

        track2_direction = np.array(get_bbox_middle_pos(track2_head[-1]["bbox"])) - np.array(get_bbox_middle_pos(track2_head[0]["bbox"]))

        if np.count_nonzero(track1_direction) == 0 or np.count_nonzero(track2_direction) == 0:
            return 0.0

        result = spatial.distance.cosine(track1_direction,track2_direction)

        return result


    def close_track_gaps_interpolation(self, tracks):
        result_tracks = []

        def bbox_interpolation(bbox_start, bbox_end, frame_no_start, frame_no_end, frame_no_interpolate):

            direction_vector = bbox_end - bbox_start
            scalar_interpolator = (frame_no_interpolate - frame_no_start) / (frame_no_end - frame_no_start)
            interpolated_box = bbox_start + scalar_interpolator * direction_vector

            return interpolated_box

        for track in tracks:

            filled_gap_track = []

            if len(track) == 1:
                result_tracks.append(track)
                continue

            for track_pos_idx in range(len(track) - 1):
                lower_frame_no = track[track_pos_idx]["frame_no_cam"]
                upper_frame_no = track[track_pos_idx + 1]["frame_no_cam"]
                bbox_start = np.array(track[track_pos_idx]["bbox"])
                bbox_end = np.array(track[track_pos_idx + 1]["bbox"])

                frame_diff = upper_frame_no - lower_frame_no
                filled_gap_track.append(track[track_pos_idx])
                if frame_diff > 1 and frame_diff < 50:

                    for insert_frame_no in range(lower_frame_no + 1, upper_frame_no):
                        predicted_bbox = bbox_interpolation(bbox_start,bbox_end,lower_frame_no,upper_frame_no,insert_frame_no)
                        filled_gap_track.append({"bbox": tuple(predicted_bbox)
                                                    , "frame_no_cam": insert_frame_no
                                                    , "filled_bbox": True
                                                    , "track_no": track[track_pos_idx]["track_no"]})

            #At the end of the track the last element has to be inserted because of range(len(track) - 1)
            filled_gap_track.append(track[-1])


            result_tracks.append(filled_gap_track)
        return result_tracks


    def get_overlap_cam_ids_to_frame_no_cam_to_track_pos(self,track_results_folder):

        def get_cam_id_to_frame_no_to_track_pos_path():

            folder = os.path.join(self.work_dirs
                                                                 ,"clustering"
                                                                 ,"overlap_cam_ids_to_frame_no_to_track_pos"
                                                                 ,os.path.basename(track_results_folder))

            os.makedirs(folder,exist_ok=True)

            pickle_path = os.path.join(folder
                                      ,"cam_ids_to_frame_no_to_track_pos.pkl")

            return pickle_path

        pickle_path = get_cam_id_to_frame_no_to_track_pos_path()

        if os.path.exists(pickle_path):
            print("Found cam_ids_to_frame_no_cam_to_track_pos")
            print(pickle_path)
            with open(pickle_path, "rb") as pickle_file:
                return pickle.load(pickle_file)

        cam_ids_to_frame_no_cam_to_track_pos = defaultdict(defaultdict_defaultdict_list)
        print("Calculating overlapping cam_ids_to_frame_no_cam_to_track_pos")
        for outer_cam_id in range(self.cam_count):
            track_results_path = os.path.join(track_results_folder, "track_results_{}.txt".format(outer_cam_id))
            track_results = pd.read_csv(track_results_path)
            for inner_cam_id in tqdm(range(self.cam_count),total=self.cam_count):

                for idx,result_row in track_results.iterrows():
                    frame_no_cam = result_row["frame_no_cam"]
                    person_id = result_row["person_id"]
                    bbox = get_bbox_of_row_track_results(result_row)
                    person_pos = get_bbox_middle_pos(bbox)
                    #Check if the current point can be seen in the clustering cam
                    #This is a try to get a higher performance

                    point_visible_in_clustering_cam = self.cam_id_to_cam_id_to_polygon[outer_cam_id][inner_cam_id].contains(Point(person_pos))

                    if point_visible_in_clustering_cam:
                        track_pos = { "person_id" : person_id, "frame_no_cam" : frame_no_cam, "bbox" : bbox }
                        cam_ids_to_frame_no_cam_to_track_pos[outer_cam_id][inner_cam_id][frame_no_cam].append(track_pos)

        with open(pickle_path, "wb") as pickle_file:
            print("Writing cam_ids_to_frame_no_cam_to_track_pos")
            print(pickle_path)
            pickle.dump(cam_ids_to_frame_no_cam_to_track_pos, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        return cam_ids_to_frame_no_cam_to_track_pos

    def initialize_maximum_link_predict_distance(self,track_results_folder,maximum_link_frames=500):

        vc = Velocity_calculation(track_results_folder,list(range(self.cam_count)),self.work_dirs)


        velocity_stats = vc.get_velocity_stats(n_tail=40,step_width=10)
        velocity_mean = velocity_stats["velocity_mean"]

        self.maximum_link_pred_distance = velocity_mean * maximum_link_frames


    def initialize_homography_match_score(self,track_results_folder):

        cam_homographies_pkl_path = get_cam_homographies_path(dataset_folder=self.train_dataset_folder,work_dirs=self.work_dirs)
        self.cam_homographies = get_cam_homographies(self.train_dataset_folder
                                                     , self.work_dirs
                                                     , cam_count=self.cam_count
                                                     , person_identifier=self.person_identifier
                                                     , pickle_path=cam_homographies_pkl_path
                                                     )

        overlapping_area_hulls_path = get_overlapping_area_hulls_path(dataset_folder=self.train_dataset_folder,work_dirs=self.work_dirs)
        cam_id_to_cam_id_to_hull = get_overlapping_areas(self.train_dataset_folder
                                                         , self.work_dirs
                                                         , cam_count=self.cam_count
                                                         , person_identifier=self.person_identifier
                                                         , pickle_path=overlapping_area_hulls_path)

        self.cam_id_to_cam_id_to_polygon = hull_to_polygon(cam_id_to_cam_id_to_hull)

        '''
        Because the calculation of the get_homography_match_score is very time consuming. The matches of one track 
        in the current cam that will be clustered, will be cached in this variable.
        
        '''
        self.track_match_cache = {}

        self.cam_ids_to_frame_no_cam_to_track_results = self.get_overlap_cam_ids_to_frame_no_cam_to_track_pos(track_results_folder)


    def get_homography_match_score(self,track1,track1_idx,track2,track2_idx,match_threshold=10):


        def count_transformed_point_matches(track_results_one_frame,transformed_point,origin_point_bbox,person_id_to_matches):

            for track_pos in track_results_one_frame:

                bbox = track_pos["bbox"]

                #Checking if the bbox of the origin point is larger. That means the quality of the track might be better of the larger one.
                #If the quality of the original track is better it makes no sense to take the information of the other one into account
                if get_bbox_height(bbox) < get_bbox_height(origin_point_bbox):
                    return

                target_position = get_bbox_middle_pos(bbox)
                transformed_to_target_distance = np.linalg.norm(np.array(target_position) - transformed_point)

                if transformed_to_target_distance < match_threshold:
                    person_id = track_pos["person_id"]
                    person_id_to_matches[person_id] += 1




        def get_track_matches(track,track_idx):
            if track_idx in self.track_match_cache:
                return self.track_match_cache[track_idx]

            transformed_cam_id_to_person_id_to_match_count = defaultdict(lambda: defaultdict(lambda:0))

            transformed_cam_id_to_track_length = defaultdict(lambda:0)

            for track_pos in track:
                track_pos_bbox = track_pos["bbox"]
                track_point = get_bbox_middle_pos(track_pos_bbox)
                frame_no_cam = track_pos["frame_no_cam"]

                for transformed_cam_id in range(self.cam_count):
                    if self.cam_id_to_cam_id_to_polygon[self.clustering_cam_id][transformed_cam_id].contains(Point(track_point)):

                        homography = self.cam_homographies[self.clustering_cam_id][transformed_cam_id]

                        homogen_point = np.append(np.array(track_point), [1])
                        transformed_point = np.matmul(homography, homogen_point)

                        # Homogeneous coords back to cartesian
                        transformed_point = np.true_divide(transformed_point[:2], transformed_point[-1])

                        person_to_match_count = transformed_cam_id_to_person_id_to_match_count[transformed_cam_id]

                        track_results_one_cam = self.cam_ids_to_frame_no_cam_to_track_results[transformed_cam_id][self.clustering_cam_id]
                        track_results_one_frame = track_results_one_cam[frame_no_cam]

                        count_transformed_point_matches(track_results_one_frame,transformed_point,track_pos_bbox,person_to_match_count)

                        transformed_cam_id_to_track_length[transformed_cam_id] += 1

            self.track_match_cache[track_idx] = transformed_cam_id_to_person_id_to_match_count

            return transformed_cam_id_to_person_id_to_match_count


        def find_maximum_match_person_id(transformed_cam_id_to_person_id_to_match_count):
            max_count_and_person_id_and_cam_id = (0,-1,-1)
            for cam_id, person_id_to_match_count in transformed_cam_id_to_person_id_to_match_count.items():
                for person_id, match_count in person_id_to_match_count.items():

                    max_count_and_person_id_and_cam_id = max(max_count_and_person_id_and_cam_id,(match_count,person_id,cam_id),key=lambda x:x[0])

            return max_count_and_person_id_and_cam_id


        transformed_cam_id_to_person_id_to_match_count_track1 = get_track_matches(track1,tuple(track1_idx))
        transformed_cam_id_to_person_id_to_match_count_track2 = get_track_matches(track2,tuple(track2_idx))

        track1_max_match_count_and_person_id_and_cam_id = find_maximum_match_person_id(transformed_cam_id_to_person_id_to_match_count_track1)
        track2_max_match_count_and_person_id_and_cam_id = find_maximum_match_person_id(transformed_cam_id_to_person_id_to_match_count_track2)

        #if the maximum is the same for both tracks regarding to one person in one cam
        if track1_max_match_count_and_person_id_and_cam_id[1] == track2_max_match_count_and_person_id_and_cam_id[1] and \
                track1_max_match_count_and_person_id_and_cam_id != (0,-1,-1) and \
                track2_max_match_count_and_person_id_and_cam_id != (0,-1,-1):

            tracks_match_sum = track1_max_match_count_and_person_id_and_cam_id[0] + track2_max_match_count_and_person_id_and_cam_id[0]

            return -1

        return 0





    def calculate_track_distance(self,candidate_track_idx,partner_track_idx,dataset, not_disjunct_distance=10000):

        candidate_track = map_idx_to_tracks(candidate_track_idx, dataset)

        partner_track = map_idx_to_tracks(partner_track_idx, dataset)

        candidate_track = flatten_list(candidate_track)
        partner_track = flatten_list(partner_track)

        if not self.are_tracks_disjunct(candidate_track, partner_track):
            return not_disjunct_distance

        pred_pos_start_dist = self.get_track_pred_pos_start_distance(candidate_track, partner_track)

        cosine_distance_tracks = self.get_track_cosine_distance(candidate_track, partner_track)
        pred_pos_start_dist += cosine_distance_tracks

        #pred_pos_start_dist += self.get_homography_match_score(candidate_track,candidate_track_idx,partner_track,partner_track_idx)

        return pred_pos_start_dist





    def calculate_track_distances(self,candidate_track_idx,partner_track_idx,dataset):

        result_distances = {}
        candidate_track = map_idx_to_tracks(candidate_track_idx, dataset)

        partner_track = map_idx_to_tracks(partner_track_idx, dataset)

        candidate_track = flatten_list(candidate_track)
        partner_track = flatten_list(partner_track)

        if not self.are_tracks_disjunct(candidate_track, partner_track):
            result_distances["are_tracks_disjunct"] = np.Inf
            result_distances["track_pred_pos_start_distance"] = 0
            result_distances["track_cosine_distance"] = 0
            result_distances["homography_match_score"] = 0

            return result_distances
        result_distances["are_tracks_disjunct"] = 0
        result_distances["track_pred_pos_start_distance"] = self.get_track_pred_pos_start_distance(candidate_track, partner_track)
        result_distances["track_cosine_distance"] = self.get_track_cosine_distance(candidate_track, partner_track)
        result_distances["homography_match_score"] = self.get_homography_match_score(candidate_track,candidate_track_idx,partner_track,partner_track_idx)

        return result_distances

    def get_output_track_results_path(self,track_results_folder,cam_id):
        track_results_base_name =os.path.basename(track_results_folder)

        output_track_results_folder = os.path.join(self.work_dirs, "clustering", "single_camera_refinement", track_results_base_name)
        os.makedirs(output_track_results_folder, exist_ok=True)
        output_track_results_path = os.path.join(output_track_results_folder, "track_results_{}.txt".format(cam_id))

        return output_track_results_path



    def cluster_via_hierarchical(self,cam_id
                                 ,dataset_type
                                 ,dist_name_to_distance_weights
                                 ,person_count=50
                                 ,threshold=1000.0):

        self.clustering_cam_id = cam_id

        track_results_folder = None
        if dataset_type == "train":
            track_results_folder = self.train_track_results_folder
        elif dataset_type == "test":
            track_results_folder = self.test_track_results_folder

        self.initialize_maximum_link_predict_distance(track_results_folder)
        self.initialize_homography_match_score(track_results_folder)
        self.tracks_all_persons = get_person_id_to_track(os.path.join(track_results_folder
                                                                 , "track_results_{}.txt".format(cam_id))).values()



        self.tracks_all_persons = list(self.tracks_all_persons)
        #self.tracks_all_persons = self.tracks_all_persons[:100]

        current_clusters = set([(track_idx,) for track_idx, _ in enumerate(self.tracks_all_persons)])

        old_clusters = []


        distances_and_indices = get_distances_and_indices(self.tracks_all_persons,self.calculate_track_distances)
        data_pairwise_dist = compute_pairwise_distance_normalized(distances_and_indices
                                                            ,dist_name_to_distance_weights=dist_name_to_distance_weights)

        self.track_pair_to_dist = get_track_pair_to_dist(data_pairwise_dist)

        heap = build_priority_queue(data_pairwise_dist)

        pbar = tqdm(total=(len(data_pairwise_dist)))

        print("Clustering tracks")
        while len(current_clusters) > person_count:
            pbar.update()

            dist, min_item = heapq.heappop(heap)
            if dist > threshold:
                break

            pair_data = min_item[1]

            if not valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = sum(pair_data, [])

            for pair_item in pair_data:
                old_clusters.append(pair_item)
                current_clusters.remove(tuple(pair_item))

            self.add_heap_entry(heap, new_cluster, current_clusters)

            current_clusters.add(tuple(new_cluster))

        print("Number of tracks: {}".format(len(current_clusters)))
        print_sorted_distances(heap)
        cluster_tracks = map_clusters_track_indices_to_tracks(current_clusters, self.tracks_all_persons)
        cluster_tracks = add_up_lists(cluster_tracks)

        output_track_results_path = self.get_output_track_results_path(track_results_folder,cam_id)
        output_track_results_path = save_combined_tracks(cluster_tracks,cam_id
                                                  ,output_track_results_path)

        return { "track_results_path" : output_track_results_path, "track_count" : len(current_clusters) }



    def add_heap_entry(self, heap, new_cluster, current_clusters):
        for ex_cluster in current_clusters:
            new_heap_entry = []
            dist = self.calculate_dist_clustered_tracks(ex_cluster,new_cluster,self.tracks_all_persons)
            new_heap_entry.append(dist)
            new_heap_entry.append([list(ex_cluster), list(new_cluster)])
            heapq.heappush(heap, (dist, new_heap_entry))



    def get_cam_thresholds(self,single_cam_multiple_tresh_results_path,cam_ids=range(6)
                           ,start_thresh=0.5,end_thresh=4.0,step_count=5):
        evaluation_results = pd.DataFrame()

        def calculate_thresh_one_cam(cam_id,start_thresh,end_thresh,step_count):
            steps = np.linspace(start=start_thresh,stop=end_thresh,num=step_count)
            evaluation_results = pd.DataFrame()

            for thresh in steps:
                #The person count is being set to one because the threshold should be the stop criterion
                clustering_results = self.cluster_via_hierarchical(person_count=1,cam_id=cam_id,threshold=thresh,dataset_type="train")

                result = Motmetrics_evaluation(
                            ground_truth_path=os.path.join(self.train_dataset_folder, "cam_{}".format(cam_id),
                                                           "coords_cam_{}.csv".format(cam_id))
                            , track_results_path=clustering_results["track_results_path"]
                            , working_dir=self.work_dirs).evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)

                summary = result["summary"]
                summary["threshold"] = thresh
                summary["cam_id"] = cam_id
                summary["track_count"] = clustering_results["track_count"]
                evaluation_results = evaluation_results.append(summary,ignore_index=True)

            return evaluation_results

        for cam_id in cam_ids:
            evaluation_results_one_cam = calculate_thresh_one_cam(cam_id,start_thresh,end_thresh,step_count)
            evaluation_results = evaluation_results.append(evaluation_results_one_cam,ignore_index=True)

        evaluation_results.to_csv(single_cam_multiple_tresh_results_path)


    def find_best_cam_settings(self,single_cam_multiple_tresh_results_path,best_single_cam_settings_path,metric="idf1"):
        best_settings = pd.DataFrame()
        single_cam_multiple_tresh_results = pd.read_csv(single_cam_multiple_tresh_results_path)

        grouped_cam_ids = single_cam_multiple_tresh_results.groupby("cam_id",as_index=False).count()
        cam_ids = grouped_cam_ids["cam_id"]

        for cam_id in cam_ids:
            results_one_cam = single_cam_multiple_tresh_results[single_cam_multiple_tresh_results["cam_id"] == cam_id]

            best_metric_row = results_one_cam.loc[results_one_cam[metric].idxmax()]

            metric_score = best_metric_row.iloc[0][metric]
            track_count = int(best_metric_row.iloc[0]["track_count"])
            threshold = best_metric_row.iloc[0]["threshold"]

            best_settings = best_settings.append({ "cam_id" : cam_id , "track_count" : track_count ,
                                                    "threshold" : threshold , metric : metric_score },ignore_index=True)


        best_settings = best_settings.astype({ "cam_id" : int,
                                               "track_count" : int,
                                               "threshold" : float,
                                               metric : float})
        best_settings.to_csv(best_single_cam_settings_path)

    def get_best_single_cam_settings_path(self):

        folder_path = os.path.join(self.work_dirs , "clustering"
                                                     , "best_single_cam_settings"
                                                     , os.path.basename(self.train_track_results_folder))


        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, "single_cam_best_settings.csv")

        return csv_path

    def get_single_cam_multiple_tresh_results_path(self):

        folder_path = os.path.join(self.work_dirs , "clustering"
                                                     , "single_cam_multiple_tresh_results"
                                                     , os.path.basename(self.train_track_results_folder))


        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, "single_cam_multiple_tresh_results.csv")

        return csv_path


    def create_best_cam_settings(self,cam_ids=range(6),start_thresh=0.5,end_thresh=4.0,step_count=5):

        best_single_cam_settings_path = self.get_best_single_cam_settings_path()
        single_cam_multiple_tresh_results_path = self.get_single_cam_multiple_tresh_results_path()
        self.get_cam_thresholds(single_cam_multiple_tresh_results_path
                               ,cam_ids=cam_ids,start_thresh=start_thresh,end_thresh=end_thresh,step_count=step_count)

        self.find_best_cam_settings(single_cam_multiple_tresh_results_path,best_single_cam_settings_path)


    def cluster_from_cam_settings(self,use_threshold):

        best_single_cam_settings_path = self.get_best_single_cam_settings_path()
        print("Clustering from best_single_cam_settings_path.")
        print(best_single_cam_settings_path)
        cam_settings = pd.read_csv(best_single_cam_settings_path)

        for idx,cam_setting in cam_settings.iterrows():
            cam_id = int(cam_setting["cam_id"])
            threshold = cam_setting["threshold"]
            track_count = int(cam_setting["track_count"])

            #If use_threshold is true then the person_count has to be set to one cluster
            #and if the threshold has been reached the clustering stops
            #
            if use_threshold:
                self.cluster_via_hierarchical(cam_id,person_count=1,threshold=threshold,dataset_type="test")
            else:
                self.cluster_via_hierarchical(cam_id, person_count=track_count,threshold=1000,dataset_type="test")


    def cluster_from_weights(self,best_weights_path,default_weights,cam_ids,dataset_type):

        if dataset_type == "test":
            dataset_folder = self.test_dataset_folder
            track_results_folder = self.test_track_results_folder
        elif dataset_type == "train":
            dataset_folder = self.train_dataset_folder
            track_results_folder = self.train_track_results_folder
        else:
            raise Exception("Wrong dataset type provided: {}".format(dataset_type))

        def get_single_cam_clustering_results_path():


            folder_path = os.path.join(self.work_dirs, "clustering"
                                       , "single_camera_clustering_results"
                                       , os.path.basename(track_results_folder))

            os.makedirs(folder_path, exist_ok=True)

            csv_path = os.path.join(folder_path, "single_camera_clustering_results.csv")

            return csv_path

        def get_best_weights_one_cam(best_weights, default_weights,cam_id):
            best_weights = best_weights.astype({"cam_id": int})

            cam_id_best_weights = best_weights[best_weights["cam_id"] == cam_id]

            if len(cam_id_best_weights) == 0:
                return default_weights

            new_dist_name_to_weight = {}
            for dist_name, dist_weight in default_weights.items():

                weight_of_dist_name = cam_id_best_weights[cam_id_best_weights["dist_name"] == dist_name]

                if len(weight_of_dist_name) > 0:

                    weight_of_dist_name = weight_of_dist_name.iloc[0]["dist_weight"]

                    new_dist_name_to_weight[dist_name] = weight_of_dist_name

                else:
                    new_dist_name_to_weight[dist_name] = dist_weight

            return new_dist_name_to_weight

        start_time = time.time()
        best_weights = pd.read_csv(best_weights_path)
        evaluation_results = pd.DataFrame()

        for cam_id in cam_ids:
            best_weights_one_cam = get_best_weights_one_cam(best_weights=best_weights
                                                            ,default_weights=default_weights
                                                            ,cam_id=cam_id)

            self.logger.info(get_elapsed_time_and_msg(start_time,str(best_weights_one_cam)))
            self.logger.info("Starting clustering for cam_id: {}".format(cam_id))
            clustering_results = self.cluster_via_hierarchical(cam_id=cam_id
                                                                ,dist_name_to_distance_weights=best_weights_one_cam
                                                                ,threshold=1.0
                                                                ,person_count=1
                                                                ,dataset_type=dataset_type)

            result = Motmetrics_evaluation(
                ground_truth_path=os.path.join(dataset_folder, "cam_{}".format(cam_id),
                                               "coords_cam_{}.csv".format(cam_id))
                , track_results_path=clustering_results["track_results_path"]
                , working_dir=self.work_dirs).evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)

            self.logger.info(get_elapsed_time_and_msg(start_time, result["strsummary"]))
            evaluation_summary = result["summary"]

            evaluation_summary["cam_id"] = cam_id

            evaluation_results = evaluation_results.append(evaluation_summary)

        single_cam_clustering_results_path = get_single_cam_clustering_results_path()
        evaluation_results.to_csv(single_cam_clustering_results_path)





    def get_best_weights_path(self):

        best_weights_folder = os.path.join(self.work_dirs, "clustering"
                                             ,"single_camera_best_weights"
                                             ,os.path.basename(self.train_track_results_folder) )

        os.makedirs(best_weights_folder,exist_ok=True)

        best_weights_path = os.path.join(best_weights_folder,"best_weights.csv")

        return best_weights_path


    def find_weights(self,cam_ids,weight_search_configs,dist_name_to_distance_weights):

        original_dist_name_to_distance_weights = copy.deepcopy(dist_name_to_distance_weights)
        def get_best_weight(evaluation_results,metric="idf1"):

            max_row = evaluation_results.loc[evaluation_results[metric].idxmax()]

            return max_row["weight"]

        def evaluate_one_config(cam_id,weight_search_config,dist_name_to_distance_weights):
            dist_name = weight_search_config["dist_name"]
            weight_start = weight_search_config["start_value"]
            weight_stop = weight_search_config["stop_value"]
            steps = weight_search_config["steps"]
            search_weights = np.linspace(start=weight_start,stop=weight_stop,num=steps)

            evaluation_results = pd.DataFrame()

            for weight_no,weight in enumerate(search_weights):
                dist_name_to_distance_weights[dist_name] = weight

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          , message="Starting clustering for dist_name: {} weight: {} "
                                                                    "weight_no of weights: {} of {}".format(dist_name,weight,weight_no+1,len(search_weights))))

                clustering_results = self.cluster_via_hierarchical(cam_id=cam_id
                                              ,dataset_type="train"
                                              ,dist_name_to_distance_weights=dist_name_to_distance_weights
                                              ,person_count=1
                                              ,threshold=1.0)

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          ,
                                                          message="Starting evaluation for dist_name: {} weight: {} ".format(
                                                              dist_name, weight)))

                result = Motmetrics_evaluation(
                    ground_truth_path=os.path.join(self.train_dataset_folder, "cam_{}".format(cam_id),
                                                   "coords_cam_{}.csv".format(cam_id))
                    , track_results_path=clustering_results["track_results_path"]
                    , working_dir=self.work_dirs).evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)


                evaluation_summary = result["summary"]

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          ,
                                                          message="Finished evaluation for dist_name: {} weight: {} \n result: {}".format(
                                                              dist_name, weight, result["strsummary"])))

                evaluation_summary["weight"] = weight
                evaluation_summary["dist_name"] = dist_name
                evaluation_summary["cam_id"] = cam_id
                evaluation_summary["track_count"] = clustering_results["track_count"]
                evaluation_results = evaluation_results.append(evaluation_summary, ignore_index=True)

            return evaluation_results

        def save_weight_evaluation_results(evaluation_results):

            weight_evaluation_output_folder = os.path.join(self.work_dirs,"clustering"
                                                           ,"single_camera_weight_evaluations"
                                                           ,os.path.basename(self.train_track_results_folder))


            os.makedirs(weight_evaluation_output_folder,exist_ok=True)

            output_path = os.path.join(weight_evaluation_output_folder,"weight_evaluation.csv")

            print("Saving weight evaluation results to: {}".format(output_path))

            evaluation_results.to_csv(output_path)

        def save_best_weights(cam_id_to_dist_name_to_distance_weights):

            best_weights_df = pd.DataFrame({ "dist_name" : [] , "dist_weight" : [] , "cam_id" : []})

            for cam_id, dist_name_to_distance_weights in cam_id_to_dist_name_to_distance_weights.items():
                for dist_name, distance_weight in dist_name_to_distance_weights.items():
                    best_weights_df = best_weights_df.append({ "dist_name" : dist_name
                                                               ,"dist_weight" : distance_weight
                                                               ,"cam_id" : cam_id},ignore_index=True)

            best_weights_path = self.get_best_weights_path()

            print("Saving best weights to: {}".format(best_weights_path))
            best_weights_df.to_csv(best_weights_path)

        find_weights_start_time = time.time()

        all_evaluation_results = pd.DataFrame()
        cam_id_to_dist_name_to_distance_weights = {}
        for cam_id in cam_ids:
            evaluation_results = pd.DataFrame()
            dist_name_to_distance_weights = copy.deepcopy(original_dist_name_to_distance_weights)
            self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                      ,message="Starting weight search for cam_id: {}".format(cam_id)))

            for weight_search_config in weight_search_configs:
                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          , message="Starting evaluation of weights for dist_name: {}".format(
                        weight_search_config["dist_name"])))

                evaluation_results = evaluate_one_config(cam_id,weight_search_config,dist_name_to_distance_weights)

                best_weight = get_best_weight(evaluation_results)
                dist_name = evaluation_results.iloc[0]["dist_name"]

                dist_name_to_distance_weights[dist_name] = best_weight

                all_evaluation_results = all_evaluation_results.append(evaluation_results)
            cam_id_to_dist_name_to_distance_weights[cam_id] = dist_name_to_distance_weights

        save_best_weights(cam_id_to_dist_name_to_distance_weights)
        save_weight_evaluation_results(all_evaluation_results)







if __name__ == "__main__":



    '''
    scc = Single_cam_clustering(work_dirs="/media/philipp/philippkoehl_ssd/work_dirs"
                                ,train_track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta1207"
                                ,test_track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta1207"
                                ,train_dataset_folder="/home/philipp/Downloads/Recording_12.07.2019"
                                ,test_dataset_folder="/home/philipp/Downloads/Recording_12.07.2019"
                                ,cam_count=6
                                ,person_identifier="ped_id")
    '''


    scc = Single_cam_clustering(work_dirs="/media/philipp/philippkoehl_ssd/work_dirs"
                                , train_track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train"
                                , test_track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
                                , train_dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/train"
                                , test_dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                                , cam_count=6
                                , person_identifier="person_id")



    '''
    scc = Single_cam_clustering(work_dirs="/home/koehlp/Downloads/work_dirs"
                                ,train_track_results_folder="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train_iosb"
                                ,test_track_results_folder="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test_iosb"
                                ,train_dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/train"
                                ,test_dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/test"
                                ,cam_count=6
                                ,person_identifier="person_id")
                                
    '''

    dist_name_to_distance_weights = {"are_tracks_disjunct": 1.0
        , "track_pred_pos_start_distance": 50
        , "track_cosine_distance": 0.0
        , "homography_match_score": 5.0}


    scc.cluster_via_hierarchical(1,dist_name_to_distance_weights=dist_name_to_distance_weights,dataset_type="train")

    #scc.combine_all_cams(range(6))

    #scc.create_best_cam_settings(cam_ids=range(6), start_thresh=0.5, end_thresh=5.0, step_count=1)

    #scc.cluster_from_cam_settings(use_threshold=True)

