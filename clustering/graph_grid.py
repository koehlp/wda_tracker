

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

from clustering.clustering_utils import get_person_id_to_track
from clustering.clustering_utils import save_combined_tracks

from utilities.helper import drawBoundingBox

def init_defaultdict_int():
    return 0

class Graph_grid:
    def __init__(self,work_dirs
                 ,cam_coords_path
                 ,cam_images_folder
                 ,track_results_folder
                 ,visualization_img_path
                 ,track_result_cam_ids
                 ,draw_videos_output_folder
                 ,img_dims=(1920,1080)
                 ,cell_size=(50,50)
                 ,person_identifier="ped_id"):

        self.draw_videos_output_folder = draw_videos_output_folder
        self.track_result_cam_ids = track_result_cam_ids
        self.track_results_folder = track_results_folder
        self.cam_images_folder = cam_images_folder
        self.work_dirs = work_dirs
        self.visualization_img_path = visualization_img_path
        self.cam_coords_path = cam_coords_path
        self.cell_size = cell_size
        self.person_identifier = person_identifier
        self.x_split_count = int(img_dims[0] / self.cell_size[0]) + 1
        self.y_split_count = int(img_dims[1] / self.cell_size[0]) + 1

        self.x_grid_positions = np.linspace(0,img_dims[0],self.x_split_count)
        self.y_grid_positions = np.linspace(0,img_dims[1],self.y_split_count)


        self.node_to_nodes = defaultdict(set)
        self.edge_to_transition_distances = defaultdict(list)
        self.edge_to_frequency = defaultdict(init_defaultdict_int)

        self.graph_node_position_list = []
        self.person_appeared_cell_to_count = defaultdict(init_defaultdict_int)
        self.cached_position_to_nearest_cell = {}

        self.img_dims = img_dims


    def load_groundtruth_csv(self):
        self.cam_coords = load_csv(self.work_dirs, self.cam_coords_path)

        self.cam_coords = self.group_cam_coords(self.cam_coords)


    def get_bbox_height(self,bbox):
        xtl, ytl, xbr, ybr = bbox
        return ybr - ytl

    def get_bbox_middle_pos(self,bbox):

        xtl, ytl, xbr, ybr = bbox
        x = xtl + ((xbr - xtl) / 2.)
        y = ytl + ((ybr - ytl) / 2.)

        return (x,y)

    def cell_indices_to_center_pos(self,from_cell):
        x_grid_idx_from, y_grid_idx_from = from_cell
        xtl_grid_cell_from = self.x_grid_positions[x_grid_idx_from]
        ytl_grid_cell_from = self.y_grid_positions[y_grid_idx_from]
        xbr_grid_cell_from = self.x_grid_positions[x_grid_idx_from + 1]
        ybr_grid_cell_from = self.y_grid_positions[y_grid_idx_from + 1]

        cell_bbox_from = (xtl_grid_cell_from, ytl_grid_cell_from, xbr_grid_cell_from, ybr_grid_cell_from)

        cell_center_from = self.get_bbox_middle_pos(cell_bbox_from)

        cell_center_from = tuple(map(int,cell_center_from))
        return cell_center_from

    def draw_graph(self,img):

        img = cv2.imread(img)
        max_freq = max(self.edge_to_frequency.values())

        for edge, edge_count in self.edge_to_frequency.items():
            from_cell, to_cell = edge

            from_cell_center = self.cell_indices_to_center_pos(from_cell)

            to_cell_center = self.cell_indices_to_center_pos(to_cell)

            cv2.circle(img,from_cell_center,radius=2,color=(0,0,255),thickness=-1)
            cv2.circle(img, to_cell_center,radius=2,color=(0,0,255),thickness=-1)



            thickness = (10-1) / (max_freq - 1) * (edge_count - max_freq) + 10

            #edge_count = (max'-min') / (max - min) * (value - max) + max
            thickness = int(thickness)



            cv2.line(img, from_cell_center, to_cell_center, color=(0,255,0), thickness=thickness)

        cv2.imshow("show", img)
        cv2.waitKey(0)



    def group_cam_coords(self,cam_coords):
        if "ped_id" in cam_coords.columns:
            person_id_name = "ped_id"
        else:
            person_id_name = "person_id"

        cam_coords = cam_coords.groupby(["frame_no_gta", person_id_name], as_index=False).mean()
        cam_coords = adjustCoordsTypes(cam_coords,self.person_identifier)

        return cam_coords

    def is_person_pos_in_cell(self,cell_bbox,person_pos):
        x_person_pos,y_person_pos = person_pos

        xtl, ytl, xbr,ybr = cell_bbox

        return (x_person_pos >= xtl) \
               and ((x_person_pos < xbr) or x_person_pos == self.img_dims[0]) \
               and (y_person_pos >= ytl) \
               and ((y_person_pos < ybr) or y_person_pos == self.img_dims[1])


    def get_person_cell(self,person_pos):
        for x_grid_idx in range(len(self.x_grid_positions) - 1):
            for y_grid_idx in range(len(self.y_grid_positions) - 1):
                xtl_grid_cell = self.x_grid_positions[x_grid_idx]
                ytl_grid_cell = self.y_grid_positions[y_grid_idx]
                xbr_grid_cell = self.x_grid_positions[x_grid_idx + 1]
                ybr_grid_cell = self.y_grid_positions[y_grid_idx + 1]

                is_person_in_cell = self.is_person_pos_in_cell((xtl_grid_cell, ytl_grid_cell, xbr_grid_cell, ybr_grid_cell), person_pos)

                if is_person_in_cell:
                    return (x_grid_idx,y_grid_idx)


        assert(False,"Person position outside of image.")


    def load_graph_grid(self):

        cam_coords_folder, cam_coords_filename = os.path.split(self.cam_coords_path)
        cam_coords_folder = os.path.normpath(cam_coords_folder)
        dataset_foldername = cam_coords_folder.split(os.sep)[-2]

        graph_grid_folder = os.path.join(self.work_dirs,"clustering","graph_grid",dataset_foldername)

        os.makedirs(graph_grid_folder,exist_ok=True)

        graph_grid_pickle_path = os.path.join(graph_grid_folder,"graph_grid.pkl")

        if os.path.exists(graph_grid_pickle_path):
            with open(graph_grid_pickle_path, "rb") as pickle_file:

                graph_grid_dict = pickle.load(pickle_file)
                self.node_to_nodes = graph_grid_dict["node_to_nodes"]

                self.edge_to_frequency = graph_grid_dict["edge_to_frequency"]

                self.person_appeared_cell_to_count = graph_grid_dict["person_appeared_cell_to_count"]

                self.start_cell_frequency = graph_grid_dict["start_cell_frequency"]

                self.end_cell_frequency = graph_grid_dict["end_cell_frequency"]

                self.edge_to_transition_distances = graph_grid_dict["edge_to_transition_distances"]

                self.person_id_to_track_positions = graph_grid_dict["person_id_to_track_positions"]

                self.edge_to_mean_transition_distance = self.calculate_edge_to_mean_distance(self.edge_to_transition_distances)

                self.all_persons_mean_velocity = self.calculate_all_persons_mean_velocity(self.person_id_to_track_positions)


        else:
            self.calculate_graph_grid()
            self.edge_to_mean_transition_distance = self.calculate_edge_to_mean_distance(self.edge_to_transition_distances)
            self.all_persons_mean_velocity = self.calculate_all_persons_mean_velocity(self.person_id_to_track_positions)

            with open(graph_grid_pickle_path, "wb") as pickle_file:


                graph_grid_dict = { "node_to_nodes" : self.node_to_nodes
                                    , "edge_to_frequency" : self.edge_to_frequency
                                    , "person_appeared_cell_to_count" : self.person_appeared_cell_to_count
                                    , "start_cell_frequency" : self.start_cell_frequency
                                    , "end_cell_frequency" : self.end_cell_frequency
                                    , "edge_to_transition_distances" : self.edge_to_transition_distances
                                    , "person_id_to_track_positions" : self.person_id_to_track_positions}

                print("Saved graph grid")
                pickle.dump(graph_grid_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


    def get_bbox_from_row(self,row):

        return (row["xtl"],row["ytl"],row["xbr"],row["ybr"])


    def count_track_fragments(self):
        self.load_calculated_tracks_per_person()

        track_fragment_len_to_count = defaultdict(lambda: 0)

        for tracks_per_person in self.person_id_to_tracks.values():
            track_fragment_len_to_count[len(tracks_per_person)] += 1


        for frag_len, frag_count in track_fragment_len_to_count.items():
            print("fragment count: {} with frequency: {}".format(frag_len,frag_count))

    def draw_fragment_length_histogram(self):

        import matplotlib.pyplot as plt
        from collections import Counter

        self.load_calculated_tracks_per_person()
        track_fragment_lengths = []
        for tracks_per_person in self.person_id_to_tracks.values():
            for track_fragment in tracks_per_person:
                track_fragment_lengths.append(len(track_fragment))

        _ = plt.hist(track_fragment_lengths, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")

        for fragment_length, count_ in sorted(Counter(track_fragment_lengths).items(), key=lambda item: item[1]):
            print("fragment_length: {} with frequency: {}".format(fragment_length, count_))


    def draw_fragment_end_positions(self):
        self.load_calculated_tracks_per_person()

        img = cv2.imread(self.visualization_img_path)

        for tracks_per_person in self.person_id_to_tracks.values():
            for track_fragment in tracks_per_person:
                end_bbox = track_fragment[-1]["bbox"]
                end_position = self.get_bbox_middle_pos(end_bbox)
                end_position = tuple(map(int,end_position))

                cv2.circle(img, end_position, radius=1, color=(0, 0, 255), thickness=-1)

        cv2.imshow("show", img)
        cv2.waitKey(0)


    def create_node_position_list(self):
        if len(self.graph_node_position_list) > 0:
            return

        for graph_node in self.node_to_nodes.keys():
            self.graph_node_position_list.append((graph_node ,self.cell_indices_to_center_pos(graph_node)))

    def get_nearest_graph_node(self,query_position):
        self.create_node_position_list()

        real_cell = self.get_person_cell(query_position)
        if real_cell in self.node_to_nodes:
            return real_cell

        if query_position in self.cached_position_to_nearest_cell:
            return self.cached_position_to_nearest_cell[query_position]

        current_min = (None,sys.maxsize)
        for cell, position in self.graph_node_position_list:

            dist = np.linalg.norm(np.array(position)-np.array(query_position))
            if dist < current_min[1]:
                current_min = (cell,dist)

        self.cached_position_to_nearest_cell[query_position] = current_min[0]

        return current_min[0]



    def draw_track_videos(self, tracks, cam_id, output_folder=None):
        print("drawing combined tracks in images.")


        if output_folder is None:
            output_folder = os.path.join(self.work_dirs, "clustering", "drawn_track_videos")

        print("Output Path: {}".format(output_folder))

        def flat_track_to_fragments(track):
            result = []

            track = sorted(track,key=lambda x: x["frame_no_cam"])

            from itertools import groupby

            for k, frame_no_group in groupby(enumerate(track), lambda x: x[0] - x[1]["frame_no_cam"]):
                result.append(list(map(lambda x: x[1], frame_no_group)))

            return result

        def track_fragments_split_filled_bboxes(track_fragments):
            tracks_fragmented = []

            for fragment in track_fragments:

                # separating the track positions of every fragment
                result_filled = []
                not_filled = []
                for track_pos in fragment:

                    if "filled_bbox" in track_pos:
                        result_filled.append(track_pos)

                    else:
                        not_filled.append(track_pos)

                #Because a filled part of the fragment may be enclosed of not filled the separated postions have to be fragmented
                filled_fragments = []
                if len(result_filled) > 0:
                    filled_fragments = flat_track_to_fragments(result_filled)

                not_filled_fragments = []
                if len(not_filled) > 0:
                    not_filled_fragments = flat_track_to_fragments(not_filled)

                splitted_fragments = not_filled_fragments + filled_fragments

                def max_key(track_pos):
                    return track_pos["frame_no_cam"]

                def sort_key(fragment):
                    return max(fragment,key=max_key)["frame_no_cam"]

                #Then they have to be sorted
                splitted_fragments = sorted(splitted_fragments,key=sort_key)

                for new_frag in splitted_fragments: tracks_fragmented.append(new_frag)




            return tracks_fragmented



        def flat_tracks_to_fragments(tracks):
            result = []
            for track in tracks:
                one_track_fragments = flat_track_to_fragments(track)


                fragments = track_fragments_split_filled_bboxes(one_track_fragments)
                result.append(fragments)
            return result

        def get_fragmented_track_length(track):
            result = 0
            for track_frag in track:
                result += len(track_frag)

            return result

        def draw_fragmented_track(track,img,until_idx):
            current_idx = 0
            for track_frag_no, track_fragment in enumerate(track):

                if track_frag_no % 2 == 0:
                    line_color = (0, 0, 255)  # red
                else:
                    line_color = (0, 255, 0)  # green

                for track_pos_idx in range(len(track_fragment)):

                    pos_bbox_start = track_fragment[track_pos_idx]["bbox"]
                    track_pos_2d_start = self.get_bbox_middle_pos(pos_bbox_start)
                    track_pos_2d_start = tuple(map(int, track_pos_2d_start))

                    if "filled_bbox" in track_fragment[track_pos_idx]:
                        line_color = (255, 0, 0)



                    # Draw circle at end and beginning of a fragment
                    if track_pos_idx == 0 or track_pos_idx == (len(track_fragment) - 1):
                        cv2.circle(img, track_pos_2d_start, radius=3, color=(255, 255, 0), thickness=-1)


                    cv2.circle(img, track_pos_2d_start, radius=1, color=line_color, thickness=-1)

                    if current_idx == until_idx:
                        pos_bbox_start = tuple(map(int,pos_bbox_start))
                        drawBoundingBox(img,pos_bbox_start)

                        track_length = get_fragmented_track_length(track)

                        cv2.putText(img=img, text="Track Length {}".format(track_length), org=(1000, 850),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX
                                    , fontScale=0.6, color=(255, 255, 255), thickness=2)


                        cv2.putText(img=img, text="Fragment No. {}".format(track_frag_no), org=(1000, 900),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX
                                    , fontScale=0.6, color=(255, 255, 255), thickness=2)

                        cv2.putText(img=img, text="Frame No. {}".format(track_fragment[track_pos_idx]["frame_no_cam"]), org=(1000, 950),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX
                                    , fontScale=0.6, color=(255, 255, 255), thickness=2)

                        #Has to be here because it should be only drawn one time
                        if "filled_bbox" in track_fragment[track_pos_idx]:
                            cv2.putText(img=img, text="Interpolated Bounding Box", org=(1000, 1000), fontFace=cv2.FONT_HERSHEY_SIMPLEX
                                        , fontScale=0.5, color=(255, 255, 255), thickness=2)




                    current_idx += 1

                    if current_idx > until_idx:
                        return img

            return img

        def draw_video_one_track(track,track_idx,cam_id):
            drawn_video_folder = os.path.join(output_folder,"cam_{}".format(cam_id), "track_no_{}".format(track_idx))
            os.makedirs(drawn_video_folder, exist_ok=True)
            track_pos_idx = 0
            for track_fragment in track:
                for track_pos in track_fragment:
                    track_pos_idx += 1

                    cam_image_name = os.path.join(self.cam_images_folder,
                                                  "image_{}_{}.jpg".format(track_pos["frame_no_cam"],cam_id))
                    track_image = cv2.imread(cam_image_name)

                    track_image = draw_fragmented_track(track,track_image,track_pos_idx)

                    cv2.imwrite(os.path.join(drawn_video_folder,
                                             "track_img_{}.jpg".format(track_pos_idx))
                                , track_image)



        tracks = flat_tracks_to_fragments(tracks)
        #draw_video_one_track(tracks[40], 40, cam_id)


        for track_idx, track in tqdm(enumerate(tracks)):
            draw_video_one_track(track,track_idx,cam_id)
            





    def draw_combined_tracks(self,tracks):
        print("drawing combined tracks in images.")
        img = cv2.imread(self.visualization_img_path)
        drawn_tracks_folder = os.path.join(self.work_dirs, "clustering", "drawn_tracks")
        os.makedirs(drawn_tracks_folder,exist_ok=True)

        for track_idx,track in enumerate(tracks):
            last_track_no = -1
            track_image = img.copy()

            if len(track) <= 1:
                continue

            for track_pos_idx in range(len(track)-1):
                if track[track_pos_idx]["track_no"] % 2 == 0:
                    line_color = (0,0,255)
                else:
                    line_color = (0, 255, 0)

                pos_bbox_start = track[track_pos_idx]["bbox"]
                pos_bbox_end = track[track_pos_idx+1]["bbox"]
                track_pos_2d_start = self.get_bbox_middle_pos(pos_bbox_start)
                track_pos_2d_end = self.get_bbox_middle_pos(pos_bbox_end)
                track_pos_2d_start = tuple(map(int,track_pos_2d_start))
                track_pos_2d_end = tuple(map(int, track_pos_2d_end))


                cv2.line(track_image, track_pos_2d_start, track_pos_2d_end, color=line_color, thickness=1)
                #cv2.circle(track_image, track_pos_2d_end, radius=1, color=(155,255,0), thickness=-1)
                #cv2.circle(track_image, track_pos_2d_start, radius=1, color=(155,255,0), thickness=-1)

                if "filled_bbox" in track[track_pos_idx]:
                    drawBoundingBox(track_image,np.array(pos_bbox_start,dtype=int),(255,0,0))

                if "filled_bbox" in track[track_pos_idx+1]:
                    drawBoundingBox(track_image, np.array(pos_bbox_end,dtype=int), (255, 0, 0))

            cv2.imwrite(os.path.join(drawn_tracks_folder,"track_img_{}.jpg".format(track_idx)),track_image)


    def draw_some_results_tracks(self):
        import random
        import cmapy
        self.load_calculated_tracks_per_person()

        img = cv2.imread(self.visualization_img_path)

        person_id_to_tracks_choices = random.choices(list(self.person_id_to_tracks.values()),k=100)


        for person_no,tracks_per_person in enumerate(list(self.person_id_to_tracks.values())[:100]):
            for track_fragment in tracks_per_person:
                for track_pos_idx in range(len(track_fragment)-1):
                    pos_bbox_start = track_fragment[track_pos_idx]["bbox"]
                    pos_bbox_end = track_fragment[track_pos_idx+1]["bbox"]
                    track_pos_2d_start = self.get_bbox_middle_pos(pos_bbox_start)
                    track_pos_2d_end = self.get_bbox_middle_pos(pos_bbox_end)
                    track_pos_2d_start = tuple(map(int,track_pos_2d_start))
                    track_pos_2d_end = tuple(map(int, track_pos_2d_end))
                    bgr_color = cmapy.color('gist_ncar', person_no)

                    #cv2.circle(img, track_pos_2d_start, radius=1, color=bgr_color, thickness=-1)
                    cv2.line(img, track_pos_2d_start, track_pos_2d_end, color=bgr_color, thickness=1)
                    #cv2.circle(img, track_pos_2d_end, radius=1, color=bgr_color, thickness=-1)


        cv2.imshow("show", img)
        cv2.waitKey(0)

    def calculate_loga_graph(self):
        self.loga_graph = nx.Graph()
        max_freq = max(self.edge_to_frequency.values())
        for edge, edge_count in self.edge_to_frequency.items():
            edge_weight = math.log10(float(max_freq)/float(edge_count))


            self.loga_graph.add_edge(edge[0],edge[1], weight=edge_weight)




    def multiply_path_weights(self,path):

        result = 1
        for path_index in range(len(path)-1):

            node1 = path[path_index]
            node2 = path[path_index+1]

            edge_weight = self.edge_to_frequency[(node1,node2)]
            if edge_weight == 0:
                edge_weight = self.edge_to_frequency[(node2, node1)]

            result *= (1. / float(edge_weight))

        return result


    def get_track_results_simple(self,cam_id):

        track_results_path = os.path.join(self.track_results_folder,"track_results_cam_{}.txt".format(cam_id))


        self.track_results = pd.read_csv(track_results_path)
        person_id_to_track = defaultdict(list)
        frame_numbers = self.track_results.groupby("frame_no_cam",as_index=False).mean()
        frame_numbers = frame_numbers["frame_no_cam"].tolist()
        frame_numbers = list(map(int,frame_numbers))

        frame_numbers = sorted(frame_numbers)
        for frame_no in frame_numbers:
            one_frame = self.track_results[self.track_results["frame_no_cam"] == frame_no]

            for index,track_row in one_frame.iterrows():

                bbox = (track_row["xtl"],track_row["ytl"],track_row["xbr"],track_row["ybr"])
                person_id_to_track[track_row["person_id"]].append({"frame_no_cam" : track_row["frame_no_cam"]
                                                                   ,"bbox" : bbox})



        return person_id_to_track.values()



    def get_combined_fragments(self,person_id_to_tracks):

        tracks_all_persons = []
        for track_fragments_of_person in self.person_id_to_tracks.values():
            one_track = [item for sublist in track_fragments_of_person for item in sublist]
            tracks_all_persons.append(one_track)



        return tracks_all_persons

    def get_separated_fragments(self,person_id_to_tracks):

        tracks_all_persons = []
        for track_fragments_of_person in self.person_id_to_tracks.values():
            for fragment in track_fragments_of_person:
                tracks_all_persons.append(fragment)



        return tracks_all_persons



    def calculate_track_velocity(self,track,n_tail=5):
        last_n_positons = track[-n_tail:]

        first_pos_track1 = self.get_bbox_middle_pos(last_n_positons[0]["bbox"])
        second_pos_track1 = self.get_bbox_middle_pos(last_n_positons[-1]["bbox"])
        first_pos_frame_no_track1 = last_n_positons[0]["frame_no_cam"]
        second_pos_frame_no_track1 = last_n_positons[-1]["frame_no_cam"]

        velocity = (np.array(second_pos_track1) - np.array(first_pos_track1)) / (second_pos_frame_no_track1 - first_pos_frame_no_track1)


        return velocity


    def get_predicted_position(self,track1,track2,n_tail=40):


        #We need track1[-1] < track2[0]
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        last_n_positons = track1[-n_tail:]




        first_pos_track1 = self.get_bbox_middle_pos(last_n_positons[0]["bbox"])
        second_pos_track1 = self.get_bbox_middle_pos(last_n_positons[-1]["bbox"])
        first_pos_frame_no_track1 = last_n_positons[0]["frame_no_cam"]
        second_pos_frame_no_track1 = last_n_positons[-1]["frame_no_cam"]

        if len(track1) == 1:
            return np.array(first_pos_track1)

        velocity = (np.array(second_pos_track1) - np.array(first_pos_track1)) / (second_pos_frame_no_track1 - first_pos_frame_no_track1)


        start_pos_frame_no_track2 = track2[0]["frame_no_cam"]

        passed_time = start_pos_frame_no_track2 - second_pos_frame_no_track1

        predicted_position = np.array(second_pos_track1) + (velocity * passed_time)


        return predicted_position


    def are_tracks_disjunct(self,track1,track2):

        return set([track_pos["frame_no_cam"] for track_pos in track1]).isdisjoint([track_pos["frame_no_cam"] for track_pos in track2])



    def get_track_cosine_distance(self,track1,track2,n_tail=40):
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp


        track1_tail = track1[-n_tail:]
        track2_head = track2[:n_tail]

        track1_direction = np.array(self.get_bbox_middle_pos(track1_tail[-1]["bbox"])) - np.array(self.get_bbox_middle_pos(track1_tail[0]["bbox"]))

        track2_direction = np.array(self.get_bbox_middle_pos(track2_head[-1]["bbox"])) - np.array(self.get_bbox_middle_pos(track2_head[0]["bbox"]))

        if np.count_nonzero(track1_direction) == 0 or np.count_nonzero(track2_direction) == 0:
            return 0.0

        result = spatial.distance.cosine(track1_direction,track2_direction)

        return result


    def get_older_track_bbox_height(self,track1,track2):
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        track1_bbox = track1[-1]["bbox"]

        return self.get_bbox_height(track1_bbox)


    def get_average_bbox_height(self,track1,track2):
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp


        track1_bbox = track1[-1]["bbox"]
        track2_bbox = track2[0]["bbox"]

        result = (self.get_bbox_height(track1_bbox) + self.get_bbox_height(track2_bbox)) / 2.
        return result

    def get_track_pred_pos_start_distance(self, track1, track2):


        # We need track1[-1] < track2[0]
        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        pred_position = self.get_predicted_position(track1, track2)

        track2_start = self.get_bbox_middle_pos(track2[0]["bbox"])
        bbox_height = self.get_bbox_height(track1[-1]["bbox"])
        return np.linalg.norm(np.array(pred_position)-np.array(track2_start)) / bbox_height

    def combine_tracks(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        return track1 + track2

    def get_tracks_distance(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp


        return np.linalg.norm(np.array(self.get_bbox_middle_pos(track2[0]["bbox"])) - np.array(self.get_bbox_middle_pos(track1[-1]["bbox"])))

    def mark_tracker_tracks(self,tracks):
        result_tracks = []
        for track_no, track in enumerate(tracks):
            marked_track = []
            for track_pos in track:

                track_pos["track_no"] = track_no

                marked_track.append(track_pos)
            result_tracks.append(marked_track)
        return result_tracks


    def just_close_gaps(self):
        tracks_all_persons = get_person_id_to_track(self.track_results_path).values()

        tracks_all_persons = self.mark_tracker_tracks(tracks_all_persons)
        tracks_all_persons = self.close_track_gaps_interpolation(tracks_all_persons)
        #self.draw_combined_tracks(tracks_all_persons)
        print("number of tracks: {}".format(len(tracks_all_persons)))

        save_combined_tracks(tracks_all_persons,work_dirs=self.work_dirs)


    def is_vector_pointing_towards_point(self,vector_point1,vector_point2,point):

        distance_vector_point1_to_point = np.linalg.norm(np.array(vector_point1) - np.array(point))

        distance_vector_point2_to_point = np.linalg.norm(np.array(vector_point2) - np.array(point))

        return distance_vector_point2_to_point < distance_vector_point1_to_point

    def older_track_ends_in_vanish_point(self,track1,track2,distance_from_vanish_point=30,min_freq=2,n_tail=40):


        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        track1_tail = track1[-n_tail:]
        track1_tail_start = self.get_bbox_middle_pos(track1_tail[0]["bbox"])
        track1_tail_end = self.get_bbox_middle_pos(track1_tail[-1]["bbox"])

        for cell,freq in self.end_cell_frequency.items():
            if freq >= min_freq:
                person_vanish_pos = self.cell_indices_to_center_pos(cell)

                #Check if the track is moving in a direction where persons vanish
                pointing_to_vanish_point = self.is_vector_pointing_towards_point(track1_tail_start,track1_tail_end,person_vanish_pos)

                #Check if the track end is near a point where persons vanish
                track_end_is_in_vanish_point_distance = np.linalg.norm(np.array(track1_tail_end)-np.array(person_vanish_pos)) < distance_from_vanish_point

                if pointing_to_vanish_point and track_end_is_in_vanish_point_distance:
                    return True

        return False



    def combine_via_min_predicted_pos_distance_hierarchical_nn_compare(self, number_of_persons=105, max_bbox_height_dist=2):
        self.load_graph_grid()
        tracks_all_persons = get_person_id_to_track(self.track_results_path).values()
        tracks_all_persons = self.mark_tracker_tracks(tracks_all_persons)
        pbar = tqdm(total=(len(tracks_all_persons)-number_of_persons))

        while len(tracks_all_persons) > number_of_persons:
            pbar.update()

            current_min = (None,None,sys.maxsize)

            for candidate_track_idx, candidate_track in enumerate(tracks_all_persons):
                for partner_track_idx, partner_track in enumerate(tracks_all_persons):

                    if not self.are_tracks_disjunct(candidate_track,partner_track):
                        continue


                    pred_pos_start_dist = self.get_track_pred_pos_start_distance(candidate_track, partner_track)


                    cosine_distance_tracks = self.get_track_cosine_distance(candidate_track,partner_track)
                    pred_pos_start_dist += cosine_distance_tracks

                    current_min = min(current_min,(candidate_track,partner_track,pred_pos_start_dist),key=lambda x:x[-1])


            candidate_track = current_min[0]
            tracks_all_persons.remove(current_min[0])
            partner_track = current_min[1]
            tracks_all_persons.remove(current_min[1])

            tracks_all_persons.append(self.combine_tracks(candidate_track,partner_track))



        tracks_all_persons = self.close_track_gaps_interpolation(tracks_all_persons)

        print("number of tracks: {}".format(len(tracks_all_persons)))

        save_combined_tracks(tracks_all_persons,work_dirs=self.work_dirs)


    def combine_all_cams(self):
        for cam_id in self.track_result_cam_ids:
            self.combine_via_min_predicted_pos_distance(cam_id=cam_id)


    def combine_via_min_predicted_pos_distance(self,cam_id, number_of_persons=1, clustering_threshold=20):

        tracks_all_persons = self.get_track_results_simple(cam_id)
        tracks_all_persons = self.mark_tracker_tracks(tracks_all_persons)
        not_combinable_tracks = []
        #pbar = tqdm(total=(len(tracks_all_persons)-number_of_persons))

        while len(tracks_all_persons) > number_of_persons:
            #pbar.update()
            candidate_track = tracks_all_persons.pop(0)

            current_min = (None,sys.maxsize)
            for partner_track_idx, partner_track in enumerate(tracks_all_persons):

                if not self.are_tracks_disjunct(candidate_track,partner_track):
                    continue

                pred_pos_start_dist = self.get_track_pred_pos_start_distance(candidate_track, partner_track)


                cosine_distance_tracks = self.get_track_cosine_distance(candidate_track,partner_track)
                pred_pos_start_dist += cosine_distance_tracks


                current_min = min(current_min,(partner_track_idx,pred_pos_start_dist),key=lambda x:x[1])


            #The candidate track cant be combined with any other track anymore
            if current_min[0] is None:
                not_combinable_tracks.append(candidate_track)
                continue

            partner_track = tracks_all_persons.pop(current_min[0])
            tracks_all_persons.append(self.combine_tracks(candidate_track,partner_track))


            if current_min[1] > clustering_threshold:
                break

        tracks_all_persons.extend(not_combinable_tracks)

        tracks_all_persons = self.close_track_gaps_interpolation(tracks_all_persons)
        #self.draw_combined_tracks(tracks_all_persons)

        self.draw_track_videos(tracks_all_persons,cam_id=cam_id,output_folder=self.draw_videos_output_folder)
        print("number of tracks: {}".format(len(tracks_all_persons)))

        save_combined_tracks(tracks_all_persons,work_dirs=self.work_dirs,cam_id=cam_id)


    def calculate_bbox_vertex_velocities(self,track,n_tail=5):



        last_n_positons = track[-n_tail:]

        first_bbox_points_pos_track = np.array(last_n_positons[0]["bbox"])
        second_bbox_points_pos_track =  np.array(last_n_positons[-1]["bbox"])

        first_pos_frame_no_track = last_n_positons[0]["frame_no_cam"]
        second_pos_frame_no_track = last_n_positons[-1]["frame_no_cam"]

        #Don't know if this works like this
        velocities = (np.array(second_bbox_points_pos_track) - np.array(first_bbox_points_pos_track)) / (
                    second_pos_frame_no_track - first_pos_frame_no_track)

        return velocities




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


    def close_track_gaps_velocity(self, tracks):
        result_tracks = []
        for track in tracks:

            if len(track) == 1: continue


            filled_gap_track = []
            for track_pos_idx in range(len(track)-1):
                lower_frame_no = track[track_pos_idx]["frame_no_cam"]
                upper_frame_no = track[track_pos_idx + 1]["frame_no_cam"]
                frame_diff = upper_frame_no - lower_frame_no

                fill_up_upper = min(upper_frame_no,lower_frame_no + 30)

                if frame_diff > 1:

                    for insert_frame_no in range(lower_frame_no+1,fill_up_upper):
                        frame_no_delta = insert_frame_no - lower_frame_no
                        if len(track[:track_pos_idx]) < 2:
                            continue

                        bbox_velocities = self.calculate_bbox_vertex_velocities(track[:track_pos_idx])
                        bbox_delta = bbox_velocities*frame_no_delta

                        predicted_bbox = np.array(track[track_pos_idx]["bbox"]) + bbox_delta
                        filled_gap_track.append({"bbox" : tuple(predicted_bbox)
                                                    , "frame_no_cam" : insert_frame_no
                                                    , "filled_bbox" : True
                                                    ,"track_no" : track[track_pos_idx]["track_no"]})
                else:
                    filled_gap_track.append(track[track_pos_idx])

            filled_gap_track.append(track[-1])
            result_tracks.append(filled_gap_track)
        return result_tracks





    def calculate_path_distances_sum(self,path):
        result = 0
        for edge_idx in range(len(path)-1):

            cell1 = path[edge_idx]
            cell2 = path[edge_idx+1]
            edge = (cell1,cell2)
            if edge in self.edge_to_mean_transition_distance:
                result += self.edge_to_mean_transition_distance[edge]
                continue

            edge = (cell2, cell1)
            if edge in self.edge_to_mean_transition_distance:
                result += self.edge_to_mean_transition_distance[edge]
                continue

        return result

    def calculate_frame_no(self,time_in_minutes, fps=41):
        return fps * 60 * time_in_minutes

    def calculate_shortest_path(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        track1_pos = self.get_bbox_middle_pos(track1[-1]["bbox"])
        track1_nearest_node = self.get_nearest_graph_node(track1_pos)

        track2_pos = self.get_bbox_middle_pos(track2[0]["bbox"])
        track2_nearest_node = self.get_nearest_graph_node(track2_pos)


        shortest_path = nx.shortest_path(self.loga_graph, track1_nearest_node, track2_nearest_node, weight="weight")
        # shortest_path_distance = nx.shortest_path_length(self.loga_graph, combine_candidate_cell, combine_partner, weight="weight")

        return shortest_path


    def calculate_older_track_velocity(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        if len(track1) == 1:
            return self.all_persons_mean_velocity
            #Todo: Still not optimal because this mean velocity is created with all bboxes there should be one for every height

        return np.linalg.norm(self.calculate_track_velocity(track1)) / self.get_bbox_height(track1[-1]["bbox"])

    def calculate_time_difference(self,track1,track2):

        if not (track1[-1]["frame_no_cam"] < track2[0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        return track2[0]["frame_no_cam"] - track1[-1]["frame_no_cam"]

    def combine_via_shortest_paths(self, number_of_persons=105,time_forward_tracks=2):
        self.load_calculated_tracks_per_person()
        self.load_graph_grid()
        self.calculate_loga_graph()

        cells_to_path = {}

        tracks_all_persons = self.get_combined_fragments(self.person_id_to_tracks.values())

        pbar = tqdm(total=len(tracks_all_persons)-number_of_persons)

        while len(tracks_all_persons) > number_of_persons:
            pbar.update()
            #print("tracks_all_persons len: {}".format(len(tracks_all_persons)))
            combine_candidate = tracks_all_persons.pop(0)
            combine_candidate_frame_nos = [track_pos["frame_no_cam"] for track_pos in combine_candidate]


            current_track_idx_and_min = (None,sys.maxsize)
            for combine_partner_idx,combine_partner in enumerate(tracks_all_persons):

                combine_partner_frame_nos = [track_pos["frame_no_cam"] for track_pos in combine_partner]

                if not set(combine_candidate_frame_nos).isdisjoint(combine_partner_frame_nos):
                    continue

                shortest_path = self.calculate_shortest_path(combine_candidate,combine_partner)


                path_distances_sum = self.calculate_path_distances_sum(shortest_path)

                older_track_velocity = self.calculate_older_track_velocity(combine_candidate,combine_partner) #normed with bbox height in function
                time_difference_tracks = self.calculate_time_difference(combine_candidate,combine_partner)
                possible_distance = older_track_velocity * time_difference_tracks

                path_distance_possible_distance_difference = abs(possible_distance - path_distances_sum)
                current_track_idx_and_min = min((combine_partner_idx,path_distance_possible_distance_difference)
                                                ,current_track_idx_and_min,key=lambda x:x[1])


            combine_partner = tracks_all_persons.pop(current_track_idx_and_min[0])

            tracks_all_persons.append(self.combine_tracks(combine_candidate,combine_partner))
        self.save_combined_tracks(tracks_all_persons)


    def calculate_edge_to_mean_distance(self,edge_to_transition_distances):

        edge_to_mean_distance = {}
        for edge, transition_distances in self.edge_to_transition_distances.items():

            mean_distance = np.mean(np.array(transition_distances),axis=0)

            edge_to_mean_distance[edge] = mean_distance

        return edge_to_mean_distance

    def calculate_all_persons_mean_velocity(self,person_id_to_track_positions,n_tail=5):
        all_velocities = []
        for track_positions in person_id_to_track_positions.values():


            velocity_vector = self.calculate_track_velocity(track_positions)
            bbox_height = self.get_bbox_height(track_positions[-1]["bbox"])
            velocity_magnitude = np.linalg.norm(velocity_vector) / bbox_height
            all_velocities.append(velocity_magnitude)


        velocity_mean = np.mean(all_velocities,axis=0)

        return velocity_mean



    def calculate_graph_grid(self):
        self.load_groundtruth_csv()

        person_id_to_last_frame_no = {}
        self.person_id_to_track_positions = defaultdict(list)
        person_id_to_last_cell = defaultdict(lambda: -1)


        print("Starting to calculate graph grid")
        for index,cam_coord_row in tqdm(self.cam_coords.iterrows(),total=len(self.cam_coords)):



            person_id = int(cam_coord_row[self.person_identifier])
            frame_no_cam = cam_coord_row["frame_no_cam"]


            person_bbox = ( cam_coord_row["x_top_left_BB"]
                            , cam_coord_row["y_top_left_BB"]
                            , cam_coord_row["x_bottom_right_BB"]
                            , cam_coord_row["y_bottom_right_BB"] )

            person_bbox = constrain_bbox_to_img_dims(person_bbox)

            person_pos = self.get_bbox_middle_pos(person_bbox)

            self.person_id_to_track_positions[person_id].append({"bbox" : tuple(person_bbox) , "frame_no_cam" : frame_no_cam})

            cell = self.get_person_cell(person_pos)

            last_cell = person_id_to_last_cell[person_id]

            if last_cell == -1:
                #New appeared Person
                person_id_to_last_cell[person_id] = cell
                person_id_to_last_frame_no[person_id] = frame_no_cam
                self.person_appeared_cell_to_count[cell] += 1

            elif last_cell != cell:
                #Person has been seen before
                #Person went to a different cell
                self.node_to_nodes[last_cell].add(cell)
                self.node_to_nodes[cell].add(last_cell)

                #Calculating a possible distance that a person might have walked to get from last cell to cell
                time_difference = frame_no_cam - person_id_to_last_frame_no[person_id]
                person_track = self.person_id_to_track_positions[person_id]
                persons_velocity = self.calculate_track_velocity(person_track)
                estimated_distance = time_difference * np.linalg.norm(persons_velocity) / self.get_bbox_height(person_track[-1]["bbox"])

                self.edge_to_transition_distances[(last_cell,cell)].append(estimated_distance)
                self.edge_to_frequency[(last_cell,cell)] += 1

                #Update the cell in which the person currently resides
                person_id_to_last_cell[person_id] = cell
                person_id_to_last_frame_no[person_id] = frame_no_cam

        #Get and start and end positions of tracks
        start_position_cells = []
        end_position_cells = []
        for person_id, track_positions in self.person_id_to_track_positions.items():
            if len(track_positions) >= 2:
                start_pos = self.get_bbox_middle_pos(track_positions[0]["bbox"])
                end_pos = self.get_bbox_middle_pos(track_positions[-1]["bbox"])
                start_pos_cell = self.get_person_cell(start_pos)
                end_pos_cell = self.get_person_cell(end_pos)
                start_position_cells.append(start_pos_cell)
                end_position_cells.append(end_pos_cell)

        self.start_cell_frequency = dict(Counter(start_position_cells))
        self.end_cell_frequency = dict(Counter(end_position_cells))










if __name__ == "__main__":
    '''
    graph_grid = Graph_grid(work_dirs="/home/philipp/work_dirs"
                            ,cam_coords_path="/home/philipp/Downloads/Recording_12.07.2019/cam_2/coords_cam_2.csv"
                            ,cam_images_folder="/home/philipp/Downloads/Recording_12.07.2019/cam_2/"
                            ,track_results_folder="/home/philipp/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta1207"
                            ,track_result_cam_ids=range(6)
                            ,visualization_img_path="/home/philipp/Downloads/Recording_12.07.2019/cam_2/image_0_2.jpg")
    '''

    graph_grid = Graph_grid(work_dirs="/home/koehlp/Downloads/work_dirs"
                            , cam_coords_path="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train/cam_2/coords_cam_2.csv"
                            , cam_images_folder="/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/"
                            ,track_results_folder="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta1207_iosb"
                            , track_result_cam_ids=range(6)
                            , draw_videos_output_folder= "/net/merkur/storage/deeplearning/users/koehl/gta/drawn_videos_thresh_12.07"
                            , visualization_img_path="/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/image_0_2.jpg")



    '''
    graph_grid = Graph_grid("/home/koehlp/Downloads/work_dirs"
                            ,"/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train/cam_2/coords_cam_2.csv"
                            ,"/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_fpn_1x_strong_reid_Gta2207_iosb/track_results_cam_2.txt"
                            ,"/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/image_0_2.jpg"
                            , person_identifier="person_id")

    '''

    '''
    graph_grid = Graph_grid("/home/koehlp/Downloads/work_dirs"
                            ,"/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/coords_cam_2.csv"
                            ,"/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2"
                            ,"/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_fpn_1x_strong_reid_Gta1207_iosb_copy/track_results_cam_2.txt"
                            ,"/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/image_0_2.jpg")

    '''
    graph_grid.combine_via_min_predicted_pos_distance(cam_id=2)


