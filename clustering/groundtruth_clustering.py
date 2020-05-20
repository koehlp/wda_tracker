from feature_extractors.reid_strong_extractor import Reid_strong_extractor

import mmcv
import numpy as np
import cv2
from utilities.pandas_loader import load_csv
from utilities.helper import takespread
import pickle
import os
import sys
import glob
import multiprocessing as mp
from math import ceil

from collections import Counter

from clustering.finch_cluster.python.finch import FINCH

from tqdm import tqdm
import pandas as pd

from collections import defaultdict

from utilities.helper import many_xyxy2xywh,constrain_bbox_to_img_dims



class Reid_clustering:

    def __init__(self,cfg):

        self.feature_extractor = Reid_strong_extractor(cfg)
        self.model_name = os.path.basename(cfg["feature_extractor"]["checkpoint_file"])

    def get_features(self, bboxes_xtylwh, ori_img,debug=False):
        im_crops = []
        for bbox in bboxes_xtylwh:
            x,y,w,h = bbox
            im = ori_img[y:y+h, x:x+w]

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

    def get_feature_path(self,work_dirs,cam_id,dataset_name):
        feature_extractor_path = os.path.join(work_dirs, "feature_extractor/")
        features_path = os.path.join(feature_extractor_path, "features/")
        features_model_path = os.path.join(features_path, self.model_name.replace(".pth", "") + "_" + dataset_name)
        features_model_cam_path = os.path.join(features_model_path,"cam_{}".format(cam_id))

        return features_model_cam_path


    def get_track_path(self, work_dirs, cam_id):
        pass

    def generate_features_one_cam(self, cam_coords: pd.DataFrame, cam_folder, work_dirs ,dataset_name):
        '''

        :param cam_coords:
        :param cam_folder:
        :param work_dirs:
        :return: Nothing. Writes a dict to a image_{frame_no_cam}_{cam_id}.pkl file with the format:
        { person_id : feature }
        '''

        # In old csv files it was called ped_id than another id was used: person_id
        if "ped_id" in cam_coords.columns:
            person_id_name = "ped_id"
        else:
            person_id_name = "person_id"

        cam_coords = cam_coords.groupby(["frame_no_gta", person_id_name], as_index=False).mean()
        cam_coords = cam_coords.astype({"frame_no_gta": "int32", person_id_name: "int32"})

        frame_no_groups = cam_coords.groupby('frame_no_gta', as_index=False).mean()
        frame_no_groups = frame_no_groups.astype({"frame_no_gta": "int32"})

        frame_nos_gta = frame_no_groups["frame_no_gta"].tolist()

        cam_folder_name = os.path.basename(os.path.normpath(cam_folder))
        cam_id = int(cam_folder_name.split("_")[1])

        features_model_path = self.get_feature_path(work_dirs,cam_id,dataset_name)
        os.makedirs(features_model_path,exist_ok=True)

        feature_path_regex = os.path.join(features_model_path, "*.pkl")

        feature_count = len(glob.glob(feature_path_regex))

        if feature_count > 0:
            return

        for frame_no_gta in tqdm(frame_nos_gta):
            current_frame = cam_coords[cam_coords["frame_no_gta"] == int(frame_no_gta)]

            person_ids = current_frame[person_id_name].tolist()
            xyxy_bboxes = zip(current_frame["x_top_left_BB"]
                                    ,current_frame["y_top_left_BB"]
                                    ,current_frame["x_bottom_right_BB"]
                                    ,current_frame["y_bottom_right_BB"])

            xyxy_bboxes = list(map(constrain_bbox_to_img_dims, xyxy_bboxes))
            xyxy_bboxes = list(map(lambda bbox: np.array(bbox,dtype=np.int32), xyxy_bboxes))

            tlwh_bboxes = many_xyxy2xywh(xyxy_bboxes)


            frame_no_cam = int(current_frame.iloc[0]["frame_no_cam"])
            image_name = "image_{}_{}.jpg".format(frame_no_cam,cam_id)
            image_path = os.path.join(cam_folder,image_name)
            image = mmcv.imread(image_path)
            features_one_img = self.get_features(tlwh_bboxes, image)

            person_id_to_feature = {}
            for person_id, feature in zip(np.array(person_ids, dtype=np.int32),features_one_img):
                person_id_to_feature[person_id] = feature

            feature_pickle_name = os.path.join(features_model_path, image_name.replace(".jpg",""))
            with open(feature_pickle_name + ".pkl", 'wb') as handle:
                pickle.dump(person_id_to_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def calculate_all_tracks_feature_means(self,tracks,work_dirs,cam_id,dataset_name):
        tracks_with_feature_mean = []
        for track in tracks:
            feature_mean = self.calculate_track_feature_mean(track,cam_id,work_dirs,dataset_name)
            track["feature_mean"] = feature_mean
            tracks_with_feature_mean.append(track)

        return tracks



    def calculate_track_feature_mean(self,track,cam_id,work_dirs,dataset_name):

        track_list = track["track"]
        person_id = track["person_id"]
        track_features = []
        for track_entry in track_list:

            frame_no_cam = track_entry["frame_no_cam"]

            features_model_path = self.get_feature_path(work_dirs,cam_id,dataset_name)
            feature_pickle_name = "image_{}_{}.pkl".format(frame_no_cam, cam_id)

            feature_pickle_name = os.path.join(features_model_path, feature_pickle_name)

            with open(feature_pickle_name, 'rb') as handle:
                feature_dict = pickle.load(handle)

                track_features.append(feature_dict[person_id])


        track_features = np.array(track_features)
        track_mean = np.mean(track_features,axis=0)
        return track_mean



    def calculate_all_tracks_feature_samples(self,tracks,work_dirs,cam_id,dataset_name):
        tracks_with_feature_mean = []
        for track in tracks:
            feature_samples = self.calculate_track_feature_samples(track,cam_id,work_dirs,dataset_name)
            track["feature_samples"] = feature_samples
            tracks_with_feature_mean.append(track)

        return tracks

    def calculate_track_feature_samples(self, track, cam_id, work_dirs, dataset_name,num_samples=10):

        track_list = track["track"]
        person_id = track["person_id"]
        track_features = []
        for track_entry in track_list:
            frame_no_cam = track_entry["frame_no_cam"]

            features_model_path = self.get_feature_path(work_dirs, cam_id, dataset_name)
            feature_pickle_name = "image_{}_{}.pkl".format(frame_no_cam, cam_id)

            feature_pickle_name = os.path.join(features_model_path, feature_pickle_name)

            with open(feature_pickle_name, 'rb') as handle:
                feature_dict = pickle.load(handle)

                track_features.append(feature_dict[person_id])

        track_features = takespread(track_features,num_samples)

        if len(track_features) == 0:
            print("len features zero of track")

        return track_features

    def get_tracks_one_cam(self,cam_coords: pd.DataFrame,cam_id):
        '''


        :param cam_coords:
        :return: A list of tracks. One track looks like a dict:

        {
            "cam_id" : cam_id,
            "person_id" : person_id,
            "track" : [
                            { "frame_no_gta" : int(frame_no_gta)
                                , "bbox" : bbox #xyxy
                               , "frame_no_cam" : int(frame_no_cam) }
                            } ,
                             { "frame_no_gta" : int(frame_no_gta)
                                , "bbox" : bbox #xyxy
                               , "frame_no_cam" : int(frame_no_cam) }
                            }
                    ...
                    ]

        }

        '''

        # In old csv files it was called ped_id than another id was used: person_id
        if "ped_id" in cam_coords.columns:
            person_id_name = "ped_id"
        else:
            person_id_name = "person_id"

        cam_coords = cam_coords.groupby(["frame_no_gta", person_id_name], as_index=False).mean()
        cam_coords = cam_coords.astype({"frame_no_gta": "int32", person_id_name : "int32"})

        frame_no_groups = cam_coords.groupby('frame_no_gta',as_index=False).mean()
        frame_no_groups = frame_no_groups.astype({"frame_no_gta": "int32"})

        frame_nos_gta = frame_no_groups["frame_no_gta"].tolist()

        finished_tracks = []
        current_active_tracks = defaultdict(list) #person_id -> [frame_no_gta..]
        for frame_no_gta in frame_nos_gta:
            current_frame = cam_coords[cam_coords["frame_no_gta"] == int(frame_no_gta)]

            person_ids_in_frame = set(current_frame[person_id_name].tolist())
            person_ids_active_tracks = set(current_active_tracks.keys())

            #Get all person_ids that are in active tracks but are not in the current frame
            #That means this person was in the previous frame but now disappeared
            track_person_ids_not_in_frame = person_ids_active_tracks - person_ids_in_frame

            #Remove theses tracks whose persons are not beeing seen anymore from the active tracks and append it to the finished_tracks
            for person_id in track_person_ids_not_in_frame:
                finished_tracks.append({ "cam_id" : cam_id, "person_id" : person_id , "track" : current_active_tracks.pop(person_id, None) })


            def add_to_active_track(row):
                frame_no_gta = row["frame_no_gta"]
                person_id = row[person_id_name]
                bbox = [row["x_top_left_BB"]
                        ,row["y_top_left_BB"]
                        ,row["x_bottom_right_BB"]
                        ,row["y_bottom_right_BB"]]

                frame_no_cam = row["frame_no_cam"]

                current_active_tracks[int(person_id)].append({ "frame_no_gta" : int(frame_no_gta)
                                                                 , "bbox" : bbox
                                                               , "frame_no_cam" : int(frame_no_cam) })

            #Add all person_ids in the current frame to the active tracks
            #This also may create new tracks if these person_ids have not been seen before
            current_frame.apply(lambda row: add_to_active_track(row), axis=1)


        #After all frames from this camera have been checked. The current active tracks have to be added to the finised tracks
        for person_id, track in current_active_tracks.items():
            finished_tracks.append({"cam_id" : cam_id , "person_id" : person_id, "track": track})


        return finished_tracks



    def cluster_samples_via_finch(self,tracks):

        corresponding_track_ids = []
        feature_samples_tracks = []
        for track_id,track in enumerate(tracks):
            feature_samples = track["feature_samples"]
            if len(feature_samples) == 0:
                feature_samples = [track["feature_mean"]]
            feature_samples_tracks.extend(feature_samples)
            corresponding_track_ids.extend([track_id]*len(feature_samples))

        feature_samples_tracks = np.array(feature_samples_tracks)

        c, num_clust, req_c = FINCH(feature_samples_tracks,verbose=True)

        partition_1_clusters = c[:,0]


        track_id_to_sample_cluster_ids = defaultdict(list)
        for track_id, cluster_idx in zip(corresponding_track_ids,partition_1_clusters):

            track_id_to_sample_cluster_ids[track_id].append(cluster_idx)

        tracks_with_clusters = []
        for track_id, cluster_idxs in track_id_to_sample_cluster_ids.items():

            most_common_idx, num_most_common = Counter(cluster_idxs).most_common(1)[0]
            track = tracks[track_id]
            track["cluster_idx"] = most_common_idx
            tracks_with_clusters.append(track)

        return tracks_with_clusters

    def cluster_samples_via_kmeans(self, tracks):
        from sklearn.cluster import KMeans
        corresponding_track_ids = []
        feature_samples_tracks = []
        for track_id,track in enumerate(tracks):
            feature_samples = track["feature_samples"]
            if len(feature_samples) == 0:
                print("Feature samples len zero.")
            feature_samples_tracks.extend(feature_samples)
            corresponding_track_ids.extend([track_id]*len(feature_samples))


        feature_mean_tracks = np.array(feature_samples_tracks)

        clustering = KMeans(n_clusters=105,verbose=1).fit(feature_mean_tracks)



        track_id_to_sample_cluster_ids = defaultdict(list)
        for track_id, cluster_idx in zip(corresponding_track_ids,clustering.labels_):

            track_id_to_sample_cluster_ids[track_id].append(cluster_idx)

        tracks_with_clusters = []
        for track_id, cluster_idxs in track_id_to_sample_cluster_ids.items():

            most_common_idx, num_most_common = Counter(cluster_idxs).most_common(1)[0]
            track = tracks[track_id]
            track["cluster_idx"] = most_common_idx
            tracks_with_clusters.append(track)

        return tracks_with_clusters


    def cluster_via_finch(self,tracks):

        feature_mean_tracks = []
        for track in tracks:
            feature_mean_tracks.append(track["feature_mean"])

        feature_mean_tracks = np.array(feature_mean_tracks)


        c, num_clust, req_c = FINCH(feature_mean_tracks,verbose=True)

        partition_1_clusters = c[:,0]

        tracks_with_cluster_idx = []
        for track, cluster_idx in zip(tracks,partition_1_clusters):

            track["cluster_idx"] = cluster_idx
            tracks_with_cluster_idx.append(track)

        return tracks_with_cluster_idx

    def cluster_via_birch(self,tracks):
        feature_mean_tracks = []
        for track in tracks:
            feature_mean_tracks.append(track["feature_mean"])

        feature_mean_tracks = np.array(feature_mean_tracks)


        from sklearn.cluster import Birch
        brc = Birch(branching_factor=50, n_clusters=106, threshold=0.5,
        compute_labels = True)
        birch_labels = brc.fit(feature_mean_tracks).labels_

        tracks_with_cluster_idx = []
        for track, cluster_idx in zip(tracks, birch_labels):
            track["cluster_idx"] = cluster_idx
            tracks_with_cluster_idx.append(track)

        return tracks_with_cluster_idx


    def cluster_via_kmeans(self, tracks):
        from sklearn.cluster import KMeans
        feature_mean_tracks = []
        for track in tracks:
            feature_mean_tracks.append(track["feature_mean"])

        feature_mean_tracks = np.array(feature_mean_tracks)

        kmeans = KMeans(max_iter=300,n_clusters=2046, random_state=0,verbose=1).fit(feature_mean_tracks)



        tracks_with_cluster_idx = []
        for track, cluster_idx in zip(tracks, kmeans.labels_):
            track["cluster_idx"] = cluster_idx
            tracks_with_cluster_idx.append(track)

        return tracks_with_cluster_idx


    def cluster_via_agglomerative(self,tracks):
        from sklearn.cluster import AgglomerativeClustering
        feature_mean_tracks = []
        for track in tracks:
            feature_mean_tracks.append(track["feature_mean"])

        feature_mean_tracks = np.array(feature_mean_tracks)

        clustering = AgglomerativeClustering(n_clusters=245).fit(feature_mean_tracks)


        tracks_with_cluster_idx = []
        for track, cluster_idx in zip(tracks, clustering.labels_):
            track["cluster_idx"] = cluster_idx
            tracks_with_cluster_idx.append(track)

        return tracks_with_cluster_idx


    def show_purity_variance_diag(self,cluster_purities,cluster_variances):

        import numpy as np
        import matplotlib.pyplot as plt



        # Plot
        plt.scatter(cluster_purities.values(), cluster_variances.values())
        plt.title('purity variance scatter')
        plt.xlabel('cluster_purity')
        plt.ylabel('cluster_variance')
        #plt.show()
        plt.savefig(os.path.join(self.work_dirs,"tracker/",'purity_variance_scatter.png'))




    def print_cluster_histogram(self,tracks,output=sys.stdout):
        cluster_idx_to_person_id = defaultdict(list)

        cluster_variances = self.calc_cluster_variances(tracks)
        cluster_purities = self.calc_purity_per_cluster(tracks)
        self.show_purity_variance_diag(cluster_purities,cluster_variances)

        cluster_overlapping_tracks = self.calc_overlapping_tracks_in_clusters(tracks,output=output)

        for track_id,track in enumerate(tracks):
            cluster_idx_to_person_id[track["cluster_idx"]].append((track["person_id"],track["cam_id"]))

        cluster_idx_to_person_id_tuples = cluster_idx_to_person_id.items()

        cluster_idx_to_person_id_tuples = sorted(cluster_idx_to_person_id_tuples,key=lambda x:x[0])

        for cluster_idx, person_ids in cluster_idx_to_person_id_tuples:
            print("", file=output)
            print("cluster_idx: {} variance mean: {} overlap tracks: {} purity: {}".format(cluster_idx
                                                                                    ,cluster_variances[cluster_idx]
                                                                                    ,cluster_overlapping_tracks[cluster_idx]
                                                                                   ,cluster_purities[cluster_idx]),file=output)
            print("person_ids: {}".format(str(sorted(person_ids))),file=output)

        return cluster_idx_to_person_id


    def print_person_count(self,tracks,output=sys.stdout):
        person_ids = set()
        for track in tracks:
            person_ids.add(track["person_id"])
        print("person id count: {}".format(len(person_ids)),file=output)

    def print_track_count(self,tracks,output=sys.stdout):
        print("track count: {}".format(len(tracks)),file=output)

    def get_tracks_in_clusters(self,tracks):
        clusters = defaultdict(list)
        for track in tracks:
            clusters[track["cluster_idx"]].append(track)

        return clusters

    def calc_cluster_variances(self,tracks):
        clusters = self.get_tracks_in_clusters(tracks)
        cluster_variances = dict()


        for cluster_idx, tracks_in_cluster in clusters.items():
            feature_means = []
            for track in tracks_in_cluster:
                feature_means.append(track["feature_mean"])

            feature_var_elementwise = np.var(np.array(feature_means),axis=0)

            cluster_variance_mean = np.mean(np.array([feature_var_elementwise]), axis=1)
            cluster_variances[cluster_idx] = cluster_variance_mean


        return cluster_variances


    def calc_purity_per_cluster(self,tracks):

        clusters = self.get_tracks_in_clusters(tracks)
        cluster_purities = dict()
        for cluster_idx, tracks_in_cluster in clusters.items():
            person_ids_in_cluster = []
            for track in tracks_in_cluster:
                person_ids_in_cluster.append(track["person_id"])

            cluster_purities[cluster_idx] = max(Counter(person_ids_in_cluster).values()) / len(person_ids_in_cluster)

        return cluster_purities


    def calc_overlapping_tracks_in_clusters(self, tracks, output=sys.stdout):

        clusters = self.get_tracks_in_clusters(tracks)
        overall_overlapping_tracks = 0
        cluster_overlapping_tracks = defaultdict(lambda: 0)


        for cluster_idx, tracks_in_cluster in clusters.items():
            frame_no_gta_tracks = []


            #Add all frame_no_gta of the tracks in the cluster to a list frame_no_gta_tracks
            for track in tracks_in_cluster:
                for track_point in track["track"]:
                    frame_no_gta_tracks.append(track_point["frame_no_gta"])

            frame_no_gta_tracks = sorted(frame_no_gta_tracks)

            for track in tracks_in_cluster:
                #Remove every frame_no_gta from the current track from the list frame_no_gta_tracks
                for track_point in track["track"]:
                    frame_no_gta_tracks.remove(track_point["frame_no_gta"])

                #Check if one frame_no_gta of the current track is still in the list
                #If yes that would mean another track had the same frame_no_gta. So it is an overlapping track.
                for track_point in track["track"]:
                    if track_point["frame_no_gta"] in frame_no_gta_tracks:
                        overall_overlapping_tracks += 1
                        cluster_overlapping_tracks[cluster_idx] += 1
                        break

                #Add back all the previously removed frame no gtas from to the frame_no_gta_tracks List
                for track_point in track["track"]:
                    frame_no_gta_tracks.append(track_point["frame_no_gta"])



        print("Overlapping tracks: {}".format(overall_overlapping_tracks),file=output)

        return cluster_overlapping_tracks



    def print_cluster_count(self, tracks, output=sys.stdout):

        clusters = set()
        for track in tracks:
            clusters.add(track["cluster_idx"])

        print("cluster count: {}".format(len(clusters)), file=output)


    def create_clustering_all_cams(self,cam_count,base_dataset_folder,work_dirs,dataset_name="",output=sys.stdout):

        self.work_dirs = work_dirs

        tracker_path = os.path.join(work_dirs, "tracker/")
        tracks_all_cams_path = os.path.join(tracker_path, "tracks_all_cams" + dataset_name + ".pkl" )

        tracks_all_cams = []
        if os.path.exists(tracks_all_cams_path):
            with open(tracks_all_cams_path, 'rb') as handle:
                tracks_all_cams = pickle.load(handle)

        else:

            for cam_id in range(cam_count):
                cam_folder_path = os.path.join(base_dataset_folder, "cam_{}".format(cam_id))
                cam_coords_path = os.path.join(cam_folder_path, "coords_cam_{}.csv".format(cam_id))
                coords = load_csv(work_dirs, cam_coords_path)

                self.generate_features_one_cam(coords, cam_folder_path, work_dirs, dataset_name)

                tracks = self.get_tracks_one_cam(coords,cam_id)

                tracks = self.calculate_all_tracks_feature_means(tracks, work_dirs, cam_id,dataset_name)

                tracks = self.calculate_all_tracks_feature_samples(tracks, work_dirs, cam_id, dataset_name)

                tracks_all_cams.extend(tracks)



            with open(tracks_all_cams_path, 'wb') as handle:
                pickle.dump(tracks_all_cams, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #tracks_all_cams = self.cluster_samples_via_finch(tracks_all_cams)
        #tracks_all_cams = self.cluster_samples_via_kmeans(tracks_all_cams)
        #tracks_all_cams = self.cluster_samples_via_finch(tracks_all_cams)
        tracks_all_cams = self.cluster_via_finch(tracks_all_cams)
        #tracks_all_cams = self.cluster_via_optics(tracks_all_cams)
        #tracks_all_cams = self.cluster_via_kmeans(tracks_all_cams)
        #tracks_all_cams = self.cluster_via_birch(tracks_all_cams)
        #tracks_all_cams = self.cluster_via_agglomerative(tracks_all_cams)


        print("purity: {}".format(self.calculate_purity(tracks_all_cams)),file=output)
        self.print_cluster_histogram(tracks_all_cams,output=output)
        self.print_person_count(tracks_all_cams,output=output)
        self.print_track_count(tracks_all_cams,output=output)


    def calculate_purity(self,tracks):
        cluster_idx_to_person_id = defaultdict(list)
        for track_id,track in enumerate(tracks):
            cluster_idx_to_person_id[track["cluster_idx"]].append(track["person_id"])

        max_count_sum = 0
        track_count = len(tracks)
        for cluster_idx, person_ids in cluster_idx_to_person_id.items():
            max_person_id_count = max(Counter(person_ids).values())
            max_count_sum += max_person_id_count


        purity = float(max_count_sum) / float(track_count)

        return purity



if __name__ == "__main__":

    '''
    cfg_dict = {"feature_extractor": {
        "reid_strong_baseline_config": "/home/philipp/Dokumente/masterarbeit/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
        "checkpoint_file": "/home/philipp/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
        "device": "cuda:0"}}

    cfg = mmcv.Config(cfg_dict)

    reid_clustering = Reid_clustering(cfg)


    dataset_base_folder = "/home/philipp/Downloads/Recording_12.07.2019/"
    work_dirs = "/home/philipp/work_dirs/"

    tracker_path = os.path.join(work_dirs,"tracker/clustering_result.txt")

    dataset_name = ""
    
    '''


    '''
    cfg_dict = {"feature_extractor": {
        "reid_strong_baseline_config": "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
        "checkpoint_file": "/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
        "device": "cuda:5"}}

    cfg = mmcv.Config(cfg_dict)


    reid_clustering = Reid_clustering(cfg)

    dataset_base_folder = "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/"
    work_dirs = "/home/koehlp/Downloads/work_dirs"
    dataset_name = ""

    tracker_path = os.path.join(work_dirs, "tracker/clustering_result.txt")
    '''



    cfg_dict = {"feature_extractor": {
        "reid_strong_baseline_config": "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
        "checkpoint_file": "/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
        "device": "cuda:5"}}

    cfg = mmcv.Config(cfg_dict)

    dataset_base_folder = "/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17"
    work_dirs = "/home/koehlp/Downloads/work_dirs"
    dataset_name = "GTA_12.07.2019_17"

    reid_clustering = Reid_clustering(cfg)



    tracker_path = os.path.join(work_dirs, "tracker/clustering_result_"+ dataset_name + ".txt")

    with open(tracker_path,"w") as file:

        reid_clustering.create_clustering_all_cams(cam_count=6
                                                   ,base_dataset_folder=dataset_base_folder
                                                   ,work_dirs=work_dirs
                                                    ,dataset_name=dataset_name
                                                   ,output=file)









