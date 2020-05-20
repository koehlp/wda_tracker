
import os

root = {

        "work_dirs" : "/home/koehlp/Downloads/work_dirs"
        ,"train_track_results_folder" : "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_train_iosb"
        ,"test_track_results_folder" : "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
        ,"train_dataset_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
        ,"test_dataset_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
        , "cam_count" : 6
        , "person_identifier" : "person_id"
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "config_run_path" : "/home/koehlp/Downloads/work_dirs/clustering/config_runs"


    , "find_weights": {
        "dist_name_to_distance_weights": {"are_tracks_disjunct": 1.0
            , "track_pred_pos_start_distance": 0.0
            , "track_cosine_distance": 0.0
            , "homography_match_score": 0.0}

        , "weight_search_configs": [
            {"dist_name": "track_pred_pos_start_distance"
                , "start_value": 0
                , "stop_value": 10
                , "steps": 10},

            {"dist_name": "homography_match_score"
                , "start_value": 0
                , "stop_value": 20
                , "steps": 5}
        ]

        , "find_weights_cam_ids": list(range(6))

        , "run": False

    }

    , "cluster_from_weights": {

        "best_weights_path": "/home/koehlp/Downloads/work_dirs/clustering/single_camera_best_weights/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train_iosb/best_weights.csv"
        , "default_weights": {"are_tracks_disjunct": 1.0
            , "track_pred_pos_start_distance": 2
            , "track_cosine_distance": 0.0
            , "homography_match_score": 10}

        , "cluster_cam_ids": list(range(6))
        , "dataset_type": "test"
        , "run": True
    }

}