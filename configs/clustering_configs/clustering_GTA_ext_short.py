
import os



root = {

        "work_dirs" : "/media/philipp/philippkoehl_ssd/work_dirs"
        ,"train_track_results_folder" : "/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train"
        ,"test_track_results_folder" : "/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
        ,"train_dataset_folder" : "/media/philipp/philippkoehl_ssd/GTA_ext_short/train"
        ,"test_dataset_folder" : "/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
        , "cam_count" : 6
        , "person_identifier" : "person_id"
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "config_run_path" : "/media/philipp/philippkoehl_ssd/work_dirs/clustering/config_runs"




        ,"find_weights" : {
                            "dist_name_to_distance_weights" : {"are_tracks_disjunct": 1.0
                                                                            , "track_pred_pos_start_distance": 0.0
                                                                            , "track_cosine_distance": 0.0
                                                                            , "homography_match_score": 0.0}

                            ,"weight_search_configs" : [
                                                            { "dist_name" : "track_pred_pos_start_distance"
                                                                        ,"start_value" : 0
                                                                        ,"stop_value" : 10
                                                                        ,"steps" : 5},

                                                            {"dist_name": "homography_match_score"
                                                                        ,"start_value": 0
                                                                        ,"stop_value": 20
                                                                        ,"steps": 5}
                                                                    ]

                            ,"find_weights_cam_ids" : [1]

                            ,"run" : False


         }

        ,"cluster_from_weights" : {

            "best_weights_path" : "/media/philipp/philippkoehl_ssd/work_dirs/clustering/single_camera_best_weights/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train/best_weights.csv"
            , "default_weights" : {"are_tracks_disjunct": 1.0
                                    , "track_pred_pos_start_distance": 2
                                    , "track_cosine_distance": 0.0
                                    , "homography_match_score": 10}

            ,"cluster_cam_ids" : [1]
            ,"dataset_type" : "test"
            ,"run" : False
        },

        "feature_extractor_cfg_dict" : {"feature_extractor": {
                "reid_strong_baseline_config": "/home/philipp/Dokumente/masterarbeit/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
                "checkpoint_file": "/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
                "device": "cuda:0"}}



        ,"find_weights_multi_cam" : {
                #Default weights which will be partly overwritten during the search
                "dist_name_to_distance_weights" : {
                "are_tracks_cam_id_frame_disjunct": 1
                ,"are_tracks_frame_overlap_disjunct": 1
                ,"overlapping_match_score": 0.0
                ,"feature_mean_distance": 0.0
            },

            "weight_search_configs" : [
                                        { "dist_name" : "feature_mean_distance"
                                                    ,"start_value" : 0
                                                    ,"stop_value" : 10
                                                    ,"steps" : 5},

                                        {"dist_name": "overlapping_match_score"
                                                    ,"start_value": 0
                                                    ,"stop_value": 5
                                                    ,"steps": 5}
                                                ]

        ,"find_weights_cam_ids" : [1]


        ,"run" : True

        }

}