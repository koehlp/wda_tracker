
import os



root = {

        "work_dirs" : "/home/koehlp/Downloads/work_dirs"
        ,"train_track_results_folder": "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_train_iosb"
        ,"test_track_results_folder": "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
        , "train_dataset_folder": "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
        , "test_dataset_folder": "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
        , "cam_count" : 6
        , "person_identifier" : "person_id"
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "config_run_path" : "/home/koehlp/Downloads/work_dirs/clustering/config_runs"
        , "max_memory_in_gb" : 200
        , "min_free_memory_in_gb" : 75
        , "memory_watchdog_interval" : 0.5


        ,"feature_extractor_cfg_dict" : {"feature_extractor": {
            "reid_strong_baseline_config": "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
            "checkpoint_file": "/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
            "device": "cuda:0"
            ,"visible_device" : "2"}}

        ,"cluster_from_weights" : {


            "split_count" : 10

            ,"best_weights_path" : "no_path"
            , "default_weights" : {

                    "are_tracks_cam_id_frame_disjunct": 1
                    , "are_tracks_frame_overlap_disjunct": 1
                    , "overlapping_match_score": 0
                    , "feature_mean_distance": 4
                    , "track_pred_pos_start_distance": 0.44

                }

            ,"dataset_type" : "test"
            ,"run" : True
        },


        "find_weights" : {


                #Default weights which will be partly overwritten during the search
                "dist_name_to_distance_weights" : {
                    "are_tracks_cam_id_frame_disjunct": 1
                    ,"are_tracks_frame_overlap_disjunct": 1
                    ,"overlapping_match_score": 0.0
                    ,"feature_mean_distance": 0.0
                    , "track_pred_pos_start_distance" : 0.0
            },

            "weight_search_configs" : [
                                        { "dist_name" : "feature_mean_distance"
                                                    ,"start_value" : 4.0
                                                    ,"stop_value" : 12
                                                    ,"steps" : 5},


                                        {"dist_name": "track_pred_pos_start_distance"
                                                    ,"start_value": 0.05
                                                    ,"stop_value": 2
                                                    ,"steps": 6}
                                                ]


        ,"run" : False

        }

}