
import os



root = {

        "work_dirs" : "/home/koehlp/Downloads/work_dirs"
        ,"train_track_results_folder" : "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_train_iosb"
        ,"test_track_results_folder" : "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test_iosb"
        ,"train_dataset_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/train"
        ,"test_dataset_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/test"
        , "cam_count" : 6
        , "person_identifier" : "person_id"
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "config_run_path" : "/home/koehlp/Downloads/work_dirs/clustering/config_runs"
        , "max_memory_in_gb": 200
        , "min_free_memory_in_gb": 75
        , "memory_watchdog_interval": 0.5




            ,"feature_extractor_cfg_dict": {

                "feature_extractor_name" : "abd_net_extractor"

                ,"reid_strong_extractor": {
                    "reid_strong_baseline_config": "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
                    "checkpoint_file": "/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
                    "device": "cuda:0"
                    ,"visible_device" : "1"}

                ,"abd_net_extractor" : ['-s'
                                        , 'market1501'
                                        , '-t'
                                        , 'market1501'
                                        , '--height'
                                        , '256'
                                        , '--width'
                                        , '128'
                                        , '--test-batch-size'
                                        , '100'
                                        , '--evaluate'
                                        , '-a'
                                        , 'resnet50'
                                        , '--load-weights',
                                                    '/net/merkur/storage/deeplearning/users/koehl/reid/market_checkpoint_best.pth.tar',
                                                    '--save-dir',
                                                    '/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/abd-net/log/eval-resnet50',
                                                    '--gpu-devices', '0'
                                        , '--root',
                                                    '/media/philipp/philippkoehl_ssd/work_dirs/datasets'

                                        ]


        }

        ,"cluster_from_weights" : {


            "split_count" : 1

            ,"best_weights_path" : "no_path"
            , "default_weights" : {

                "are_tracks_cam_id_frame_disjunct": 1
                ,"are_tracks_frame_overlap_disjunct": 1
                ,"overlapping_match_score": 1.8
                ,"feature_mean_distance":4.5
                ,"track_pred_pos_start_distance" : 0.325

                }

            ,"dataset_type" : "test"
            ,"run" : True
        },


        "find_weights" : {
                "take_frames_per_cam" : 10000,

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
                                                    ,"stop_value" : 5
                                                    ,"steps" : 3},

                                        {"dist_name": "overlapping_match_score"
                                                    ,"start_value": 1.8
                                                    ,"stop_value": 2.3
                                                    ,"steps": 3},

                                        {"dist_name": "track_pred_pos_start_distance"
                                                    ,"start_value": 0.1
                                                    ,"stop_value": 1
                                                    ,"steps": 5}
                                                ]


        ,"run" : False

        }

}