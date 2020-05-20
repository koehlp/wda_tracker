
import os


from configs.clustering_configs import multi_cam_clustering_GTA_ext_short_iosb

root = multi_cam_clustering_GTA_ext_short_iosb.root

root["find_weights"]["run"] = False
root["cluster_from_weights"]["run"] = True
root["cluster_from_weights"]["dataset_type"] = "test"

root["config_basename"] = os.path.basename(__file__).replace(".py","")

root["cluster_from_weights"]["best_weights_path"]  = "no_path"
root["cluster_from_weights"]["default_weights"] =  {

                "are_tracks_cam_id_frame_disjunct": 1
                ,"are_tracks_frame_overlap_disjunct": 0
                ,"overlapping_match_score": 2.3
                ,"feature_mean_distance": 6
                ,"track_pred_pos_start_distance" : 1

                }


root["find_weights"] = {
                "take_frames_per_cam" : 50000000,

                #Default weights which will be partly overwritten during the search
                "dist_name_to_distance_weights" : {
                    "are_tracks_cam_id_frame_disjunct": 1
                    ,"are_tracks_frame_overlap_disjunct": 0
                    ,"overlapping_match_score": 0.0
                    ,"feature_mean_distance": 0.0
                    , "track_pred_pos_start_distance" : 0.0
            },

            "weight_search_configs" : [
                                        { "dist_name" : "feature_mean_distance"
                                                    ,"start_value" : 4.0
                                                    ,"stop_value" : 12
                                                    ,"steps" : 5},

                                        {"dist_name": "overlapping_match_score"
                                                    ,"start_value": 1.1
                                                    ,"stop_value": 2.3
                                                    ,"steps": 3},

                                        {"dist_name": "track_pred_pos_start_distance"
                                                    ,"start_value": 0.05
                                                    ,"stop_value": 1
                                                    ,"steps": 3}
                                                ]


        ,"run" : False

        }