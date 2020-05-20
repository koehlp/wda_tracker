import os

root = {

            "dataset_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
          , "track_results_folder" : "/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
          , "cam_ids" : list(range(6))
          , "working_dir" : "/home/koehlp/Downloads/work_dirs"
          , "n_parts" : 10
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "evaluate_multi_cam" : True
        , "evaluate_single_cam" : False
        ,"config_run_path" : "/home/koehlp/Downloads/work_dirs/evaluation/config_runs"

}