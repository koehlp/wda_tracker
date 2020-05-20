import os

root = {

            "dataset_folder" : "/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
          , "track_results_folder" : "/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
          , "cam_ids" : list(range(6))
          , "working_dir" : "/media/philipp/philippkoehl_ssd/work_dirs"
          , "n_parts" : 10
        , "config_basename" : os.path.basename(__file__).replace(".py","")
        , "evaluate_multi_cam" : False
        , "evaluate_single_cam" : True
        ,"config_run_path" : "/media/philipp/philippkoehl_ssd/work_dirs/evaluation/config_runs"

}