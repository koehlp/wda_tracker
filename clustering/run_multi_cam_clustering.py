
import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6

import sys

def append_to_pythonpath(paths):
    for path in paths:
        sys.path.append(path)


append_to_pythonpath(['/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/clustering',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/trackers/iou_tracker',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/evaluation/py_motmetrics',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python37.zip',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/lib-dynload',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/site-packages',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/clustering',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python37.zip',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/lib-dynload',
                      '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/site-packages',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/ABD_Net'])

import argparse
import mmcv
from utils.logger import setup_logger
import json

from clustering.multi_cam_clustering import splitted_clustering_from_weights, find_clustering_weights


class Run_clustering:
    def __init__(self,args):

        self.cfg = mmcv.Config.fromfile(args.config).root

        self.cfg.config_basename = os.path.basename(args.config).replace(".py","")

        config_run_path = os.path.join(self.cfg.config_run_path, self.cfg.config_basename)
        setattr(self.cfg, "config_run_path", config_run_path)

        os.makedirs(config_run_path,exist_ok=True)

        logger = setup_logger("clustering_logger", self.cfg.config_run_path, 0)

        logger.info(json.dumps(self.cfg, sort_keys=True, indent=4))





    def run(self):

        feature_ext_cfg = mmcv.Config(self.cfg.feature_extractor_cfg_dict)


        if self.cfg.find_weights.run:
            find_clustering_weights(test_track_results_folder=self.cfg.test_track_results_folder
                                             , train_track_results_folder=self.cfg.train_track_results_folder
                                             , work_dirs=self.cfg.work_dirs
                                             , test_dataset_folder=self.cfg.test_dataset_folder
                                             , train_dataset_folder=self.cfg.train_dataset_folder
                                             , feature_extractor_cfg=feature_ext_cfg
                                             , cam_count=self.cfg.cam_count
                                             , take_frames_per_cam=self.cfg.find_weights.take_frames_per_cam
                                             , weight_search_configs=self.cfg.find_weights.weight_search_configs
                                             , dist_name_to_distance_weights=self.cfg.find_weights.dist_name_to_distance_weights
                                             , config_basename=self.cfg.config_basename
                                             , person_identifier=self.cfg.person_identifier
                                             )


        if self.cfg.cluster_from_weights.run:


            splitted_clustering_from_weights(test_track_results_folder=self.cfg.test_track_results_folder
                                             , train_track_results_folder=self.cfg.train_track_results_folder
                                             , work_dirs=self.cfg.work_dirs
                                             , test_dataset_folder=self.cfg.test_dataset_folder
                                             , train_dataset_folder=self.cfg.train_dataset_folder
                                             , feature_extractor_cfg=feature_ext_cfg
                                             , cam_count=self.cfg.cam_count
                                             , best_weights_path=self.cfg.cluster_from_weights.best_weights_path
                                             , default_weights=self.cfg.cluster_from_weights.default_weights
                                             , config_basename=self.cfg.config_basename
                                             , person_identifier=self.cfg.person_identifier
                                             , n_split_parts=self.cfg.cluster_from_weights.split_count
                                             )






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    return parser.parse_args()



if __name__=="__main__":


    args = parse_args()

    run_clustering = Run_clustering(args)


    run_clustering.run()