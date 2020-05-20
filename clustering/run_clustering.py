import argparse
import mmcv
from clustering.single_cam_clustering import Single_cam_clustering
from utils.logger import setup_logger
import os
import json


class Run_clustering:
    def __init__(self,args):
        self.cfg = mmcv.Config.fromfile(args.config).root

        config_run_path = os.path.join(self.cfg.config_run_path, self.cfg.config_basename)
        setattr(self.cfg, "config_run_path", config_run_path)

        os.makedirs(config_run_path,exist_ok=True)

        logger = setup_logger("clustering_logger", self.cfg.config_run_path, 0)

        logger.info(json.dumps(self.cfg, sort_keys=True, indent=4))



    def run(self):

        scc = Single_cam_clustering(work_dirs=self.cfg.work_dirs
                                    ,train_track_results_folder=self.cfg.train_track_results_folder
                                    ,test_track_results_folder=self.cfg.test_track_results_folder
                                    ,train_dataset_folder=self.cfg.train_dataset_folder
                                    ,test_dataset_folder=self.cfg.test_dataset_folder
                                    ,cam_count=self.cfg.cam_count
                                    ,person_identifier=self.cfg.person_identifier)

        if self.cfg.find_weights.run:
            scc.find_weights(cam_ids=self.cfg.find_weights.find_weights_cam_ids
                             , weight_search_configs=self.cfg.find_weights.weight_search_configs
                             , dist_name_to_distance_weights=self.cfg.find_weights.dist_name_to_distance_weights)

        if self.cfg.cluster_from_weights.run:
            scc.cluster_from_weights(best_weights_path=self.cfg.cluster_from_weights.best_weights_path
                                     ,default_weights=self.cfg.cluster_from_weights.default_weights
                                     ,cam_ids=self.cfg.cluster_from_weights.cluster_cam_ids
                                     ,dataset_type=self.cfg.cluster_from_weights.dataset_type)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    return parser.parse_args()



if __name__=="__main__":


    args = parse_args()

    run_clustering = Run_clustering(args)


    run_clustering.run()