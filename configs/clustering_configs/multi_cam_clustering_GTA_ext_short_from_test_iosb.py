
import os


from configs.clustering_configs import multi_cam_clustering_GTA_ext_short_iosb

root = multi_cam_clustering_GTA_ext_short_iosb.root

root["find_weights"]["run"] = False
root["cluster_from_weights"]["run"] = True
root["cluster_from_weights"]["dataset_type"] = "test"

root["config_basename"] = os.path.basename(__file__).replace(".py","")