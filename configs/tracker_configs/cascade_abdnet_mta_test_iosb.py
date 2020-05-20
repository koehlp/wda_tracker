#Faster R-CNN will be used as detector and the ckpt.t7 file from the deep_sort_pytorch website will be used as feature extractor
import os


root = {

    "general" : {

        "display_viewer" : False,
        #The visible GPUS will be restricted to the numbers listed here. The pytorch (cuda:0) numeration will start at 0
        #This is a trick to get everything onto the wanted gpus because just setting cuda:4 in the function calls will
        #not work for mmdetection. There will still be things on gpu cuda:0.
        "cuda_visible_devices" : "6",
        #These paths will be appended to the PYTHONPATH. This needs to be done because the subprojects like mmdetection import from their relative
        #root path.
        "source_root_paths" : ['/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/evaluation/py_motmetrics', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/evaluation/object_det_metrics/lib', '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/ABD_Net',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python37.zip',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/lib-dynload',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/site-packages',
                                '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/clustering',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python37.zip',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/lib-dynload',
                                '/home/koehlp/anaconda3/envs/reid-baseline1/lib/python3.7/site-packages'],

        "config_run_path" : "/home/koehlp/Downloads/work_dirs/config_runs",
        "config_basename" : os.path.basename(__file__).replace(".py",""),
        "work_dirs" : "/home/koehlp/Downloads/work_dirs/",
        "save_track_results" : True

    },

    "data" : {

        "module_name" : "datasets.gta_dataset",
        "function_name" : "get_gta_cam_iterators",
        "selection_interval" : [0,50000000],
        "extension" : ".jpg",

        "source" : {
            #To increase the speed while developing an specific interval of all frames can be set.

            "base_folder" : "/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test",
            "cam_ids" : [0,1,2,3,4,5]
        }


    },


    "detector" : {

        "module_name" : "detectors.mmdetection_detector",
        "class_name" : "Mmdetection_detector",

        "mmdetection_config" : "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x_gta.py",
        "mmdetection_checkpoint_file" : "/home/koehlp/Downloads/work_dirs/detector/cascade_rcnn_x101/epoch_12.pth",
        "device" : "cuda:0",
        #Remove all detections with a confidence less than min_confidence
        "min_confidence" : 0.8,
        "detections_path" : "/home/koehlp/Downloads/work_dirs/detector/detections"



    },


    "feature_extractor" : {

        "module_name" : "feature_extractors.abd_net_extractor",
        "class_name" : "Abd_net_extractor",
        "features_path" : "/home/koehlp/Downloads/work_dirs/feature_extractor/features",

        "feature_extractor_name" : "abd_net_extractor"

            ,"reid_strong_extractor": {
                "reid_strong_baseline_config": "/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
                "checkpoint_file": "/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
                "device": "cuda:0"
                ,"visible_device" : "0"}

            ,"abd_net_extractor" : dict(abd_dan=['cam', 'pam'], abd_dan_no_head=False, abd_dim=1024, abd_np=2, adam_beta1=0.9,
                  adam_beta2=0.999, arch='resnet50', branches=['global', 'abd'], compatibility=False, criterion='htri',
                  cuhk03_classic_split=False, cuhk03_labeled=False, dan_dan=[], dan_dan_no_head=False, dan_dim=1024,
                  data_augment=['crop,random-erase'], day_only=False, dropout=0.5, eval_freq=5, evaluate=False,
                  fixbase=False, fixbase_epoch=10, flip_eval=False, gamma=0.1, global_dim=1024,
                  global_max_pooling=False, gpu_devices='6', height=384, htri_only=False, label_smooth=True,
                  lambda_htri=0.1, lambda_xent=1, lr=0.0003, margin=1.2, max_epoch=80, min_height=-1,
                  momentum=0.9, night_only=False, np_dim=1024, np_max_pooling=False, np_np=2, np_with_global=False,
                  num_instances=4, of_beta=1e-06, of_position=['before', 'after', 'cam', 'pam', 'intermediate'],
                  of_start_epoch=23, open_layers=['classifier'], optim='adam', ow_beta=0.001,
                  pool_tracklet_features='avg', print_freq=10, resume='', rmsprop_alpha=0.99
                  , load_weights='/net/merkur/storage/deeplearning/users/koehl/reid/checkpoint_ep30_non_clean.pth.tar'
                  , root='/should/not/be/necessary'
                       , sample_method='evenly'
                       , save_dir='/should/not/be/necessary'
                       , seed=1, seq_len=15,
                  sgd_dampening=0, sgd_nesterov=False, shallow_cam=True, source_names=['mta_ext'], split_id=0,
                  start_epoch=0, start_eval=0, stepsize=[20, 40], target_names=['market1501'],
                  test_batch_size=100, train_batch_size=64, train_sampler='', use_avai_gpus=False, use_cpu=False,
                  use_metric_cuhk03=False, use_of=True, use_ow=True, visualize_ranks=False, weight_decay=0.0005,
                  width=128, workers=4)


    },

    "tracker" : {
        "type" : "DeepSort",
        "nn_budget" : 100

    }

}

