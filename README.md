# WDA tracker

The WDA (weighted distance aggregation) tracker is an offline multi camera tracking approach.

It is published in the same paper as the MTA Dataset
(https://github.com/schuar-iosb/mta-dataset). Its purpose is to provide a baseline
for the MTA Dataset.


This repository is structured in **two parts**:  

 The **first part** can be used to create single camera tracks 
from an input video. Startable via `run_tracker.py`.

The **second part** can be used to cluster single camera tracks
to obtain multi camera tracks with subsequent evaluation. Startable via `run_multi_cam_clustering.py`.




## Getting started


**Setting up an artefacts folder**  
Download work_dirs.zip and unzip it in your base repository folder or create a symlink e.g. 
`ln -s your/path/work_dirs your/path/wda_tracker/work_dirs`. 
It contains one or two re-id and detector models.
Furthermore output result files will be stored in this folder.



**Download the MTA Dataset**

Go to https://github.com/schuar-iosb/mta-dataset 
and follow the instructions for obtaining the MTA Dataset. It is also possible to use 
the smaller extracted version MTA ext short at first. Unzip the dataset somewhere.

**Configure the single camera tracker**

E.g. in `configs/tracker_configs/frcnn50_new_abd_test.py` and `configs/tracker_configs/frcnn50_new_abd_train.py` 
set the data -> source -> base_folder to your MTA dataset location.


```python

...
"data" : {
        "selection_interval" : [0,30],

        "source" : {
            "base_folder" : "/media/philipp/philippkoehl_ssd/MTA_ext_short/test",
            "cam_ids" : [0,1,2,3,4,5]
        }
    },
...
```

**Run the single camera tracking**

Run the single camera tracking to generate single camera tracks.

For the train set:

```python
python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_train.py
```

And for the test set:

```python

python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_test.py

```


**Configure the multi camera clustering**

Set the following paths of the single camera tracker results in the multi camera clustering config.

E.g. in `configs/clustering_configs/mta_es_abd_non_clean.py`

```python
...
"work_dirs" : "/media/philipp/philippkoehl_ssd/work_dirs"
,"train_track_results_folder" : "/home/philipp/Documents/repos/wda_tracker/work_dirs/tracker/config_runs/frcnn50_new_abd_train/tracker_results"
,"test_track_results_folder" : "/home/philipp/Documents/repos/wda_tracker/work_dirs/tracker/config_runs/frcnn50_new_abd_test/tracker_results"
,"train_dataset_folder" : "/media/philipp/philippkoehl_ssd/MTA_ext_short/train"
,"test_dataset_folder" : "/media/philipp/philippkoehl_ssd/MTA_ext_short/test"
...
```


**Run the multi camera clustering**

Run the following command to start the clustering of single camera tracks which
have been specified in the config file to obtain multi camera tracks.
```python
python run_multi_cam_clustering.py \
    --config configs/clustering_configs/mta_es_abd_non_clean.py
```


## Tracking results

TODO

## Contained repositories

This repository contains a person detector called mmdetection
 (https://github.com/open-mmlab/mmdetection).
 
 
It also contains two person re-identification approaches strong reid baseline (https://github.com/michuanhaohao/reid-strong-baseline) and
ABD-Net (https://github.com/TAMU-VITA/ABD-Net). 
 
For multi person single camera tracking it contains DeepSort from (https://github.com/ZQPei/deep_sort_pytorch) 
which is originally from (https://github.com/nwojke/deep_sort).

The IOU-Tracker (https://github.com/bochinski/iou-tracker) is also contained but not integrated into the system.

Parts of  (https://github.com/ZwEin27/Hierarchical-Clustering) are used for clustering.

For evaluation the py-motmetrics is contained (https://github.com/cheind/py-motmetrics).

An approach for getting distinct colors is used:

(https://github.com/taketwo/glasbey).

## Citation

If you use it, please cite our work.
The affiliated paper will be published soon at the CVPR 2020 VUHCS Workshop (https://vuhcs.github.io/)

```latex
The MTA Dataset for Multi Target Multi Camera Pedestrian Tracking by Weighted Distance Aggregation. 
Philipp Köhl (Fraunhofer IOSB); Andreas Specker (Fraunhofer IOSB); Arne Schumann (Fraunhofer IOSB)* (Oral)
```
