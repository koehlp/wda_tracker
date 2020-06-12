# WDA tracker

The WDA (weighted distance aggregation) tracker is an offline multi camera tracking approach.

It was published in the same paper as the MTA Dataset
(https://github.com/schuar-iosb/mta-dataset). Its purpose is to provide a baseline
for the MTA Dataset.


This repository contains a person detector called mmdetection
 (https://github.com/open-mmlab/mmdetection).
 
 
It also contains two person re-identification approaches strong reid baseline (https://github.com/michuanhaohao/reid-strong-baseline) and
ABD-Net (https://github.com/TAMU-VITA/ABD-Net). 
 
For multi person single camera tracking it contains DeepSort from (https://github.com/ZQPei/deep_sort_pytorch) 
which is originally from (https://github.com/nwojke/deep_sort).

Stuff from (https://github.com/ZwEin27/Hierarchical-Clustering) is used for clustering.

For evaluation the py-motmetrics is contained (https://github.com/cheind/py-motmetrics).

There is also an approach for getting distinct colors:

(https://github.com/taketwo/glasbey).


 
 
 



## Getting started


**Creating work_dirs**  
Download work_dirs.zip and unzip it in your base repository folder or create a symlink e.g. 
`ln -s your/path/work_dirs your/path/wda_tracker/work_dirs`. 
It contains one or two re-id and detector models.
Furthermore output result files will be stored in this folder.
Some paths in config files are relative to the repository root. Therefore it is required
that this folder is in the repository root. 


**Download the MTA Dataset**

Go to https://github.com/schuar-iosb/mta-dataset 
and follow the instructions for obtaining the MTA Dataset.

**Single camera tracker**

Adjust configs/tracker_configs/frcnn50_new_abd.py

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



