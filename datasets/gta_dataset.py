import mmcv
import os.path as osp
from datasets.base_dataset import Base_dataset, Base_dataset_image
from utilities.helper import natural_keys
import os

class Gta_dataset_image(Base_dataset_image):
    def __init__(self,image_path,extension):
        self._image_path = image_path

        self._img = mmcv.imread(image_path)
        #Image name structure: image_{frame_no_cam}_{cam_id}
        self._image_name_no_ext = osp.basename(image_path).replace(extension,"")
        image_name_splitted = self.image_name_no_ext.split("_")

        self._frame_no_cam = int(image_name_splitted[1])
        self._cam_id = int(image_name_splitted[2])

        #opencv python delivers (height,width) but we want (width,height)
        self._img_dims = (self._img.shape[:2][1],self._img.shape[:2][0])

    @property
    def cam_id(self):
        return self._cam_id

    @property
    def image_name_no_ext(self):
        return self._image_name_no_ext

    @property
    def img(self):
        return self._img


    @property
    def image_path(self):
        return self._image_path

    @property
    def frame_no_cam(self):
        return self._frame_no_cam

    @property
    def img_dims(self):
        return self._img_dims



def build_cam_paths(dataset_base: str,cam_ids: list):

    paths = []
    for cam_id in cam_ids:
        cam_folder_name = "cam_{}".format(cam_id)
        paths.append(osp.join(dataset_base,cam_folder_name))

    return paths


def get_cam_iterators(cfg,dataset_base, cam_ids):
    cam_paths = build_cam_paths(dataset_base,cam_ids)

    iterators = []
    for cam_path in cam_paths:
        iterators.append(Gta_dataset(cfg,cam_path))

    return iterators


def len_sum(cam_iterators):
    return sum([ len(cam_it) for cam_it in cam_iterators])



class Gta_dataset(Base_dataset):

    def __init__(self, cfg, cam_path):
        super().__init__()
        self.cfg = cfg

        self.cam_id = int(os.path.basename(cam_path).split("_")[1])
        self.coords_cam_path = os.path.join(cam_path,"coords_cam_{}.csv".format(self.cam_id))

        self.cam_path = cam_path
        self.image_paths = []
        self.image_paths_iter = iter(self.image_paths)

        self.selection_interval = self.cfg.data.selection_interval
        self.load_images_from_cam_path()


    def __len__(self):
        return len(self.image_paths)

    def load_images_from_cam_path(self):
        selection_interval_start, selection_interval_end = self.selection_interval


        image_names = os.listdir(self.cam_path)

        #Just images with the desired extension which can be specified in the cfg
        image_names = [ image_name for image_name in image_names if image_name.endswith(self.cfg.data.extension) ]

        #Sort the image names
        image_names.sort(key=natural_keys)

        image_names_len = len(image_names)

        #Select only those within the specified selection
        if image_names_len < selection_interval_start:
            selection_interval_start = image_names_len

        if image_names_len < selection_interval_end:
            selection_interval_end = image_names_len

        image_names = image_names[selection_interval_start:selection_interval_end]

        self.image_paths = [ osp.join(self.cam_path,image_name) for image_name in image_names ]

        self.image_paths_iter = iter(self.image_paths)



    def __iter__(self):
        return self

    def __next__(self):
        image_path = next(self.image_paths_iter,None)

        if image_path is None:
            raise StopIteration

        return Gta_dataset_image(image_path,self.cfg.data.extension)


if __name__ == "__main__":
    cfg = mmcv.Config.fromfile("../configs/tracker_configs/faster_rcnn_r50_fpn_1x_default_ckpt.py")
    gta_dataset = Gta_dataset(cfg,cfg.data.train.paths)

    for dataset_image in gta_dataset:
        print("frame_no_cam: {} , cam_id: {} ".format(dataset_image.frame_no_cam,dataset_image.cam_id))

