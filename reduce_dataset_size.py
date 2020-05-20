import zipfile
from utilities.helper import group_and_drop_unnecessary
from utilities.pandas_loader import *
import pandas as pd

def get_min_frame_no(dataset_folder,dataset_type):

    cam_folder = os.path.join(dataset_folder,dataset_type,"cam_0/")

    cam_images = os.listdir(cam_folder)
    cam_images = [ image_name for image_name in cam_images if image_name.endswith(".jpg") ]
    #image_1133_0.jpg
    frame_nos = [ int(image_name.split("_")[1]) for image_name in cam_images ]

    min_frame_no = min(frame_nos)
    return min_frame_no


def convert_dataset_images_to_videos(dataset_folder,dataset_output_folder):

    for dataset_type in ["train","test"]:
        min_frame_no = get_min_frame_no(dataset_folder=dataset_folder,dataset_type=dataset_type)

        for cam_id in [0,1,2,3,4,5]:
            image_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"image_%d_{}.jpg".format(cam_id))
            output_video_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_video_folder,exist_ok=True)
            output_video_path = os.path.join(output_video_folder,"cam_{}.mp4".format(cam_id))

            print("Starting to convert {}".format(output_video_path))
            os.system("taskset -c 0-30 ffmpeg -r 41 -start_number {} -i {} -b:v 10M {}".format(min_frame_no,image_path,output_video_path))





def zip_cam_coords(dataset_folder,dataset_output_folder):

    for dataset_type in ["train","test"]:

        for cam_id in [0,1,2,3,4,5]:
            cam_coords_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"coords_cam_{}.csv".format(cam_id))
            output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_folder,exist_ok=True)
            output_zip_path = os.path.join(output_folder,"coords_cam_{}.csv.zip".format(cam_id))

            zipf = zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED)
            zipf.write(cam_coords_path,arcname=os.path.basename(cam_coords_path))
            zipf.close()

            compression_ratio = 1 - os.stat(output_zip_path).st_size / os.stat(cam_coords_path).st_size
            print("compression ratio of file {} : {}".format(os.path.basename(cam_coords_path),compression_ratio))



def reduce_cam_coords(dataset_folder,dataset_output_folder,working_dir):

    for dataset_type in ["train","test"]:

        for cam_id in [0,1,2,3,4,5]:
            cam_coords_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"coords_cam_{}.csv".format(cam_id))
            output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_folder,exist_ok=True)
            output_path = os.path.join(output_folder, "coords_fib_cam_{}.csv".format(cam_id))

            coords_dataframe = load_csv(working_dir, cam_coords_path)

            coords_dataframe = coords_dataframe.astype(int)

            coords_dataframe = group_and_drop_unnecessary(coords_dataframe)

            coords_dataframe.to_csv(path_or_buf=output_path,index=False)


if __name__ == "__main__":
    '''
    convert_dataset_images_to_videos(dataset_folder = "/media/philipp/philippkoehl_ssd/GTA_ext_short"
                                    , dataset_output_folder = "/media/philipp/philippkoehl_ssd/MTA_ext_short")
    '''
    '''
    convert_dataset_images_to_videos(dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019"
                                      , dataset_output_folder="/net/merkur/storage/deeplearning/users/koehl/gta/MTA_videos")
    '''

    '''
    zip_cam_coords(dataset_folder = "/media/philipp/philippkoehl_ssd/GTA_ext_short"
                                    , dataset_output_folder = "/media/philipp/philippkoehl_ssd/MTA_ext_short_coords_zip")
    '''

    zip_cam_coords(dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019"
                   , dataset_output_folder="/net/merkur/storage/deeplearning/users/koehl/gta/MTA_coords_zip")

    reduce_cam_coords(dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019"
                                      , dataset_output_folder="/net/merkur/storage/deeplearning/users/koehl/gta/MTA_videos"
                                        ,working_dir="/home/koehlp/Downloads/work_dirs/")


    '''
    reduce_cam_coords(dataset_folder = "/media/philipp/philippkoehl_ssd/GTA_ext_short"
                                    , dataset_output_folder = "/media/philipp/philippkoehl_ssd/MTA_ext_short"
                                    ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs/")
    '''