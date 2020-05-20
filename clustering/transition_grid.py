

import numpy as np
from utilities.pandas_loader import load_csv
from utilities.helper import adjustCoordsTypes
from collections import defaultdict
from utilities.helper import constrain_bbox_to_img_dims
from tqdm import tqdm
import pickle
import os

import cv2

class Transition_grid:
    def __init__(self,work_dirs,cam_coords_path,img_dims=(1920,1080),cell_size=(20,20),person_identifier="ped_id"):
        self.work_dirs = work_dirs
        self.cam_coords_path = cam_coords_path
        self.cell_size = cell_size
        self.person_identifier = person_identifier
        self.x_split_count = int(img_dims[0] / self.cell_size[0]) + 1
        self.y_split_count = int(img_dims[1] / self.cell_size[0]) + 1

        self.x_grid_positions = np.linspace(0,img_dims[0],self.x_split_count)
        self.y_grid_positions = np.linspace(0, img_dims[1], self.y_split_count)

        self.transition_matrix = np.zeros((8,self.y_split_count-1,self.x_split_count-1),dtype=np.int32)

        self.entry_exit_matrix = np.zeros((2,self.y_split_count-1,self.x_split_count-1),dtype=np.int32)



        self.img_dims = img_dims

        #Maps the offset from a cell in the center to its eight neighbours in a clockwise manner
        self.cell_xy_offset_to_index = {

            (1, 0): 0,
            (1, -1): 1,
            (0, -1): 2,
            (-1, -1): 3,
            (-1, 0): 4,
            (-1, 1): 5,
            (0, 1): 6,
            (1, 1): 7

        }

    def load_groundtruth_csv(self):
        self.cam_coords = load_csv(self.work_dirs, self.cam_coords_path)

        self.cam_coords = self.group_cam_coords(self.cam_coords)

    def calculate_abs_freq(self):
        transitions_matrix_sum = np.sum(self.transition_matrix,axis=0)


        result = np.add(transitions_matrix_sum,self.entry_exit_matrix[1,:,:])

        return result

    def draw_abs_freq(self):

        import matplotlib.pyplot as plt
        import numpy as np

        abs_freq = self.calculate_abs_freq()

        plt.imshow(abs_freq)
        plt.show()

    def draw_entry_exits(self):

        import matplotlib.pyplot as plt
        import numpy as np

        abs_freq = self.entry_exit_matrix[1,:,:]

        plt.imshow(abs_freq)
        plt.show()



    def get_bbox_middle_pos(self,bbox):

        xtl, ytl, xbr, ybr = bbox
        x = xtl + ((xbr - xtl) / 2.)
        y = ytl + ((ybr - ytl) / 2.)

        return (x,y)

    def group_cam_coords(self,cam_coords):
        if "ped_id" in cam_coords.columns:
            person_id_name = "ped_id"
        else:
            person_id_name = "person_id"

        cam_coords = cam_coords.groupby(["frame_no_gta", person_id_name], as_index=False).mean()
        cam_coords = adjustCoordsTypes(cam_coords,self.person_identifier)

        return cam_coords

    def is_person_pos_in_cell(self,cell_bbox,person_pos):
        x_person_pos,y_person_pos = person_pos

        xtl, ytl, xbr,ybr = cell_bbox

        return (x_person_pos >= xtl) \
               and ((x_person_pos < xbr) or x_person_pos == self.img_dims[0]) \
               and (y_person_pos >= ytl) \
               and ((y_person_pos < ybr) or y_person_pos == self.img_dims[1])


    def get_person_cell(self,person_pos):
        for x_grid_idx in range(len(self.x_grid_positions) - 1):
            for y_grid_idx in range(len(self.y_grid_positions) - 1):
                xtl_grid_cell = self.x_grid_positions[x_grid_idx]
                ytl_grid_cell = self.y_grid_positions[y_grid_idx]
                xbr_grid_cell = self.x_grid_positions[x_grid_idx + 1]
                ybr_grid_cell = self.y_grid_positions[y_grid_idx + 1]

                is_person_in_cell = self.is_person_pos_in_cell((xtl_grid_cell, ytl_grid_cell, xbr_grid_cell, ybr_grid_cell), person_pos)

                if is_person_in_cell:
                    return (x_grid_idx,y_grid_idx)


        assert(True,"Person position outside of image.")

    def get_neighbour_index(self,from_cell,to_cell):
        #A Cell has 8 neighbour cells this index will be returned from this method
        #Their indices are enumerated clockwise starting at the top neighbour cell with 0



        cell_xy_offset = tuple(np.subtract(from_cell,to_cell))

        if cell_xy_offset not in self.cell_xy_offset_to_index:
            return -1


        return self.cell_xy_offset_to_index[cell_xy_offset]


    def load_transition_matrix(self):

        cam_coords_folder, cam_coords_filename = os.path.split(self.cam_coords_path)
        cam_coords_folder = os.path.normpath(cam_coords_folder)
        dataset_foldername = cam_coords_folder.split(os.sep)[-2]

        transition_matrix_folder = os.path.join(self.work_dirs,"clustering","transition_matrix",dataset_foldername)

        os.makedirs(transition_matrix_folder,exist_ok=True)

        transition_matrix_pickle_path = os.path.join(transition_matrix_folder,"transition_matrix.pkl")

        if os.path.exists(transition_matrix_pickle_path):
            with open(transition_matrix_pickle_path, "rb") as pickle_file:

                matrices = pickle.load(pickle_file)

                self.transition_matrix = matrices["transition_matrix"]
                self.entry_exit_matrix = matrices["entry_exit_matrix"]

        else:
            self.calculate_transition_matrix()
            with open(transition_matrix_pickle_path, "wb") as pickle_file:
                matrices = { "transition_matrix" : self.transition_matrix , "entry_exit_matrix" : self.entry_exit_matrix  }
                print("Saved transition matrix.")
                pickle.dump(matrices, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)









    def calculate_transition_matrix(self):
        self.load_groundtruth_csv()
        person_id_to_cell_in_previous_frame = {}
        last_frame_no_cam = -1
        person_ids_in_frame = set()

        print("Starting to calculate transition matrix")
        for index,cam_coord_row in tqdm(self.cam_coords.iterrows(),total=len(self.cam_coords)):

            if last_frame_no_cam != int(cam_coord_row["frame_no_cam"]):
                #A new frame starts
                last_frame_no_cam = int(cam_coord_row["frame_no_cam"])

                #Person Ids that did not occur anymore in the current frame but have been observed in the one before have to be removed
                #Because a new frame begins and only a neighbour transition will be calculated not transition over multiple cells
                persons_not_observed_in_current_frame = person_id_to_cell_in_previous_frame.keys() - person_ids_in_frame

                for person_id_to_remove in persons_not_observed_in_current_frame:
                    cell_x, cell_y = person_id_to_cell_in_previous_frame[person_id_to_remove]
                    self.entry_exit_matrix[1, cell_y, cell_x] += 1
                    person_id_to_cell_in_previous_frame.pop(person_id_to_remove, None)

                person_ids_in_frame.clear()

            person_id = int(cam_coord_row[self.person_identifier])

            person_ids_in_frame.add(person_id)

            person_bbox = ( cam_coord_row["x_top_left_BB"]
                            , cam_coord_row["y_top_left_BB"]
                            , cam_coord_row["x_bottom_right_BB"]
                            , cam_coord_row["y_bottom_right_BB"] )

            person_bbox = constrain_bbox_to_img_dims(person_bbox)

            person_pos = self.get_bbox_middle_pos(person_bbox)

            cell = self.get_person_cell(person_pos)

            #If the person has been in the previous frame a transition may have occured if the new cell is different from the previous cell
            if person_id in person_id_to_cell_in_previous_frame:
                cell_in_previous_frame = person_id_to_cell_in_previous_frame[person_id]

                #Check if the current cell is different from the previous cell because a transition from cell to cell has occured if true
                if cell_in_previous_frame != cell:

                    transition_index = self.get_neighbour_index(cell_in_previous_frame,cell)

                    if transition_index == -1:
                        person_ids_in_frame.remove(person_id)
                        continue


                    cell_in_previous_frame_x,cell_in_previous_frame_y = cell_in_previous_frame

                    self.transition_matrix[transition_index,cell_in_previous_frame_y,cell_in_previous_frame_x] += 1

                    #Overwrite the old cell for the person
                    person_id_to_cell_in_previous_frame[person_id] = cell


            else:
                #The person is new and has not been seen in the previous cell
                person_id_to_cell_in_previous_frame[person_id] = cell
                cell_x, cell_y = cell
                self.entry_exit_matrix[0,cell_y,cell_x] += 1













if __name__ == "__main__":

    '''
    transition_grid = Transition_grid( "/home/philipp/work_dirs"
                                       ,"/home/philipp/Downloads/Recording_12.07.2019/cam_2/coords_cam_2.csv")
                                       
    '''


    transition_grid = Transition_grid( "/home/koehlp/Downloads/work_dirs"
                                           ,"/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train/cam_2/coords_cam_2.csv"
                                       , person_identifier="person_id")



    '''
    transition_grid = Transition_grid("/home/koehlp/Downloads/work_dirs"
                                      , "/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/coords_cam_2.csv")
    '''

    transition_grid.load_transition_matrix()
    print(transition_grid.draw_entry_exits())