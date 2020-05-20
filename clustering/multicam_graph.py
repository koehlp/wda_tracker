

from utilities.dataset_statistics import get_combined_dataframe
from collections import defaultdict

from utilities.helper import get_bbox_middle_pos,get_bbox_of_row,constrain_bbox_to_img_dims,get_bbox_height
import sys
from tqdm import tqdm
import numpy as np
import cv2

import os
import pickle


def calculate_track_velocity(track, n_tail=40):
    last_n_positons = track[-n_tail:]

    first_pos_track1 = get_bbox_middle_pos(last_n_positons[0]["bbox"])
    second_pos_track1 = get_bbox_middle_pos(last_n_positons[-1]["bbox"])
    first_pos_frame_no_track1 = last_n_positons[0]["frame_no_cam"]
    second_pos_frame_no_track1 = last_n_positons[-1]["frame_no_cam"]

    velocity = (np.array(second_pos_track1) - np.array(first_pos_track1)) / (
                second_pos_frame_no_track1 - first_pos_frame_no_track1)

    return np.linalg.norm(velocity)

def create_multicam_graph(dataset_path,working_dirs,cam_count,person_identifier="ped_id",pickle_path=None):

    def create_voronoi_grid(img_dims,cam_count,cell_size):

        cam_id_to_nodes = defaultdict(list)

        x_split_count = int(img_dims[0] / cell_size[0]) + 1
        y_split_count = int(img_dims[1] / cell_size[0]) + 1

        x_grid_positions = np.linspace(0, img_dims[0], x_split_count)
        y_grid_positions = np.linspace(0, img_dims[1], y_split_count)


        for cam_id in range(cam_count):

            nodes = []
            for x_grid_pos in x_grid_positions:
                for y_grid_pos in y_grid_positions:

                    new_node = {"node_id": (len(nodes), cam_id)
                        , "position": (x_grid_pos,y_grid_pos)
                        , "person_id_to_frame_no_cam": {}}
                    nodes.append(new_node)

            cam_id_to_nodes[cam_id] = nodes


        return cam_id_to_nodes



    def get_last_node_in_all_cams(person_id, cam_id, cam_id_to_person_id_to_last_node):

        if person_id in cam_id_to_person_id_to_last_node[cam_id]:
            return cam_id_to_person_id_to_last_node[cam_id][person_id]

        for cam_id, person_id_to_last_node in cam_id_to_person_id_to_last_node.items():
            if person_id in person_id_to_last_node:
                return person_id_to_last_node[person_id]

        return None


    def get_graph_node(query_pos,cam_id,cam_id_to_nodes):

        #Find the nearest node in one camera
        nodes = cam_id_to_nodes[cam_id]

        min_dist_and_node = (sys.maxsize,None)
        for node in nodes:

            distance = np.linalg.norm(np.array(node["position"]) - np.array(query_pos))
            min_dist_and_node = min((distance,node),min_dist_and_node,key=lambda x:x[0])


        return min_dist_and_node[1]



    def add_frame_no_and_person_id(node,frame_no_cam,person_id):
        if node is not None:
            node["person_id_to_frame_no_cam"][person_id] = frame_no_cam


    def add_track_pos_to_tail(cam_id_to_person_id_to_tail,cam_id,person_id,xyxy_bbox,frame_no_cam,n_tail=40):


        tail = cam_id_to_person_id_to_tail[cam_id][person_id]

        tail.append({"bbox" : tuple(xyxy_bbox) , "frame_no_cam" : frame_no_cam})
        tail = tail[-n_tail:]

        cam_id_to_person_id_to_tail[cam_id][person_id] = tail





    def calculate_edge_distance(person_id_to_average_velocity,person_id,last_node,new_node):


        persons_velocity = person_id_to_average_velocity[person_id]

        frame_no_cam_new_node = new_node["person_id_to_frame_no_cam"][person_id]
        frame_no_cam_last_node = last_node["person_id_to_frame_no_cam"][person_id]

        time_difference = frame_no_cam_new_node - frame_no_cam_last_node

        #calculate the distance the person might have walked having the velocity that was measured recently
        #normalize via bbox height because the velocity in pixel per frame_no is dependent on distance to the camera
        estimated_distance = time_difference * persons_velocity

        return estimated_distance



    def add_edge_distance(edge_to_distance_bins,person_id_to_average_velocity,cam_id,person_id,last_node,new_node
                          ,value_deviation_percentage=0.1):
        '''
        This will create some kind of histogram but the bins can overlap. But that is sufficient
        because it only should reduce the values.
        :param edge_to_distance_bins:
        :param cam_id_to_person_id_to_tail:
        :param cam_id:
        :param person_id:
        :param last_node:
        :param new_node:
        :param value_deviation_percentage:
        :return:
        '''

        edge_distance = calculate_edge_distance(person_id_to_average_velocity,person_id,last_node,new_node)

        last_node_id = last_node["node_id"]
        new_node_id = new_node["node_id"]

        edge = (last_node_id,new_node_id)

        distance_bins = edge_to_distance_bins[edge]
        match_bin = None
        for distance_bin in distance_bins:

            if distance_bin["min"] <= edge_distance and edge_distance <= distance_bin["max"]:
                match_bin = distance_bin
                break



        if match_bin is None:
            match_bin = { "max" : (1.0 + value_deviation_percentage) * edge_distance
                                 , "min" : (1.0 - value_deviation_percentage) * edge_distance
                                 , "count" : 0}

            distance_bins.append(match_bin)

        match_bin["count"] += 1

        edge_to_distance_bins[edge] = distance_bins


    def calculate_person_id_average_velocity(combined_df,max_velocities=1000):
        print("Calculating persons average velocities.")
        cam_id_to_person_id_to_tail = defaultdict(lambda: defaultdict(list))

        frame_nos = list(combined_df.groupby("frame_no_cam", as_index=False).count()["frame_no_cam"])

        person_id_to_velocities = defaultdict(list)
        person_id_to_average_velocity = {}

        for frame_no_cam in tqdm(frame_nos):
            coods_one_frame = combined_df[combined_df["frame_no_cam"] == frame_no_cam]

            for idx, coord_row in coods_one_frame.iterrows():
                person_id = coord_row[person_identifier]
                cam_id = coord_row["cam_id"]

                bbox = constrain_bbox_to_img_dims(get_bbox_of_row(coord_row))

                # The tail is needed to calculate the velocity of a person
                add_track_pos_to_tail(cam_id_to_person_id_to_tail, cam_id, person_id, bbox, frame_no_cam, n_tail=40)

                tail = cam_id_to_person_id_to_tail[cam_id][person_id]

                if len(tail) < 2:
                    continue

                if len(person_id_to_velocities[person_id]) > max_velocities:
                    continue

                persons_velocity = calculate_track_velocity(tail) / get_bbox_height(tail[-1]["bbox"])

                person_id_to_velocities[person_id].append(persons_velocity)


        for person_id,velocities in person_id_to_velocities.items():
            person_id_to_average_velocity[person_id] = sum(velocities) / len(velocities)

        return person_id_to_average_velocity






    if pickle_path is not None and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            result = pickle.load(pickle_file)
            return result


    combined_df = get_combined_dataframe(dataset_path=dataset_path
                                         ,working_dirs=working_dirs
                                         ,cam_count=range(cam_count)
                                         ,person_identifier=person_identifier)

    person_id_to_average_velocity = calculate_person_id_average_velocity(combined_df)

    frame_nos = list(combined_df.groupby("frame_no_cam",as_index=False).count()["frame_no_cam"])

    cam_id_to_person_id_to_last_node = defaultdict(dict)
    cam_id_to_nodes = create_voronoi_grid(img_dims=(1920,1080),cam_count=cam_count,cell_size=(200,200))

    edge_to_distance_bins = defaultdict(list)

    for frame_no_cam in tqdm(frame_nos):
        coods_one_frame = combined_df[combined_df["frame_no_cam"] == frame_no_cam]

        for idx, coord_row in coods_one_frame.iterrows():
            person_id = coord_row[person_identifier]

            #Unfortunately the bbox can have points outside the image and has to be constraint
            bbox = constrain_bbox_to_img_dims(get_bbox_of_row(coord_row))
            person_position = get_bbox_middle_pos(bbox)

            cam_id = coord_row["cam_id"]

            last_node = get_last_node_in_all_cams(person_id, cam_id, cam_id_to_person_id_to_last_node)

            #get a node out of cam_id_to_nodes if person_position is within its max_node_radius
            #Otherwise create a new node at the person_position
            new_node = get_graph_node(person_position,cam_id,cam_id_to_nodes)

            #frame_no_cam and person_id are needed to calculate a distance a person might have walked
            add_frame_no_and_person_id(new_node,frame_no_cam,person_id)

            #Only if the person moved to another node an edge has to be inserted
            if last_node is not None and last_node["node_id"] != new_node["node_id"]:
                add_edge_distance(edge_to_distance_bins,person_id_to_average_velocity,cam_id,person_id
                                    ,last_node,new_node)

            cam_id_to_person_id_to_last_node[cam_id][person_id] = new_node

    result_dict = { "edge_to_distance_bins" : edge_to_distance_bins , "cam_id_to_nodes" : cam_id_to_nodes }


    if pickle_path is not None:
        with open(pickle_path,"wb") as pickle_file:
            pickle.dump(result_dict,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

    return result_dict


def get_different_cam_edge_nodes(multi_cam_graph):

    def get_node_id_to_node(cam_id_to_nodes):
        node_id_to_node = dict()
        for cam_id,nodes in cam_id_to_nodes.items():
            for node in nodes:
                node_id_to_node[node["node_id"]] = node

        return node_id_to_node

    cam_id_to_nodes = multi_cam_graph["cam_id_to_nodes"]
    edge_to_distance_bins = multi_cam_graph["edge_to_distance_bins"]
    node_id_to_node = get_node_id_to_node(cam_id_to_nodes)

    #These node ids will have a transition from one cam to another cam
    cam_id_to_different_cam_edge_nodes = defaultdict(list)

    cam_id_to_node_id_to_diff_cam_node = defaultdict(dict)
    for edge, distance_bin in edge_to_distance_bins.items():
        #an edge looks like: (from_node_id,to_node_id) where a node_id is (len(nodes),cam_id)  e.g. ((12,2),(11,2))
        #now we want all edges with camera transitions
        if edge[0][1] != edge[1][1]:
            first_node_id = edge[0]
            second_node_id = edge[1]
            first_cam_id = first_node_id[1]
            second_cam_id = second_node_id[1]

            first_node = node_id_to_node[first_node_id]
            second_node = node_id_to_node[second_node_id]
            if "different_cam_nodes" not in first_node:
                first_node["different_cam_nodes"] = set()

            if "different_cam_nodes" not in second_node:
                second_node["different_cam_nodes"] = set()

            first_node["different_cam_nodes"].add(second_node_id)
            #second_node["different_cam_nodes"].add(first_node_id) #fixed from first_cam_id to first_node_id but +1% idf1 before
            cam_id_to_node_id_to_diff_cam_node[first_cam_id][first_node["node_id"]] = first_node
            cam_id_to_node_id_to_diff_cam_node[second_cam_id][second_node["node_id"]] = second_node

    #This will be done because one node can have more than one outgoing edges
    #So every node could be more than one time in the diff cam edge nodes list
    for cam_id, node_id_to_diff_cam_node in cam_id_to_node_id_to_diff_cam_node.items():

        for node_id, diff_cam_node in node_id_to_diff_cam_node.items():
            cam_id_to_different_cam_edge_nodes[cam_id].append(diff_cam_node)


    return cam_id_to_different_cam_edge_nodes




def visualize_graph(dataset_path, multicam_graph_path, multicam_graph):
    def get_node_id_to_node(cam_id_to_nodes):
        node_id_to_node = {}
        for cam_id, nodes in cam_id_to_nodes.items():
            for node in nodes:
                node_id_to_node[node["node_id"]] = node

        return node_id_to_node

    def create_cam_grid(cam_ids,dataset_path,shape=(3,3)):
        rows = []
        row = []
        column_no = 0
        row_no = 0
        cam_id_to_grid_position = {}
        for cam_id in cam_ids:
            cam_id_to_grid_position[cam_id] = (row_no,column_no)
            column_no += 1
            cam_img_path = os.path.join(dataset_path,"cam_{}".format(cam_id),"image_0_{}.jpg".format(cam_id))
            img = cv2.imread(cam_img_path)
            row.append(img)

            if column_no >= shape[1]:
                vis = np.concatenate(row, axis=1)
                rows.append(vis)
                column_no = 0
                row_no += 1
                row = []

        print(cam_id_to_grid_position)





        vis = np.concatenate(rows, axis=0)

        return {"concatenated_cams" : vis , "cam_id_to_grid_position" : cam_id_to_grid_position }



    def draw_edges(multicam_graph,cam_grid,img_dims=(1920,1080)):
        edge_to_distance_bins = multicam_graph["edge_to_distance_bins"]
        cam_id_to_nodes = multicam_graph["cam_id_to_nodes"]
        concatenated_cams = cam_grid["concatenated_cams"]
        cam_id_to_grid_position = cam_grid["cam_id_to_grid_position"]


        node_id_to_node = get_node_id_to_node(cam_id_to_nodes)

        for edge, distance_bins in edge_to_distance_bins.items():
            node1_id = edge[0]
            node2_id = edge[1]
            node1 = node_id_to_node[node1_id]
            node2 = node_id_to_node[node2_id]
            node1_pos = node1["position"]
            node2_pos = node2["position"]
            node1_cam_id = node1_id[1]
            node2_cam_id = node2_id[1]

            cam_node1_grid_pos = cam_id_to_grid_position[node1_cam_id]
            cam_node2_grid_pos = cam_id_to_grid_position[node2_cam_id]


            node1_pos_in_grid = []
            node1_pos_in_grid.append(img_dims[0] * cam_node1_grid_pos[1] + node1_pos[0])
            node1_pos_in_grid.append(img_dims[1] * cam_node1_grid_pos[0] + node1_pos[1])
            node1_pos_in_grid = tuple(map(int, node1_pos_in_grid))

            node2_pos_in_grid = []
            node2_pos_in_grid.append(img_dims[0] * cam_node2_grid_pos[1] + node2_pos[0])
            node2_pos_in_grid.append(img_dims[1] * cam_node2_grid_pos[0] + node2_pos[1])

            node2_pos_in_grid = tuple(map(int,node2_pos_in_grid))

            cv2.circle(concatenated_cams, node1_pos_in_grid, 2, (0, 0, 255), -1)

            cv2.line(concatenated_cams,node1_pos_in_grid,node2_pos_in_grid,color=(255,0,0))

            cv2.circle(concatenated_cams, node2_pos_in_grid, 2, (0, 0, 255), -1)

        return concatenated_cams



    cam_grid = create_cam_grid(range(6),dataset_path)
    img = draw_edges(multicam_graph,cam_grid)

    cv2.imwrite(multicam_graph_path,img)
    cv2.namedWindow('multigraph', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('multigraph', 1920, 1080)
    cv2.imshow('multigraph', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":



    multicam_graph = create_multicam_graph("/home/philipp/Downloads/Recording_12.07.2019/"
                                             , "/home/philipp/work_dirs/"
                                             , 6
                                             , person_identifier="ped_id"
                                            ,pickle_path="/home/philipp/work_dirs/clustering/multicam_graph.pkl")

    visualize_graph("/home/philipp/Downloads/Recording_12.07.2019/","/home/philipp/work_dirs/clustering/multicam_graph_visualization.jpg", multicam_graph)
    #cam_id_to_different_cam_edge_nodes = get_different_cam_edge_nodes(multicam_graph)
