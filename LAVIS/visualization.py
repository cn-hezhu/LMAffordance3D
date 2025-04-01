import open3d as o3d
import numpy as np
import random
Affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                'pull', 'listen', 'wear', 'press', 'cut', 'stab']
color_list = [[252, 19, 19], [249, 113, 45], [247, 183, 55], [251, 251, 11], [178, 244, 44], [255, 0, 0], 
              [0, 0, 255], [25, 248, 99], [46, 253, 184], [40, 253, 253], [27, 178, 253], [28, 100, 243], 
              [46, 46, 125], [105, 33, 247], [172, 10, 253], [249, 47, 249], [253, 51, 186], [250, 18, 95]]
color_list = np.array(color_list)

def get_affordance_label(str, label):
    cut_str = str.split('_')
    affordance = cut_str[-2]
    index = Affordance_label_list.index(affordance)
    label = label[:, index]
    
    return label

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def read_specific_line(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if line_number < 1 or line_number > len(lines):
        raise ValueError("Line number out of range.")
    specific_line = lines[line_number - 1].strip()
    specific_array = np.fromstring(specific_line, sep=' ')
    if specific_array.size != 2048:
        raise ValueError("Each line must contain exactly 2048 numbers.")
    specific_array = specific_array.reshape(2048, 1)
    return specific_array


if __name__=='__main__':
    pcd_path = "data/Rotation_view/Seen/Point/Test/Earphone/Point_Test_Earphone_2.txt"
    img_path = "data/Rotation_view/Seen/Img/Test/Earphone/grasp/Img_Test_Earphone_grasp_2.jpg"
    result_path = "la/result/lmaffordance3d/prediction_best.txt"
    affordance_data = read_specific_line(result_path, 2)

    with open(pcd_path,'r') as f:
        coordinates = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]
        affordance_label = get_affordance_label(img_path, affordance_label)
        f.close() 
        visual_point = o3d.geometry.PointCloud()
        visual_point.points = o3d.utility.Vector3dVector(points_coordinates)

        color = np.zeros((2048,3))
        gt_color = np.array([0, 255, 0]) # [0, 255, 0] Green 
        reference_color = np.array([255, 0, 0]) #  [255, 0, 0] Red
        back_color = np.array([190, 190, 190])

        for i, point_affordacne in enumerate(affordance_data):
            scale_i = point_affordacne
            color[i] = (reference_color-back_color) * scale_i + back_color
        visual_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        o3d.visualization.draw_geometries([visual_point], width=600, height=500)

   