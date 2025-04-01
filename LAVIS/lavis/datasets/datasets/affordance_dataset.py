
import numpy as np
import torch
import json
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transforms3d.quaternions import quat2mat

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def img_normalize_train(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def img_normalize_val(img, scale=256/224, input_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def random_rotation_matrix():
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    q = np.array([np.cos(t2)*r2, np.sin(t1)*r1, np.cos(t1)*r1, np.sin(t2)*r2])
    return quat2mat(q)


def rotate_point_cloud_SO3(batch_data):
    """
    Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along three axis
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_matrix = random_rotation_matrix()
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = (
            np.matmul(rotation_matrix, shape_pc.reshape((-1, 3)).T)).T
    return rotated_data

class AffordanceDataset(Dataset):
    def __init__(self, run_type, data_type, point_path, img_path, description_path, pair=3, img_size=(224, 224)):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.d_path = description_path
        self.pair_num = pair
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab']
        '''
        Unseen
        '''
        if data_type == 'Unseen':
            number_dict = {'Knife': 0, 'Refrigerator': 0, 'Earphone': 0, 
            'Bag': 0, 'Keyboard': 0, 'Chair': 0, 'Hat': 0, 'Door': 0, 'TrashCan': 0, 'Table': 0, 
            'Faucet': 0, 'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Display': 0, 'Mug': 0, 'Clock': 0}

        '''
        Seen
        '''
        if data_type == 'Seen':
            number_dict = {'Earphone': 0, 'Bag': 0, 'Chair': 0, 'Refrigerator': 0, 'Knife': 0, 'Dishwasher': 0, 'Keyboard': 0, 'Scissors': 0, 'Table': 0, 
            'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Microwave': 0, 'Display': 0, 'TrashCan': 0, 'Hat': 0, 'Clock': 0, 
            'Door': 0, 'Mug': 0, 'Faucet': 0, 'Vase': 0, 'Laptop': 0, 'Bed': 0}

        self.img_files = self.read_file(self.i_path)
        self.description_files = self.read_file(self.d_path)
        self.img_size = img_size

        if 'Rotation' in self.p_path:
            self.rotate = True
        else:
            self.rotate = False

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)
    
    def collater(self, samples):
        return default_collate(samples)
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        processed_data = {}
        img_path = self.img_files[index]
        description_path = self.description_files[index]

        if (self.run_type=='val'):
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('_')[-3]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0],range_[1]), self.pair_num)

        Img = Image.open(img_path).convert('RGB')

        if(self.run_type == 'train'):
            Object_cls = img_path.split('_')[-3]
            Affordance_cls = img_path.split('_')[-2]
            Description = self.read_description(description_path)
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Points_List = []
            Affordance_label_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                if self.rotate:
                    Points = rotate_point_cloud_SO3(Points[np.newaxis, :, :]).squeeze()
                Points, _, _ = pc_normalize(Points)
                Points = Points.transpose()
                Affordance_label, Affordance_index = self.get_affordance_label(img_path, affordance_label)
                Points_List.append(Points)
                Affordance_label_List.append(Affordance_label)
            processed_data['Img'] = Img
            processed_data['Points_List'] = Points_List
            processed_data['Description'] = Description
            processed_data['Affordance_label_List'] = Affordance_label_List
            processed_data['Object_cls'] = Object_cls
            processed_data['Affordance_cls'] = Affordance_cls

        else:
            Object_cls = img_path.split('_')[-3]
            Affordance_cls = img_path.split('_')[-2]
            Description = self.read_description(description_path)
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Point, Affordance_label = self.extract_point_file(point_path)
            Point, _, _ = pc_normalize(Point)
            Point = Point.transpose()

            Affordance_label, Affordance_index = self.get_affordance_label(img_path, Affordance_label)

            processed_data['Img'] = Img
            processed_data['Point'] = Point
            processed_data['Description'] = Description
            processed_data['Affordance_label'] = Affordance_label
            processed_data['Object_cls'] = Object_cls
            processed_data['Affordance_cls'] = Affordance_cls

        return processed_data

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('_')[-2]
                    number_dict[object_] +=1
                file_list.append(file)

            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
    
    def extract_point_file(self, path):
        with open(path,'r') as f:
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

        return points_coordinates, affordance_label

    def get_affordance_label(self, str, label):
        cut_str = str.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)

        label = label[:, index]
        
        return label, index

    def read_description(self, json_path):

        json_data = json.load(open(json_path, 'r'))
        Description = json_data['Description']
        # Description = "None"

        return Description