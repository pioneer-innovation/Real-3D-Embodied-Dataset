import mmcv
import numpy as np
from concurrent import futures as futures
from os import path as osp
from scipy import io as sio
import open3d as o3d
import os


def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random sampling.

    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.
        num_points (int): The number of samples.
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]


class R3ADInstance(object):

    def __init__(self, line):
        data = line.split(' ')
        data[:] = [float(x) for x in data[:]]
        self.classname = data[-1]
        # Home_1 xyz format to depth format
        self.x = -data[1]
        self.y = data[0]
        self.z = data[2]

        self.l = data[4]
        self.w = data[3]
        self.h = data[5]

        self.heading_angle = data[6]

        self.centroid = np.array([self.x, self.y, self.z])

        self.box3d = np.concatenate([
            self.centroid,
            np.array([self.l, self.w, self.h, self.heading_angle])
        ])


class R3ADData(object):
    """SUNRGBD data.

    Generate scannet infos for sunrgbd_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
        use_v1 (bool): Whether to use v1. Default: False.
    """

    def __init__(self, root_path, split='train', use_v1=False):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path, 'R3AD_trainval')
        self.classes = [
            'table', 'chair', 'sofa', 'bed', 'cabinet', 'television', 'lamp',
            'shelf', 'bottle', 'cup', 'book', 'dustbin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {
            label: self.classes[label]
            for label in range(len(self.classes))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.split_dir, f'{split}_data_idx.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)

        self.cloud_path_list = [line.rstrip() for line in open(self.split_dir + '/' + 'cloud_path.txt')]
        self.anno_path_list = [line.rstrip() for line in open(self.split_dir + '/' + 'anno_path.txt')]

    def __len__(self):
        return len(self.sample_id_list)


    def get_pc(self, idx):
        cloud_path = self.root_dir + self.cloud_path_list[idx]
        pcd = o3d.io.read_point_cloud(cloud_path)
        xyz = pcd.points
        # o3d format(yz-x) to depth format
        xyz = np.asarray(xyz)[:, [0, 2, 1]]
        xyz[:, 1] = -xyz[:, 1]
        xyz[:, 2] = xyz[:, 2] + 1.1

        RGB = pcd.colors
        RGB = np.asarray(RGB)

        pc = np.concatenate((xyz,RGB),axis=1)

        return pc


    def get_label_objects(self, idx):
        anno_path = self.root_dir + self.anno_path_list[idx]
        lines = [line.rstrip() for line in open(anno_path)]
        objects = [R3ADInstance(line) for line in lines]
        return objects

    def get_infos(self, num_workers=1, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            # convert depth to points
            SAMPLE_NUM = 50000
            # TODO: Check whether can move the point
            #  sampling process during training.
            # pc = self.get_pc(sample_idx)

            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            info['pts_path'] = self.cloud_path_list[sample_idx]

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.values()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        self.classes[int(obj.classname)] for obj in obj_list
                        if obj.classname in self.cat2label.values()
                    ])
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.values()
                    ],
                        axis=0)
                    annotations['dimensions'] = 2 * np.array([
                        [obj.l, obj.w, obj.h] for obj in obj_list
                        if obj.classname in self.cat2label.values()
                    ])  # lhw(depth) format
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.values()
                    ])
                    annotations['index'] = np.arange(
                        len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.values()
                    ])
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.values()
                        ],
                        axis=0)  # (K,8)
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if \
            sample_id_list is not None else self.sample_id_list
        infos=[]
        for i in range(len(sample_id_list)):
            infos.append(process_single_scene(int(sample_id_list[i])))

        return infos
