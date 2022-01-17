import gym
import os
import random
import torch
import open3d as o3d
import numpy as np
from mmdet3d.core.bbox import DepthInstance3DBoxes


class R3AD_V2(gym.Env):
    """
    A template to implement custom OpenAI Gym environments
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        self.action_space = gym.spaces.Discrete(6)
        self.data_path = "./data/Home_1/"
        self.coors_list = os.listdir(self.data_path)
        self.angle_list = ['0', '45', '90', '135', '180', '225', '270', '315']

        self.votenet = torch.load('./myenv/model_V.pkl')
        self.meta = torch.load('./myenv/meta_V.pkl')

        self.step_num = 0
        self.done = 0

        self.reward_rate = 1

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        coor_int = [int(i) for i in self.coor_now.split(' ')]
        coor_f = ' '.join([str(i) for i in [coor_int[0], coor_int[1] + 1]])
        coor_b = ' '.join([str(i) for i in [coor_int[0], coor_int[1] - 1]])
        coor_l = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1]]])
        coor_r = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1]]])

        if coor_f in self.coors_list and action == 0:
            self.coor_now = coor_f
            self.reward_rate = 1
        elif coor_b in self.coors_list and action == 1:
            self.coor_now = coor_b
            self.reward_rate = 1
        elif coor_l in self.coors_list and action == 2:
            self.coor_now = coor_l
            self.reward_rate = 1
        elif coor_r in self.coors_list and action == 3:
            self.coor_now = coor_r
            self.reward_rate = 1
        elif action == 4:
            self.angle_now = self.angle_list[self.angle_list.index(self.angle_now) - 1]
            self.reward_rate = 0.4
        elif action == 5:
            self.angle_now = self.angle_list[(self.angle_list.index(self.angle_now) + 1) % 8]
            self.reward_rate = 0.4

        coor_space = self.coor_space()
        action_space = torch.FloatTensor(coor_space).cuda()

        cloud_path, _, _, _ = self.path()
        self.cloud_path = cloud_path

        assert os.path.exists(cloud_path)

        reward, obs = self.reward_cal(cloud_path)

        self.step_num += 1

        if self.step_num > 100:
            self.done = True

        return obs, reward*self.reward_rate*0.01, self.done, action_space

        # return (observation, reward, done, info)

    def reset(self):
        """
        Reset the environment state and returns an initial observation
        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        self.coor_now = random.choice(self.coors_list)
        self.angle_now = random.choice(self.angle_list)
        # self.coor_now = '0 0'
        # self.angle_now = '0'
        coor_space = self.coor_space()
        action_space = torch.FloatTensor(coor_space).cuda()

        cloud_path, _, _, _ = self.path()
        self.cloud_path = cloud_path

        points = []
        points.append(self.read_pc(cloud_path))
        self.bbox_results_t1 = self.votenet.simple_test(points, self.meta, None, True)
        obsret = self.votenet.backbone.fp_ret
        obs = torch.cat((obsret['fp_xyz'][0], obsret['fp_features'][0].permute(0, 2, 1)), axis=2)
        self.bbox_results_t1 = self.cam_to_world(self.bbox_results_t1)

        self.step_num = 0
        self.done = False

        return obs, action_space

    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return

    def coor_space(self):
        coor_int = [int(i) for i in self.coor_now.split(' ')]
        coor_f = ' '.join([str(i) for i in [coor_int[0], coor_int[1] + 1]])
        coor_b = ' '.join([str(i) for i in [coor_int[0], coor_int[1] - 1]])
        coor_l = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1]]])
        coor_r = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1]]])

        coor_space = [0, 0, 0, 0, 1, 1]
        if coor_f in self.coors_list:
            coor_space[0] = 1
        if coor_b in self.coors_list:
            coor_space[1] = 1
        if coor_l in self.coors_list:
            coor_space[2] = 1
        if coor_r in self.coors_list:
            coor_space[3] = 1

        return coor_space

    def path(self):
        cloud_path = ''.join([self.data_path, self.coor_now, '/', 'cloud ', self.angle_now, '.pcd'])
        color_path = ''.join([self.data_path, self.coor_now, '/', 'color ', self.angle_now, '.png'])
        depth_path = ''.join([self.data_path, self.coor_now, '/', 'depth ', self.angle_now, '.png'])
        anno_path = ''.join([self.data_path, self.coor_now, '/', 'anno ', self.angle_now, '.txt'])
        return cloud_path, color_path, depth_path, anno_path

    def read_pc(self, path):
        pcd = o3d.io.read_point_cloud(path)
        xyz = pcd.points
        # o3d format(yz-x) to depth format
        xyz = np.asarray(xyz)[:, [0, 2, 1]]
        xyz[:, 1] = -xyz[:, 1]
        xyz[:, 2] = xyz[:, 2] + 1.1

        RGB = pcd.colors
        RGB = np.asarray(RGB)

        pc = np.concatenate((xyz, RGB), axis=1)

        choices = np.random.choice(pc.shape[0], 20000, replace=False)
        pc = pc[choices]

        floor_height = np.percentile(pc[:, 2], 0.99)
        height = pc[:, 2] - floor_height
        pc = np.concatenate([pc, np.expand_dims(height, 1)], 1)

        return torch.from_numpy(pc).float().cuda()

    def iou_cal(self, bbox_results_t1, bbox_results_t2):
        if len(bbox_results_t1[0]['boxes_3d']) == 0 or len(bbox_results_t2[0]['boxes_3d']) == 0:
            return len(bbox_results_t1[0]['boxes_3d']) + len(bbox_results_t2[0]['boxes_3d'])
        else:
            iou_sum = 0
            for i in range(0, 13):
                index1 = bbox_results_t1[0]['labels_3d'] == i
                index2 = bbox_results_t2[0]['labels_3d'] == i

                bbox1 = bbox_results_t1[0]['boxes_3d'][index1]
                bbox2 = bbox_results_t2[0]['boxes_3d'][index2]

                iou = bbox_results_t1[0]['boxes_3d'][0].overlaps(bbox1, bbox2)
                iou_sum += iou.sum(axis=[0, 1])
            return len(bbox_results_t1[0]['boxes_3d']) + len(bbox_results_t2[0]['boxes_3d']) - iou_sum.item()

    def cam_to_world(self, bbox_result):
        coor = self.coor_now.split(' ')
        coor = [int(i) * 0.25 for i in coor]
        angle = self.angle_list.index(self.angle_now)
        bbox_result[0]['boxes_3d'].tensor[:, 0] += coor[0]
        bbox_result[0]['boxes_3d'].tensor[:, 1] += coor[1]
        bbox_result[0]['boxes_3d'].tensor[:, 6] -= angle * (3.1415926 / 4)
        return bbox_result

    def reward_cal(self, path):
        points = []
        points.append(self.read_pc(path))
        self.bbox_results_t2 = self.votenet.simple_test(points, self.meta, None, True)
        obsret = self.votenet.backbone.fp_ret
        obsret = torch.cat((obsret['fp_xyz'][0], obsret['fp_features'][0].permute(0, 2, 1)), axis=2)
        self.bbox_results_t2 = self.cam_to_world(self.bbox_results_t2)

        reward = self.iou_cal(self.bbox_results_t1, self.bbox_results_t2)

        self.bbox_results_t1 = self.bbox_results_t2

        return float(reward), obsret
