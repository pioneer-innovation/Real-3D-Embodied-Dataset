import torch
from mmdet3d.models import build_backbone
import open3d as o3d
import numpy as np
import os


class VoteQN2(torch.nn.Module):
    def __init__(self, output_shape):
        """
        A Convolution Neural Network (CNN) class to approximate functions with visual/image inputs

        :param input_shape:  Shape/dimension of the input image. Assumed to be resized to C x 84 x 84
        :param output_shape: Shape/dimension of the output.
        :param device: The device (cpu or cuda) that the CNN should use to store the inputs for the forward pass
        """
        #  input_shape: C x 84 x 84
        super(VoteQN2, self).__init__()

        self.out_shape = output_shape

        neck = dict(
            type='PointNet2SASSG',
            in_channels=259,
            num_points=(64, 16, 1),
            radius=(1.2, 3.6, 3.6),
            num_samples=(16, 16, 16),
            sa_channels=((64, 64, 128), (128, 128, 256), (128, 64, output_shape)),
            fp_channels=(),
            norm_cfg=dict(type='BN2d'),
            sa_cfg=dict(
                type='PointSAModule',
                pool_mod='max',
                use_xyz=True,
                normalize_xyz=True))

        self.neck = build_backbone(neck)

    def forward(self, y, state=None, info={}):
        # obs = torch.from_numpy(y['observation']).cuda()
        # action_mask = torch.from_numpy(y['mask']).cuda()
        obs = y['observation'].cuda()
        action_mask = y['mask'].cuda()
        y = self.neck(obs)['fp_features'][0].squeeze()
        y = y * action_mask
        return y, y
