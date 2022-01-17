import torch
from mmdet3d.models import build_backbone
import open3d as o3d
import numpy as np
import os


class RNNVoteQN2(torch.nn.Module):
    def __init__(self, output_shape):
        """
        A Convolution Neural Network (CNN) class to approximate functions with visual/image inputs

        :param input_shape:  Shape/dimension of the input image. Assumed to be resized to C x 84 x 84
        :param output_shape: Shape/dimension of the output.
        :param device: The device (cpu or cuda) that the CNN should use to store the inputs for the forward pass
        """
        #  input_shape: C x 84 x 84
        super(RNNVoteQN2, self).__init__()

        self.out_shape = output_shape

        neck = dict(
            type='PointNet2SASSG',
            in_channels=259,
            num_points=(64, 16, 1),
            radius=(1.2, 3.6, 3.6),
            num_samples=(16, 16, 16),
            sa_channels=((64, 64, 128), (128, 128, 256), (128, 64, 64)),
            fp_channels=(),
            norm_cfg=dict(type='BN2d'),
            sa_cfg=dict(
                type='PointSAModule',
                pool_mod='max',
                use_xyz=True,
                normalize_xyz=True))

        self.neck = build_backbone(neck)
        self.GRU = torch.nn.GRUCell(
            input_size=64,
            hidden_size=64
        )
        self.fc = torch.nn.Linear(64,output_shape)

    def forward(self, y, state=None, info={}):
        obs = y['observation'].cuda()
        action_mask = y['mask'].cuda()
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
            action_mask = action_mask.unsqueeze(1)

        for i in range(obs.shape[1]):
            y = self.neck(obs[:,i,:,:])['fp_features'][0].squeeze(2)
            if state is None:
                state = torch.randn(obs.shape[0], 64).cuda()
                state = self.GRU(y,state)
            else:
                state = state.cuda()
                state = self.GRU(y,state)

        y = self.fc(state)
        y = y * action_mask[:,-1,:]
        return y, state.detach().cpu() # state_list[-1]
