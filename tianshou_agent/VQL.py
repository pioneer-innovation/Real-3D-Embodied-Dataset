import torch, numpy as np
from torch import nn
from mmdet3d.models import build_backbone
import open3d as o3d


class VoteQN(nn.Module):
    def __init__(self, output_shape, model_path):
        super(VoteQN, self).__init__()

        self.votenet = torch.load(model_path)
        self.backbone = self.votenet.backbone
        for param in self.backbone.parameters():
            param.requeires_grad = False
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

    def forward(self, s, state=None, info={}):
        obs = s.observation
        points = []
        for i in range(len(obs)):
            points.append(self.read_pc(obs[i]))
        points_cat = torch.stack(points)
        x = self.backbone(points_cat)
        xyz = x['fp_xyz'][0]
        feat = torch.transpose(x['fp_features'][0], 1, 2)
        y = torch.cat((xyz, feat), 2)
        y = self.neck(y)['fp_features'][0]
        y = y.squeeze(-1)
        return y,None

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