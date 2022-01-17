_base_ = [
    '../_base_/datasets/active-R3AD-3d.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        in_channels=7,
    ),
    bbox_head=dict(
        num_classes=12,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=12,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [1.1, 0.5, 0.5],
                [0.7, 1.2, 0.7],
                [1.8, 1.0, 1.0],
                [2.3, 2.3, 0.6],
                [1.6, 1.0, 1.0],
                [0.6, 1.3, 0.7],
                [0.2, 0.4, 1.7],
                [0.7, 0.4, 1.6],
                [0.1, 0.1, 0.2],
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.3],
                [0.4, 0.4, 0.8]
            ]),
    ))
