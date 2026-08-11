_base_ = ['../../_base_/default_runtime.py']

# common setting
num_keypoints = 12
input_size = (192, 256)

# runtime
max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3
train_batch_size = 256
val_batch_size = 64

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
        )),
    head=dict(
        type='SymmetryMatchRTMCCHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        symmetry_categories=[0, 1, 2],
        symmetry_pairs=[(0, 2), (1, 3), (4, 6), (5, 7)],
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))


# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/pose-dataset/BakingRefine/'

backend_args = dict(backend='local')

# BakingRefine 4-keypoint metainfo
dataset_info = dict(
    dataset_name='coco',
    keypoint_info={
        0: {'name': 'lefttop1', 'id': 0, 'color': [255, 0, 0], 'type': '', 'swap': 'righttop1'},
        1: {'name': 'leftbottom1', 'id': 1, 'color': [0, 255, 0], 'type': '', 'swap': 'rightbottom1'},
        2: {'name': 'rightbottom1', 'id': 2, 'color': [0, 0, 255], 'type': '', 'swap': 'leftbottom1'},
        3: {'name': 'righttop1', 'id': 3, 'color': [255, 255, 0], 'type': '', 'swap': 'lefttop1'},
        4: {'name': 'lefttop2', 'id': 4, 'color': [255, 0, 0], 'type': '', 'swap': 'righttop2'},
        5: {'name': 'leftbottom2', 'id': 5, 'color': [0, 255, 0], 'type': '', 'swap': 'rightbottom2'},
        6: {'name': 'rightbottom2', 'id': 6, 'color': [0, 0, 255], 'type': '', 'swap': 'leftbottom2'},
        7: {'name': 'righttop2', 'id': 7, 'color': [255, 255, 0], 'type': '', 'swap': 'lefttop2'},
        8: {'name': 'lefttop3', 'id': 8, 'color': [255, 0, 0], 'type': '', 'swap': 'righttop3'},
        9: {'name': 'leftbottom3', 'id': 9, 'color': [0, 255, 0], 'type': '', 'swap': 'rightbottom3'},
        10: {'name': 'rightbottom3', 'id': 10, 'color': [0, 0, 255], 'type': '', 'swap': 'leftbottom3'},
        11: {'name': 'righttop3', 'id': 11, 'color': [255, 255, 0], 'type': '', 'swap': 'lefttop3'},
    },
    skeleton_info={
        0: {'link': ('lefttop1', 'leftbottom1'), 'id': 0, 'color': [51, 153, 255]},
        1: {'link': ('leftbottom1', 'rightbottom1'), 'id': 1, 'color': [51, 153, 255]},
        2: {'link': ('rightbottom1', 'righttop1'), 'id': 2, 'color': [51, 153, 255]},
        3: {'link': ('righttop1', 'lefttop1'), 'id': 3, 'color': [51, 153, 255]},
        4: {'link': ('lefttop2', 'leftbottom2'), 'id': 4, 'color': [51, 153, 255]},
        5: {'link': ('leftbottom2', 'rightbottom2'), 'id': 5, 'color': [51, 153, 255]},
        6: {'link': ('rightbottom2', 'righttop2'), 'id': 6, 'color': [51, 153, 255]},
        7: {'link': ('righttop2', 'lefttop2'), 'id': 7, 'color': [51, 153, 255]},
        8: {'link': ('lefttop3', 'leftbottom3'), 'id': 8, 'color': [51, 153, 255]},
        9: {'link': ('leftbottom3', 'rightbottom3'), 'id': 9, 'color': [51, 153, 255]},
        10: {'link': ('rightbottom3', 'righttop3'), 'id': 10, 'color': [51, 153, 255]},
        11: {'link': ('righttop3', 'lefttop3'), 'id': 11, 'color': [51, 153, 255]},
    },
    joint_weights=[1.] * num_keypoints,
    sigmas=[0.05] * num_keypoints,
)

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='MergeCategory', merge_groups=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']],
         ann_file='data/pose-dataset/BakingRefine/annotations/train260807.json'),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='MergeCategory', merge_groups=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']],
         ann_file='data/pose-dataset/BakingRefine/annotations/val260807.json'),
    dict(type='PackPoseInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='MergeCategory', merge_groups=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']],
         ann_file='data/pose-dataset/BakingRefine/annotations/train260807.json'),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train260807.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        metainfo=dataset_info,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val260807.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dataset_info,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='SymmetryMatchCocoMetric',
    ann_file=data_root + 'annotations/val260807.json',
    symmetry_pairs=[(0, 2), (1, 3), (4, 6), (5, 7)],
    symmetry_categories=[0, 1, 2],
    category_merge=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']])
test_evaluator = val_evaluator