_base_ = ['../../_base_/default_runtime.py']

# common setting
num_keypoints = 4
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
        symmetry_pairs=[(0, 2), (1, 3)],
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
        0: {'name': 'lefttop', 'id': 0, 'color': [255, 0, 0], 'type': '', 'swap': 'righttop'},
        1: {'name': 'leftbottom', 'id': 1, 'color': [0, 255, 0], 'type': '', 'swap': 'rightbottom'},
        2: {'name': 'rightbottom', 'id': 2, 'color': [0, 0, 255], 'type': '', 'swap': 'leftbottom'},
        3: {'name': 'righttop', 'id': 3, 'color': [255, 255, 0], 'type': '', 'swap': 'lefttop'},
    },
    skeleton_info={
        0: {'link': ('lefttop', 'leftbottom'), 'id': 0, 'color': [51, 153, 255]},
        1: {'link': ('leftbottom', 'rightbottom'), 'id': 1, 'color': [51, 153, 255]},
        2: {'link': ('rightbottom', 'righttop'), 'id': 2, 'color': [51, 153, 255]},
        3: {'link': ('righttop', 'lefttop'), 'id': 3, 'color': [51, 153, 255]},
    },
    joint_weights=[1.] * 4,
    sigmas=[0.05, 0.05, 0.05, 0.05],
)

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.3,
        shift_prob=0.5,
        scale_factor=[0.8, 1.4],
        rotate_factor=80),
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
         ann_file='data/pose-dataset/BakingRefine/annotations/train260812.json'),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='MergeCategory', merge_groups=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']],
         ann_file='data/pose-dataset/BakingRefine/annotations/val260812e.json'),
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
    symmetry_pairs=[(0, 2), (1, 3)],
    symmetry_categories=[0, 1, 2],
    category_merge=[['Oven_TopHandle', 'Oven_BottomHandle'],['Oven_TopInner', 'Oven_BottomInner']])
test_evaluator = val_evaluator