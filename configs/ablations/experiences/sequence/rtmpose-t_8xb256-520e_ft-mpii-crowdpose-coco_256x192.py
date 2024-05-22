_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(
    type='ContinualTrainingLoop',
    # 420 for experience 1, 50 each for experience 2 and 3
    max_epochs_per_experience=[0, 50, 50],
    val_interval=10,
)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
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
        # use cosine lr from 210 to 420 epoch
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
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# data preprocessor settings
data_preprocessor = dict(
    type='PoseDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True)

# backbone settings
backbone = dict(
    _scope_='mmdet',
    type='CSPNeXt',
    arch='P5',
    expand_ratio=0.5,
    deepen_factor=0.167,
    widen_factor=0.375,
    out_indices=(4, ),
    channel_attention=True,
    norm_cfg=dict(type='SyncBN'),
    act_cfg=dict(type='SiLU'),
    init_cfg=dict(
        type='Pretrained',
        prefix='backbone.',
        checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
        'rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth'  # noqa
    ))

# heads settings
h21 = dict(  # h21: predict 21 COCO+MPII points
    type='RTMCCHead',
    in_channels=384,
    out_channels=21,
    input_size=codec['input_size'],
    in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
    simcc_split_ratio=codec['simcc_split_ratio'],
    final_layer_kernel_size=7,
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
    decoder=codec)

# model settings
model = dict(  # initial model
    type='TopdownPoseEstimator',
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    head=h21,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/rtmpose-t/task-1/rtmpose-t_8xb64-210e_mpii-256x256/best_mpii_PCK_epoch_210.pth'),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'

backend_args = dict(backend='local')

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
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
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
    dict(type='PackPoseInputs')
]

# mappings
coco2unified = [(i, i) for i in range(17)]
mpii2unified = [
    (13, 5), (12, 6), (14, 7), (11, 8), (15, 9), (10, 10),
    (3, 11), (2, 12), (4, 13), (1, 14), (5, 15), (0, 16),
    (6, 20), (7, 19), (8, 18), (9, 17),
]
crowdpose2unified = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (6, 11), (7, 12), (8, 13), (9, 14), (10, 15),
    (11, 16), (12, 20), (13, 19)
]
unified2mpii = [(dst, src) for src, dst in mpii2unified]

# datasets
dataset_coco_val = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_val2017.json',
    bbox_file=f'{data_root}/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    data_prefix=dict(img='coco/val2017/'),
    test_mode=True,
     pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=coco2unified)
    ],
)
dataset_mpii_val = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_val.json',
    headbox_file=f'{data_root}/mpii/annotations/mpii_gt_val.mat',
    data_prefix=dict(img='mpii/images/'),
    test_mode=True,
    pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=mpii2unified)
    ],
)
dataset_crowdpose_val = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
    bbox_file=f'{data_root}/crowdpose/annotations/det_for_crowd_test_0.1_0.5.json',
    data_prefix=dict(img='crowdpose/images/'),
    test_mode=True,
    pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=crowdpose2unified)
    ],
)
dataset_crowdpose_train = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=crowdpose2unified)
    ],
)
dataset_mpii_train = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=mpii2unified)
    ],
)
dataset_coco_train = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=[
        dict(type='KeypointConverter',
             num_keypoints=21,
             mapping=coco2unified)
    ],
)

# experiences
dataset_train1 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_mpii_train,
    ],
    pipeline=train_pipeline,
    test_mode=False,
)
dataset_val1 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_mpii_val,
    ],
    pipeline=val_pipeline,
)
dataset_train2 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_crowdpose_train,
    ],
    pipeline=train_pipeline,
    test_mode=False,
)
dataset_val2 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_crowdpose_val,
        dataset_mpii_val,
    ],
    pipeline=val_pipeline,
    test_mode=True,
)
dataset_train3 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_coco_train,
    ],
    pipeline=train_pipeline,
    test_mode=False,
)
dataset_val3 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dataset_coco_val,
        dataset_crowdpose_val,
        dataset_mpii_val,
    ],
    pipeline=val_pipeline,
    test_mode=True,
)

# data loaders
batch_size = 256
num_workers = 32
train_dataloader1 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_train1,
)
train_dataloader2 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_train2,
)
train_dataloader3 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_train3,
)
val_dataloader1 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dataset_val1,
)
val_dataloader2 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dataset_val2,
)
val_dataloader3 = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dataset_val3,
)

train_dataloaders = [train_dataloader1, train_dataloader2, train_dataloader3]
val_dataloaders = [val_dataloader1, val_dataloader2, val_dataloader3]
test_dataloaders = [val_dataloader1, val_dataloader2, val_dataloader3]

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='mpii/PCK', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),

    # no explicit forgetting mitigation strategy
    # this is equivalent to the fine-tuning baseline
    # which is lower bound of the performance

    # switch model before the second experience
    dict(type='DynamicSnapshotPlugin',
         mode='last'),
]

# converters
coco_converter = dict(type='KeypointConverter',
                      num_keypoints=17,
                      mapping=[(i, i) for i in range(17)])
crowdpose_converter = dict(type='KeypointConverter',
                           num_keypoints=14,
                           mapping=[(dest, src)
                                    for src, dest in crowdpose2unified])
mpii_converter = dict(type='KeypointConverter2',
                      num_keypoints=16,
                      mapping=unified2mpii)

# metrics
mpii_metric = dict(type='MpiiPCKAccuracy', prefix='mpii')
coco_metric = dict(type='CocoMetric',
                   ann_file=f'{data_root}/coco/annotations/person_keypoints_val2017.json',
                   pred_converter=coco_converter)
crowdpose_metric = dict(type='CocoMetric',
                        ann_file=f'{data_root}/crowdpose/annotations/mmpose_crowdpose_test.json',
                        use_area=False,
                        iou_type='keypoints_crowd',
                        prefix='crowdpose',
                        pred_converter=crowdpose_converter)

# evaluators
val_evaluators = [
    # use CocoMetric for experience 1 (data and model have same keypoint set)
    dict(type='UnifiedDatasetEvaluator',
         metrics=[mpii_metric],
         datasets=[dataset_mpii_val],
         converters=[mpii_converter]),

    # experience 2 (model predicts 21 keypoints, validate on 16, 17 keypoints)
    dict(type='UnifiedDatasetEvaluator',
         metrics=[mpii_metric, crowdpose_metric],
         datasets=[dataset_mpii_val, dataset_crowdpose_val],
         converters=[mpii_converter, None]),

    # experience 3 (model predicts 21 keypoints, validate on 14, 16, 17 keypoints)
    dict(type='UnifiedDatasetEvaluator',
         metrics=[mpii_metric, crowdpose_metric, coco_metric],
         datasets=[dataset_mpii_val, dataset_crowdpose_val, dataset_coco_val],
         converters=[mpii_converter, None, None]),
]
test_evaluators = val_evaluators
