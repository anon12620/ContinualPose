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
h17 = dict(  # h17: predict 17 COCO keypoints
    type='RTMCCHead',
    in_channels=384,
    out_channels=17,
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
    head=h17,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/baselines/st/rtmpose-t_8xb256-420e_st1-coco-256x192/best_coco_AP_epoch_420.pth'),
    test_cfg=dict(flip_test=True))
model2 = dict(  # model after the first experience (stage 2 and beyond)
    type='TopdownPoseEstimator',
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    head=h21,
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

# experience 1 datasets (COCO)
dataset_train1 = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=train_pipeline,
)
dataset_val1 = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_val2017.json',
    bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    data_prefix=dict(img='coco/val2017/'),
    test_mode=True,
    pipeline=val_pipeline,
)

# experience 2 datasets (MPII mapped to 21 keypoints)
mpii2unified = [
    (13, 5), (12, 6), (14, 7), (11, 8), (15, 9), (10, 10),
    (3, 11), (2, 12), (4, 13), (1, 14), (5, 15), (0, 16),
    (6, 20), (7, 19), (8, 18), (9, 17),
]
dataset_train2 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[dict(
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
    )],  # train only on experience 2 dataset
    pipeline=train_pipeline,
    test_mode=False,
)
dataset_val2 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[
        dict(
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
        ),
        dict(
            type='CocoDataset',
            data_root=data_root,
            data_mode=data_mode,
            ann_file='coco/annotations/person_keypoints_val2017.json',
            bbox_file=f'{data_root}/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
            data_prefix=dict(img='coco/val2017/'),
            test_mode=True,
        )],  # validate on both experience 1 and 2 datasets
    pipeline=val_pipeline,
    test_mode=True,
)

# experience 3 datasets (CrowdPose mapped to 21 keypoints)
crowdpose2unified = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (6, 11), (7, 12), (8, 13), (9, 14), (10, 15),
    (11, 16), (12, 20), (13, 19)
]
dataset_train3 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[dict(
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
    )],  # train only on experience 2 dataset
    pipeline=train_pipeline,
    test_mode=False,
)
dataset_val3 = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/datasets/coco21.py'),
    datasets=[  # validate on all 3 datasets
        dict(  # experience 3 dataset
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
        ),
        dict(  # experience 2 dataset
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
        ),
        dict(
            type='CocoDataset',
            data_root=data_root,
            data_mode=data_mode,
            ann_file='coco/annotations/person_keypoints_val2017.json',
            bbox_file=f'{data_root}/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
            data_prefix=dict(img='coco/val2017/'),
            test_mode=True,
        )],  # experience 1 dataset
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
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

unified2coco = [(i, i) for i in range(17)]
unified2mpii = [(dst, src) for src, dst in mpii2unified]

custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),

    # use LWFPlugin to distill knowledge from experience 1 to 2
    dict(
        type='LWFPlugin',
        temperature=2.0,
        lambda_lwf=0.8,
        converters={1: dict(num_keypoints=17, mapping=unified2coco)},
    ),

    # switch model before the second experience
    dict(type='DynamicSnapshotPlugin',
         mode='last',
         model_cfgs={1: model2}),
]

# evaluators
val_evaluators = [
    # use CocoMetric for experience 1 (data and model have same keypoint set)
    dict(type='CocoMetric',
         ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json'),

    # experience 2 (model predicts 21 keypoints, validate on 16, 17 keypoints)
    dict(type='UnifiedDatasetEvaluator',
         metrics=[
             dict(type='MpiiPCKAccuracy',
                  prefix='mpii'),

             dict(type='CocoMetric',
                  ann_file=f'{data_root}/coco/annotations/person_keypoints_val2017.json',
                  pred_converter=dict(  # map model output to 17 COCO keypoints
                      type='KeypointConverter',
                      num_keypoints=17,
                      mapping=unified2coco,
                  )),
         ],
         datasets=[
             dict(type='MpiiDataset',
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
                  ]), dataset_val1],

         converters=[
             dict(type='KeypointConverter2',
                  num_keypoints=16,
                  mapping=unified2mpii),
             None
         ]),

    # experience 3 (model predicts 21 keypoints, validate on 14, 16, 17 keypoints)
    dict(type='UnifiedDatasetEvaluator',
         metrics=[
             dict(type='CocoMetric',
                  ann_file=f'{data_root}/crowdpose/annotations/mmpose_crowdpose_test.json',
                  use_area=False,
                  iou_type='keypoints_crowd',
                  prefix='crowdpose',
                  pred_converter=dict(  # map model output to 14 CrowdPose keypoints
                      type='KeypointConverter',
                      num_keypoints=14,
                      mapping=[(dest, src) for src, dest in crowdpose2unified],
                  )),

             dict(type='MpiiPCKAccuracy',
                  prefix='mpii'),

             dict(type='CocoMetric',
                  ann_file=f'{data_root}/coco/annotations/person_keypoints_val2017.json',
                  pred_converter=dict(  # map model output to 17 COCO keypoints
                      type='KeypointConverter',
                      num_keypoints=17,
                      mapping=unified2coco,
                  )),
         ],
         datasets=[
             dict(type='CrowdPoseDataset',
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
                  ]),
             dict(type='MpiiDataset',
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
                  ]),
             dataset_val1],

         converters=[
             None,
             dict(type='KeypointConverter2',
                  num_keypoints=16,
                  mapping=unified2mpii),
             None
         ]),
]
test_evaluators = val_evaluators
