# -------------------------------- Args shortcut --------------------------------
resume = None
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth'
val_interval = 1
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

fp16 = dict(loss_scale='dynamic')
with_cp = True  # for FSDP: the checkpoint needs to be controlled by the checkpoint_check_fn.
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)

classes = ('stomatal complex',)
num_classes = len(classes)
num_feature_levels = 5  # Feature Pyramid Network (FPN), "num_feature_levels" refers to the number of feature maps that the network generates.

dataset_type = 'CocoDataset'
data_root = 'Stomata_detection//'
output_dir = 'Swin-L_DDQ_Yates'
work_dir = 'Models//' + output_dir
wandb_project = 'StomataPy'

batch_size = 4
n_gpus = 4
num_workers = 16
original_batch_size = 16  # 2
original_lr = 0.0002
original_n_gpus = 8
lr = original_lr * (n_gpus / original_n_gpus) * (batch_size / original_batch_size) * 10
auto_scale_lr = dict(base_batch_size=16, enable=False)

ReduceOnPlateauLR_patience = 50
early_stopping_patience = 150
max_epochs = 500
warmup_epochs = 30

image_size = (4320, 1620)
crop_size = (512, 512)

# -------------------------------- Data augmentation --------------------------------

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='ElasticTransform', alpha=20, sigma=15,
                 interpolation=4, border_mode=0, mask_value=(0, 0, 0),
                 approximate=True, same_dxdy=True, p=0.5),
            dict(type='ElasticTransform', alpha=40, sigma=15,
                 interpolation=4, border_mode=0, mask_value=(0, 0, 0),
                 approximate=True, same_dxdy=False, p=0.5),
        ],
        p=0.25),
    # dict(type='RandomScale', scale_limit=0.15, interpolation=4, always_apply=True),
    dict(type='AdvancedBlur', p=0.05)
]


load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='CutOut', n_holes=5, cutout_ratio=(0.025, 0.05)),
    dict(
        type='RandomChoiceResize',
        scales=[int(image_size[1] * x * 0.1) for x in range(9, 11)],
        resize_type="ResizeShortestEdge",
        max_size=image_size[1] * 2,
    ),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=(image_size[0] // 2, image_size[1] // 2),
        recompute_bbox=True,
        allow_negative_crop=False,
        bbox_clip_border=True
    ),
    # dict(type='YOLOXHSVRandomAug'),
    dict(
        type='RandomAffine',
        max_rotate_degree=10,
        max_translate_ratio=0.0,
        scaling_ratio_range=(1.0, 1.0),
        max_shear_degree=2,
        border=(0, 0),
        border_val=(114, 114, 114),
        bbox_clip_border=True
    ),
    # dict(
    #     type='FixShapeResize',
    #     width=2000,
    #     height=1500,
    #     pad_val=0,
    #     keep_ratio=True,
    #     clip_object_border=True,
    #     interpolation='lanczos'
    # ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='GeomTransform', prob=0.5, img_border_value=(0, 0, 0), interpolation='lanczos'),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_ignore_flags', 'gt_bboxes_labels'],
    #         min_visibility=0.01,
    #         filter_lost_elements=True),
    #     keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
    #     skip_img_without_anno=True
    # )
]

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(image_size[0], image_size[1]),
        center_ratio_range=(0.8, 1.2),
        pad_val=114.0,
        bbox_clip_border=True,
        prob=1
    ),
    dict(
        type='CachedMixUp',
        img_scale=(crop_size[0] * 3, crop_size[1] * 3),
        ratio_range=(1, 1),
        flip_ratio=0.5,
        pad_val=0,
        max_iters=15,
        bbox_clip_border=True,
        max_cached_images=30,
        random_pop=True,
        prob=0
    ),
    # dict(type='MinIoURandomCrop', min_ious=(0.7, 0.8, 0.9, 1.0), min_crop_size=0.1, bbox_clip_border=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute',
    #     crop_size=(image_size[0] // 2, image_size[1] // 2),
    #     recompute_bbox=True,
    #     allow_negative_crop=False,
    #     bbox_clip_border=True
    # ),
    dict(
        type='RandomCrop',
        crop_type='relative',
        crop_size=(0.125, 0.125),
        recompute_bbox=True,
        allow_negative_crop=False,
        bbox_clip_border=True),
    # dict(type='RandomChoiceResize', scales=[512, 576, 640, 704, 768], resize_type='ResizeShortestEdge', max_size=768),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-4, 1e-4)),
    dict(type='PackDetInputs', meta_keys=('img_path', 'img_shape', 'img', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(image_size[0], image_size[1]), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')
    )
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=scale, keep_ratio=True)
                for scale in [(1080, 1080), (960, 960)]
            ],
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.6), max_per_img=100)
)


# -------------------------------- Dataloader --------------------------------

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/COCO.json',
        data_prefix=dict(img='train/'),
        pipeline=load_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=batch_size,  # original 2
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=batch_size // 2,  # original 2
    num_workers=num_workers,
    drop_last=False,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/COCO.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=None)
)

test_dataloader = val_dataloader


# -------------------------------- Evaluator --------------------------------

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/val/COCO.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = val_evaluator

# -------------------------------- Parameter scheduler --------------------------------

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)

val_cfg = dict(type='ValLoop')

test_cfg = val_cfg

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=9999999999,
        by_epoch=True,
        save_last=False,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=1),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=early_stopping_patience),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1000))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MemoryProfilerHook', interval=200),
    dict(type='CheckInvalidLossHook', interval=200),
    # dict(type='EMAHook')  # not allowed during FSDP training
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        end_factor=1.0,
        begin=0,
        end=warmup_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
        verbose=False),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - warmup_epochs,
        eta_min=lr * 1e-05,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
        verbose=False,
        eta_min_ratio=None),
    dict(
        type='ReduceOnPlateauLR',
        monitor='coco/bbox_mAP',
        rule='greater',
        factor=0.75,
        patience=ReduceOnPlateauLR_patience,
        by_epoch=True,
        verbose=False)
]

# -------------------------------- Visualization backend --------------------------------

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs={'project': wandb_project, 'name': output_dir})
    ],
    name='visualizer')


# -------------------------------- Model config --------------------------------

model = dict(
    type='DDQDETR',
    num_queries=900,  # num_matching_queries
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=with_cp,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    # encoder class name: DeformableDetrTransformerEncoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    # decoder class name: DDQTransformerDecoder
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DDQDETRHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))


# --------------------------------  No need to code below (Runtime) --------------------------------

default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'

randomness = dict(seed=42, deterministic=False)

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
