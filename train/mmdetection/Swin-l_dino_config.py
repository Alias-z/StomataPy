# -------------------------------- Args shortcut --------------------------------
resume = None
load_from = 'https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
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
output_dir = 'SwinTransformer_Large_Dino_Gmax'
work_dir = 'Models//' + output_dir
wandb_project = 'StomataPy'

batch_size = 2
n_gpus = 4
num_workers = 16
original_batch_size = 16  # 2
original_lr = 0.0001
original_n_gpus = 8
lr = original_lr * (n_gpus / original_n_gpus) * (batch_size / original_batch_size) * 10
auto_scale_lr = dict(base_batch_size=16, enable=False)

ReduceOnPlateauLR_patience = 50
early_stopping_patience = 150
max_epochs = 500
warmup_epochs = 30

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
    dict(type='RandomFlip', prob=0.5),
    dict(type='CutOut', n_holes=5, cutout_ratio=(0.025, 0.05)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='FixShapeResize',
        width=2000,
        height=1500,
        pad_val=0,
        keep_ratio=True,
        clip_object_border=True,
        interpolation='lanczos'
    ),
    dict(type='GeomTransform', prob=0.5, img_border_value=(0, 0, 0), interpolation='lanczos'),
    dict(type='RandomCrop', crop_size=(0.4, 0.4), crop_type='relative', recompute_bbox=True, allow_negative_crop=False, bbox_clip_border=True),  # 0.55, 0.65
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_ignore_flags', 'gt_bboxes_labels'],
            min_visibility=0.01,
            filter_lost_elements=True),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
        skip_img_without_anno=True)
]

train_pipeline = [
    dict(
        type='CachedMixUp',
        img_scale=(crop_size[0] * 3, crop_size[1] * 3),
        ratio_range=(0.8, 1.2),
        flip_ratio=0.5,
        pad_val=0,
        max_iters=15,
        bbox_clip_border=True,
        max_cached_images=30,
        random_pop=True,
        prob=0.05
    ),
    dict(
        type='Mosaic',
        img_scale=(crop_size[0], crop_size[1]),
        center_ratio_range=(0.8, 1.2),
        pad_val=0.0,
        bbox_clip_border=True,
        prob=1
    ),
    # dict(type='MinIoURandomCrop', min_ious=(0.7, 0.8, 0.9, 1.0), min_crop_size=0.1, bbox_clip_border=True),
    dict(type='RandomCrop', crop_size=(0.3, 0.7), crop_type='relative_range', recompute_bbox=True, allow_negative_crop=False, bbox_clip_border=True),
    dict(type='RandomChoiceResize', scales=[512, 576, 640, 704, 768], resize_type='ResizeShortestEdge', max_size=768),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10)),
    dict(type='PackDetInputs', meta_keys=('img_path', 'img_shape', 'img', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='FixShapeResize',
    #     width=2000,
    #     height=1500,
    #     pad_val=0,
    #     keep_ratio=True,
    #     clip_object_border=True,
    #     interpolation='lanczos'
    # ),
    dict(type='RandomChoiceResize', scales=[1080, 1260, 1440, 1620, 1800, 1980, 2160, 2340], resize_type='ResizeShortestEdge', max_size=2340),
    dict(type='RandomFlip', prob=0.5),
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
    dict(type='EMAHook')  # not allowed during FSDP training
]

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')

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
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),

    # Backbone ResNet-50
    # backbone=dict(
    #    type='ResNet',
    #    depth=50,
    #    num_stages=4,
    #    out_indices=(1, 2, 3),
    #    frozen_stages=1,
    #    norm_cfg=dict(type='BN', requires_grad=False),
    #    norm_eval=True,
    #    style='pytorch',
    #    init_cfg=dict(
    #        type='Pretrained', checkpoint='torchvision://resnet50')),

    # Backbone SwinTransformer-Large
    # Params (M): 196.74; Flops (G): 100.04
    # https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/swin_transformer.py
    num_feature_levels=num_feature_levels,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,  # mmdet only: ratio of mlp hidden dim to embedding dim
        qkv_bias=True,  # mmdet only: add a learnable bias to q, k, v
        qk_scale=None,  # mmdet only: override default qk scale of head_dim ** -0.5 if set
        drop_rate=0.0,
        attn_drop_rate=0.0,  # mmdet only: dropout ratio of attention weight
        drop_path_rate=0.2,
        patch_norm=True,  # mmdet only: If add a norm layer for patch embed and patch merging
        out_indices=(0, 1, 2, 3),
        with_cp=with_cp,
        convert_weights=True,  # mmdet only: whether the pre-trained model is from the original repo. We may need to convert some keys to make it compatible.
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
        )
    ),

    # Backbone SwinTransformerv2-Large
    # Params (M): 196.75; Flops (G): 76.20
    # https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/swin_transformer_v2.py
    # backbone=dict(
    #     type='mmpretrain.SwinTransformerV2',
    #     arch='large',
    #     img_size=384,
    #     window_size=[24, 24, 24, 12],
    #     pretrained_window_sizes=[12, 12, 12, 6],
    #     drop_path_rate=0.3,
    #     out_indices=(0, 1, 2, 3),
    #     pad_small_map=True,
    #     with_cp=with_cp,
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth',
    #         prefix='backbone.'  # The prefix in the keys will be removed so that these weights can be normally loaded.
    #     )
    # ),

    # Backbone ConvNeXt-V2-Large
    # Params (M): 197.96; Flops (G): 101.10
    # https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/convnext.py
    # backbone=dict(
    #     type='mmpretrain.ConvNeXt',
    #     arch='large',
    #     drop_path_rate=0.4,
    #     layer_scale_init_value=0.,  # Disable layer scale when using GRN
    #     gap_before_final_norm=False,
    #     use_grn=True,  # V2 uses GRN
    #     out_indices=(0, 1, 2, 3),
    #     with_cp=with_cp,
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth',
    #         prefix='backbone.'  # The prefix in the keys will be removed so that these weights can be normally loaded.
    #     )
    # ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],  # SwinTransformer-Large
        # in_channels=[128, 256, 512, 1024],  # SwinTransformer-Large
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_feature_levels
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_feature_levels, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)  # 0.1 for DeformDETR
        )
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_feature_levels, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
        ),  # 0.1 for DeformDETR
        post_norm_cfg=None
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20  # 10000 for DeformDETR
    ),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    dn_cfg=dict(
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5
    ),
    # training and testing settings
    test_cfg=dict(max_per_img=300),  # 100 for DeformDETR
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', box_format='xywh', weight=5.0),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
            type='HungarianAssigner')
    )
)


# --------------------------------  No need to code below (Runtime) --------------------------------

default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'

randomness = dict(seed=42, deterministic=False)

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
