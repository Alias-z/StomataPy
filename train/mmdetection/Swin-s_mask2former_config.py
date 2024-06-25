resume = None
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'
load_from = 'Models/Swin-S_Mask2Former_Hvulgare0906/best_coco_segm_mAP_epoch_297.pth'
val_interval = 1
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

fp16 = dict(loss_scale='dynamic')
with_cp = True  # for FSDP: the checkpoint needs to be controlled by the checkpoint_check_fn.
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)

# classes = ('stomatal complex', 'pavement cell')
classes = ('stomatal complex', )
num_stuff_classes = 0
num_things_classes = len(classes)
num_classes = num_things_classes + num_stuff_classes

dataset_type = 'CocoDataset'
data_root = 'Stomata_detection//'
# data_root = 'Epidermal_segmentation//'
output_dir = 'Swin-S_Mask2Former_Hvulgare1206'
work_dir = 'Models//' + output_dir
wandb_project = 'StomataPy'

train_ann_file = 'train_sahi/sahi_coco.json'
val_ann_file = 'val_sahi/sahi_coco.json'

batch_size = 2
n_gpus = 4
num_workers = 16
original_batch_size = 16  # 2
original_lr = 0.0001
original_n_gpus = 8
lr = original_lr * (n_gpus / original_n_gpus) * (batch_size / original_batch_size)
auto_scale_lr = dict(base_batch_size=16, enable=False)


ReduceOnPlateauLR_patience = 50
early_stopping_patience = 150
max_epochs = 300
warmup_epochs = 30


depths = [2, 2, 18, 2]

# Swin-small
embed_dims = 96
num_heads = [3, 6, 12, 24]
checkpoint = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

# # Swin-base
# embed_dims = 128
# num_heads = [4, 8, 16, 32]
# checkpoint = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

# Swin-large
# embed_dims = 192
# num_heads = [6, 12, 24, 48]
# checkpoint = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

crop_size = (1280, 1024)
image_size = (1024, 1024)


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
    dict(type='AdvancedBlur', p=0.05)
]

load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='CutOut', n_holes=5, cutout_ratio=(0.025, 0.05)),
    # dict(
    #     type='RandomChoiceResize',
    #     scales=[int(image_size[1] * x * 0.1) for x in range(9, 11)],
    #     resize_type="ResizeShortestEdge",
    #     max_size=image_size[1] * 2,
    # ),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute',
    #     crop_size=(image_size[0], image_size[1]),
    #     recompute_bbox=True,
    #     allow_negative_crop=False,
    #     bbox_clip_border=True
    # ),
    # dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='FixShapeResize',
    #     width=2000,
    #     height=1500,
    #     pad_val=0,
    #     keep_ratio=True,
    #     clip_object_border=True,
    #     interpolation='lanczos'
    # ),
    # dict(type='RandomResize', scale=crop_size, ratio_range=(0.8, 1.6), keep_ratio=True),
    # dict(type='Pad', size=crop_size, pad_to_square=False, pad_val=0, padding_mode='constant'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GeomTransform', prob=0.5, img_border_value=(0, 0, 0), interpolation='lanczos'),
    # dict(type='RandomCrop', crop_size=crop_size, crop_type='absolute', recompute_bbox=True, allow_negative_crop=False, bbox_clip_border=True),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_ignore_flags', 'gt_bboxes_labels'],
    #         min_visibility=0.01,
    #         filter_lost_elements=True),
    #     keymap={'img': 'image', 'gt_masks': 'masks', 'gt_bboxes': 'bboxes'},
    #     skip_img_without_anno=True
    # )
]


train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=10, bbox_occluded_thr=10, mask_occluded_thr=100, selected=True, paste_by_box=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10), min_gt_mask_area=100, by_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'img', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels', 'gt_masks'))
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(crop_size[0], crop_size[1]), keep_ratio=True),  # 'scale_factor'
    dict(
        type='PackDetInputs',
        # meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        meta_keys=('img_id', 'img_path', 'img', 'img_shape', 'ori_shape', 'scale_factor', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels', 'gt_masks'))
]

# -------------------------------- Dataloader --------------------------------

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img='train_sahi/', seg='annotations/panoptic_train2017/'),
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
    batch_size=batch_size,  # original 2
    num_workers=num_workers,
    drop_last=False,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img='val_sahi/', seg='annotations/panoptic_val2017/'),
        pipeline=test_pipeline,
        test_mode=False,
        backend_args=None)
)

test_dataloader = val_dataloader

# -------------------------------- Evaluator --------------------------------

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric=['bbox', 'segm'],
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
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=1),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP',
        rule='greater',
        patience=early_stopping_patience),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=10))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MemoryProfilerHook', interval=200),
    dict(type='CheckInvalidLossHook', interval=200),
    dict(type='EMAHook')  # not allowed during FSDP training
]

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)

custom_keys = {
    'absolute_pos_embed': backbone_embed_multi,
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.norm': backbone_norm_multi,
    'backbone.patch_embed.norm': backbone_norm_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}

custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

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

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments)


model = dict(
    type='Mask2Former',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=embed_dims,
        depths=depths,
        num_heads=num_heads,
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=with_cp,
        frozen_stages=-1,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint
        ),
    ),
    data_preprocessor=data_preprocessor,
    init_cfg=None,
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=num_stuff_classes,
        num_things_classes=num_things_classes
    ),
    panoptic_head=dict(
        type='Mask2FormerHead',
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[96, 192, 384, 768],
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0] * num_classes + [0.1],
            loss_weight=2.0,
            reduction='mean',
            use_sigmoid=False
        ),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=100,
        num_stuff_classes=num_stuff_classes,
        num_things_classes=num_things_classes,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128)
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[4, 8, 16, 32],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                cross_attn_cfg=dict(  # MultiheadAttention
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(  # MultiheadAttention
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True
        )
    ),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,  # In Mask2Former's panoptic postprocessing, it will filter mask area where score is less than 0.5
        max_per_image=100,
        panoptic_on=False,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler'))
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