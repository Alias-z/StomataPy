resume = None
load_from = None
dinov2_checkpoint = 'train/checkpoints/dinov2_converted.pth'
output_dir = 'StomataPy400K_stomatal_complex_24185'

fp16 = dict(loss_scale='dynamic')
with_cp = True  # for FSDP: the checkpoint needs to be controlled by the checkpoint_check_fn.
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)

num_stuff_classes = 0
classes = ('stomatal complex', )
num_things_classes = len(classes)
num_classes = num_things_classes + num_stuff_classes

dataset_type = 'CocoEevenSamplerDataset'
data_root = 'StomataPy400K_train/'

all_datasets = [
    'ClearStain_Brightfield', 'Imprints_Brightfield', 'Imprints_DIC',
    'Leaf_Brightfield', 'Leaf_Topometry', 'Peels_Brightfield', 'Peels_SEM',
]

total_samples_train = 24185 // len(all_datasets)  # total train 24185
total_samples_val = 6068 // len(all_datasets)  # total val 6068

work_dir = 'Models//' + output_dir
wandb_project = 'StomataPy'

train_ann_file = 'sahi_coco_train.json'
val_ann_file = 'sahi_coco_val.json'
test_ann_file = 'COCO.json'

batch_size = 4
n_gpus = 6
num_workers = 16
original_batch_size = 16  # 2
original_lr = 0.0001
original_n_gpus = 8
lr = original_lr * (n_gpus / original_n_gpus) * (batch_size / original_batch_size) * 10
auto_scale_lr = dict(base_batch_size=16, enable=False)


ReduceOnPlateauLR_patience = 50
early_stopping_patience = 150
warmup_epochs = 30
max_epochs = 240
val_interval = max_epochs // 10

crop_size = (1280, 1024)
# crop_size = (1152, 896)
# crop_size = (1024, 768)
image_size = (512, 512)


# -------------------------------- Data augmentation --------------------------------

load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GeomTransform', prob=0.5, img_border_value=(0, 0, 0), interpolation='lanczos'),
    dict(type='Resize', scale=(crop_size[0], crop_size[1]), keep_ratio=True),  # 'scale_factor
]


train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=5, bbox_occluded_thr=50, mask_occluded_thr=1000, selected=True, paste_by_box=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(10, 10), min_gt_mask_area=10, by_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'img', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels', 'gt_masks'))
]


test_load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
]


test_pipeline = [
    dict(type='Resize', scale=(crop_size[0], crop_size[1]), keep_ratio=True),  # 'scale_factor'
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img', 'img_shape', 'ori_shape', 'scale_factor', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels', 'gt_masks'))
]


# -------------------------------- Dataloader --------------------------------

train_dataset = dict(
    type='MultiImageMixEvenSamplerDataset',
    batch_size=batch_size,
    n_gpus=n_gpus,
    n_workers=num_workers,
    mode='train',
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_ann_file,
        all_datasets=all_datasets,
        total_samples=total_samples_train,
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


val_dataset = dict(
    type='MultiImageMixEvenSamplerDataset',
    batch_size=batch_size,
    n_gpus=n_gpus,
    n_workers=num_workers,
    mode='val',
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        all_datasets=all_datasets,
        total_samples=total_samples_val,
        data_prefix=dict(img='val_sahi/', seg='annotations/panoptic_train2017/'),
        pipeline=test_load_pipeline,
        test_mode=False,
        backend_args=None),
    pipeline=test_pipeline)

val_dataloader = dict(
    batch_size=batch_size,  # original 2
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=val_dataset)


test_dataset = dict(
    type='MultiImageMixEvenSamplerDataset',
    batch_size=batch_size,
    mode='val',
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=test_ann_file,
        all_datasets=all_datasets,
        total_samples=total_samples_val,
        data_prefix=dict(img='test/', seg='annotations/panoptic_train2017/'),
        pipeline=test_load_pipeline,
        test_mode=False,
        backend_args=None),
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=batch_size,  # original 2
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=test_dataset)

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
        interval=max_epochs // 5,
        by_epoch=True,
        save_last=True,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=5),
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


embed_multi = dict(lr_mult=1.0, decay_mult=0.0)


optim_wrapper = dict(
    constructor='PEFTOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
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
        eta_min=lr * 1e-03,
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
        pad_seg=False,
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
    pad_seg=False,
    seg_pad_value=255,
    batch_augments=batch_augments)


model = dict(
    type='Mask2Former',
    backbone=dict(
        type='ReinsDinoVisionTransformer',
        block_chunks=0,
        depth=24,
        embed_dim=1024,
        ffn_bias=True,
        ffn_layer='mlp',
        img_size=512,
        init_cfg=dict(
            checkpoint=dinov2_checkpoint, type='Pretrained'),
        init_values=1e-05,
        mlp_ratio=4,
        num_heads=16,
        patch_size=16,
        proj_bias=True,
        qkv_bias=True,
        reins_config=dict(
            embed_dims=1024,
            link_token_to_query=True,
            lora_dim=16,
            num_layers=24,
            patch_size=16,
            token_length=100,
            type='LoRAReins')
    ),
    data_preprocessor=data_preprocessor,
    init_cfg=None,
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None
    ),
    panoptic_head=dict(
        type='ReinMask2FormerHead',
        in_channels=[1024, 1024, 1024, 1024],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)
    ),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True
    ),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')
    )
)


# --------------------------------  No need to code below (Runtime) --------------------------------


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'

randomness = dict(seed=42, deterministic=False)

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

find_unused_parameters = True
