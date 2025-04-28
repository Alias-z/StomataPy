# -------------------------------- Args shortcut --------------------------------

resume = None
load_from = None
val_interval = 1
log_processor = dict(by_epoch=True)
dinov2_checkpoint = 'train/checkpoints/dinov2_converted.pth'

fp16 = dict(loss_scale='dynamic')
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=8)

classes = ('background', 'stomatal complex', 'stoma', 'outer ledge', 'pore', 'pavement cell')
num_classes = len(classes)
class_weight = [1.0] * num_classes + [0.1]

model_crop_size = (512, 512)
dataset_type = 'StomataDataset'
data_root = 'StomataPy400K_filtered_train/'
output_dir = 'StomataPy400K_aperture_512'
work_dir = 'Models/' + output_dir
wandb_project = 'StomataPy'

batch_size = 8
n_gpus = 6
num_workers = 16
original_batch_size = 4
original_lr = 0.0001
original_n_gpus = 8
lr = original_lr * (n_gpus / original_n_gpus) * (batch_size / original_batch_size) / 10
auto_scale_lr = dict(base_batch_size=16, enable=False)

ReduceOnPlateauLR_patience = 50
early_stopping_patience = 80
max_epochs = 250
warmup_epochs = 30


# crop_size = (512, 512)
crop_size = (1024, 1024)

# -------------------------------- Data augmentation --------------------------------

albu_train_transforms = [
    dict(
        type='PadIfNeeded',
        min_height=crop_size[0],
        min_width=crop_size[1],
        border_mode=0,
        always_apply=True),
    dict(type='Flip', always_apply=True),
    dict(type='Rotate', limit=(-180, 180), interpolation=4, always_apply=True),
    # dict(type='RandomScale', scale_limit=0.1, interpolation=4, always_apply=True),
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
        p=0.5),
    dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.2, hue=0.2, always_apply=True),
    dict(type='AdvancedBlur', p=0.5),
    dict(type='CenterCrop', height=crop_size[0], width=crop_size[1], always_apply=True)
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(crop_size[0] * x * 0.1) for x in range(7, 15)],
        resize_type="ResizeShortestEdge",
        max_size=crop_size[0] * 4,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={'img': 'image', 'gt_seg_map': 'mask'}),
    dict(type='RandomCutOut', n_holes=10, cutout_ratio=(0.02, 0.05), prob=0.5),
    dict(type='Resize', scale=crop_size, keep_ratio=True, interpolation='lanczos'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'img_shape', 'img', 'gt_seg_map'))
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


# -------------------------------- Dataloader --------------------------------

train_dataloader = dict(
    batch_size=batch_size,  # original 1
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='splits//train.txt',
        pipeline=train_pipeline)
)


val_dataloader = dict(
    batch_size=1,  # must be 1
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='splits//val.txt',
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader


# -------------------------------- Evaluator --------------------------------

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

test_evaluator = val_evaluator

tta_model = dict(type='SegTTAModel')


# -------------------------------- Parameter scheduler --------------------------------

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)

val_cfg = dict(type='ValLoop')

test_cfg = val_cfg

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=600, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=999999,
        save_best='mIoU',
        save_last=True),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        patience=early_stopping_patience),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

optim_wrapper = dict(
    constructor='PEFTOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=lr,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'learnable_tokens': dict(decay_mult=0.0, lr_mult=1.0),
            'level_embed': dict(decay_mult=0.0, lr_mult=1.0),
            'norm': dict(decay_mult=0.0),
            'query_embed': dict(decay_mult=0.0, lr_mult=1.0),
            'reins.scale': dict(decay_mult=0.0, lr_mult=1.0)
        }),
        norm_decay_mult=0.0))

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
        monitor='mIoU',
        rule='greater',
        factor=0.75,
        patience=ReduceOnPlateauLR_patience,
        by_epoch=True,
        verbose=False)
]


# -------------------------------- Visualization backend --------------------------------

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs={'project': wandb_project, 'name': output_dir})
    ],
    name='visualizer')


# -------------------------------- Model config --------------------------------

model = dict(
    backbone=dict(
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
            type='LoRAReins'),
        type='ReinsDinoVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=model_crop_size,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        loss_cls=dict(
            class_weight=class_weight,
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
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
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        replace_query_feat=True,
        strides=[
            4,
            8,
            16,
            32,
        ],
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='ReinMask2FormerHead'),
    test_cfg=dict(
        crop_size=model_crop_size,
        mode='slide',
        stride=(341 * (model_crop_size[0] // 256) * 2, 341 * (model_crop_size[0] // 256) * 2)
    ),
    train_cfg=dict(),
    type='EncoderDecoder')  # type='FrozenBackboneEncoderDecoder' would prevent rein from being trained


# --------------------------------  No need to code below (Runtime) --------------------------------

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'

randomness = dict(seed=42, deterministic=False)

find_unused_parameters = True
