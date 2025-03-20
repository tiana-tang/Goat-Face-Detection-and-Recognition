_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]

# nohup ./tools/dist_train.sh configs/localatt/faster_rcnn_r50_fpn_c3-c5_1x_voc_cbam_beforerelu.py 4 &

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', 
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)

# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20)),
    backbone=dict(
        type='CBAM_ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50.pth')))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
auto_scale_lr = dict(enable=True, base_batch_size=16)
# load_from ='checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
# load_from ='pretrained_model/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'