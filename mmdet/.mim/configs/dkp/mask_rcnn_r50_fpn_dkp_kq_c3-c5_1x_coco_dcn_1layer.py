_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=14)

model = dict(
    backbone=dict(
        type='DKP_ResNet',
        dkp=dict(type='DKP',scale=4,weight_share=False,use_dcn=True),
        stage_with_dkp=(False, True, True, True),))
data = dict(
    samples_per_gpu=3)
auto_scale_lr = dict(enable=True, base_batch_size=16)
load_from ='checkpoints/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth'
find_unused_parameters=True