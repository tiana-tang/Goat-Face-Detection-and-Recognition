_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[7, 10])
# runner = dict(type='EpochBasedRunner', max_epochs=13)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

model = dict(
    backbone=dict(
        type='MEAN_ResNet',
        # stage_with_hggatt=(False,False,True,True),
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50.pth')))
data = dict(
    samples_per_gpu=4)
auto_scale_lr = dict(enable=True, base_batch_size=16)
load_from ='checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
# load_from ='pretrained_model/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'
# CUDA_VISIBLE_DEVICES=1,2 ./tools/dist_train.sh  configs/cwa/faster_rcnn_r50_fpn_meanatt_c3-c5_1x_voc.py 2