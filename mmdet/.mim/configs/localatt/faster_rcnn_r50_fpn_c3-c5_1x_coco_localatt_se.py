_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# CUDA_VISIBLE_DEVICES=1,3 nohup ./tools/dist_train.sh configs/localatt/faster_rcnn_r50_fpn_c3-c5_1x_coco_localatt.py 2 &

# CUDA_VISIBLE_DEVICES=1,3  ./tools/dist_train.sh configs/localatt/faster_rcnn_r50_fpn_c3-c5_1x_voc_cbam_regular.py 2


optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


model = dict(
    backbone=dict(
        type='LA_ResNet',
        la=dict(type='LA'),
        stage_with_la=(False, True, True, True),
        # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50.pth')
        ))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
auto_scale_lr = dict(enable=True, base_batch_size=16)
# load_from ='checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
# load_from ='pretrained_model/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'