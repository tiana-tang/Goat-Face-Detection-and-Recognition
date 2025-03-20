_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/sheep_face.py',
    '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=1),))

# data = dict(
#     samples_per_gpu=16)
# # optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[26, 32])
# runner = dict(type='EpochBasedRunner', max_epochs=36)


# auto_scale_lr = dict(enable=True, base_batch_size=16)
# load_from ='checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'



data = dict(
    samples_per_gpu=30,
    workers_per_gpu=2,)
auto_scale_lr = dict(enable=True, base_batch_size=96)
# optimizer
optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 20])
log_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=300)
