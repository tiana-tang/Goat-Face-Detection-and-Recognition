_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/sheep_face.py',
    # '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=1))
# optimizer

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=2,)
auto_scale_lr = dict(enable=True, base_batch_size=96)

# 0.002 71.5

optimizer = dict(type='SGD', lr=0.0006, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16, 20])
log_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=300)
