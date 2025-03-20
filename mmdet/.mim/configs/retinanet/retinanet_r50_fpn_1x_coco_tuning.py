_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,)
mp_start_method = 'spawn'
workflow = [('tuning', 1)]
find_unused_parameters=True
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'