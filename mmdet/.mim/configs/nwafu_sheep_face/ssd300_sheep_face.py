_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/sheep_face.py',
    '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(num_classes=1,),
    # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/ssd300_coco_20210803_015428-d231a06e.pth')
    )

data = dict(
    samples_per_gpu=30,
    workers_per_gpu=2,)
auto_scale_lr = dict(enable=True, base_batch_size=480)
# optimizer
optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 20])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(interval=10)

#  CUDA_VISIBLE_DEVICES=3 nohup python ./tools/train.py configs/nwafu_sheep_face/ssd300_sheep_face.py &
