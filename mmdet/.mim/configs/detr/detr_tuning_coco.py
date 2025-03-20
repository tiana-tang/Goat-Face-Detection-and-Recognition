_base_ = 'detr_r50_8x2_150e_coco.py'


# algorithm = dict(type='BaseAlgorithm',)
mp_start_method = 'spawn'
workflow = [('tuning', 1)]
find_unused_parameters=True
load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'