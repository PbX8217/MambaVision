import os
import os.path as osp

default_scope = 'mmdet'

# TensorBoard + local visual logs live under log_dirs/${RUN_NAME}
run_name = os.getenv('RUN_NAME', 'default_run')
log_root = os.getenv('LOG_DIR', osp.join('log_dirs', run_name))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='TensorboardVisBackend', save_dir=log_root),
    dict(type='LocalVisBackend', save_dir=log_root),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
