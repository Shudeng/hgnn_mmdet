radius=0.2
num_sample=64
mlp_channels=(64, 64, 128)
norm_cfg=dict(type='BN2d')
sa_cfg=dict(
    type='PointSAModule',
    pool_mod='max',
    use_xyz=True,
    normalize_xyz=True)




