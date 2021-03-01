import numpy as np

architecture = ['simple',
                'resnetb',
                'resnetb_strided',
                'resnetb',
                'resnetb',
                'resnetb_strided',
                'resnetb',
                'resnetb',
                'resnetb_strided',
                'resnetb',
                'resnetb',
                'resnetb_strided',
                'resnetb',
                'resnetb',
                'global_average']

num_layers = len([block for block in architecture if 'pool' in block or 'strided' in block]) + 1

layer_blocks = []
deform_layers = []

for block_i, block in enumerate(architecture):
    if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
        layer_blocks += [block]
        continue

    deform_layer = False
    if layer_blocks:
        if np.any(['deformable' in blck for blck in layer_blocks]):
            deform_layer = True

    if 'pool' in block or 'strided' in block:
        if 'deformable' in block:
            deform_layer = True

    deform_layers += [deform_layer]
    layer_blocks = []

    if 'global' in block or 'upsample' in block:
        break
        
DATA = {
    'num_classes': 3,
    'root': "/HDD1/mvpservereight/minhyeok/PointCloud/object_float_v5_txt"
}

SETTING = {
    'epoch_steps': 300,
    'batch_num': 10,
    'validation_size': 30,
    'in_features_dim': 1,
    'first_features_dim': 64,
    'num_kernel_points': 15,
    'first_subsampling_dl': 0.02,
    'num_layers': num_layers,
    'deform_layers': deform_layers,
    'deform_radius': 6.0,
    'conv_radius': 2.5,
    'input_threads': 10,
    'KP_extent': 1.2,
    'in_points_dim': 3,
    'use_batch_norm': True,
    'batch_norm_momentum': 0.05,
    'augment_rotation': 'none',
    'augment_scale_min': 0.8,
    'augment_scale_max': 1.2,
    'augment_scale_anisotropic': True,
    'augment_symmetries': [True, True, True],
    'augment_noise': 0.001
}

TRAIN = {
    'epoch': 100,
    'learning_rate': 1e-3,
    'deform_lr_factor': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    'grad_clip_norm': 100.0,
    'print_freq': 5
}