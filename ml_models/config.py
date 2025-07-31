import torch

"""Configuration for temperature SR model"""

def load_config():
    """Load configuration for temperature SR model"""
    return {
        'name': 'TemperatureSR_SwinIR_ESRGAN_x2_90k',
        'model_type': 'TemperatureSRModel',
        'scale': 2,
        'num_gpu': 1 if torch.cuda.is_available() else 0,
        'network_g': {
            'type': 'SwinIR',
            'upscale': 2,
            'in_chans': 1,
            'img_size': 64,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 60,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 4,
            'upsampler': 'pixelshuffle',
            'resi_connection': '3conv'
        },
        'network_d': {
            'type': 'UNetDiscriminatorSN',
            'num_in_ch': 1,
            'num_feat': 64,
            'skip_connection': True
        },
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'resume_state': None
        },
        'train': {
            'ema_decay': 0.999,
            'optim_g': {
                'type': 'Adam',
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'betas': [0.9, 0.99]
            }
        },
        'is_train': False,
        'dist': False
    }