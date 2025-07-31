"""
Temperature SR Model adapted for inference only
"""

import torch
import torch.nn as nn
from .network_swinir import SwinIR


class TemperatureSRModel:
    """Temperature SR Model for inference"""

    def __init__(self, opt):
        self.opt = opt
        if opt.get('device'):
            self.device = torch.device(opt['device'])
        else:
            from utils.device_utils import get_best_device
            self.device, device_name = get_best_device()
            print(f"Temperature SR Model using: {device_name}")

        # Build generator only (no discriminator needed for inference)
        self.net_g = self.build_swinir_generator(opt)
        self.net_g = self.net_g.to(self.device)

    def build_swinir_generator(self, opt):
        """Build SwinIR generator for temperature data"""
        opt_net = opt['network_g']

        model = SwinIR(
            upscale=opt_net.get('upscale', 2),
            in_chans=opt_net.get('in_chans', 1),
            img_size=opt_net.get('img_size', 64),
            window_size=opt_net.get('window_size', 8),
            img_range=opt_net.get('img_range', 1.),
            depths=opt_net.get('depths', [6, 6, 6, 6, 6, 6]),
            embed_dim=opt_net.get('embed_dim', 60),
            num_heads=opt_net.get('num_heads', [6, 6, 6, 6, 6, 6]),
            mlp_ratio=opt_net.get('mlp_ratio', 4),
            upsampler=opt_net.get('upsampler', 'pixelshuffle'),
            resi_connection=opt_net.get('resi_connection', '3conv')
        )

        return model