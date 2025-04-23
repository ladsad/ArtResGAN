import math
import torch.nn as nn
from .base_model import RRDBBlock

class ESRGANUpscaler(nn.Module):
    """ESRGAN-based upscaler module"""
    def __init__(self, scale_factor=4, in_channels=3, out_channels=3, nf=64, gc=32):
        super().__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)
        
        # RRDB blocks
        self.rrdb_blocks = nn.Sequential(*[RRDBBlock(nf, gc) for _ in range(23)])
        self.conv_body = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        # Upsampling blocks
        self.upsampling = nn.Sequential()
        for i in range(int(math.log(scale_factor, 2))):
            self.upsampling.add_module(f'up{i+1}', nn.Conv2d(nf, nf * 4, kernel_size=3, padding=1))
            self.upsampling.add_module(f'pixelshuffle{i+1}', nn.PixelShuffle(2))
            self.upsampling.add_module(f'lrelu{i+1}', nn.LeakyReLU(0.2, inplace=True))
        
        # Final output layer
        self.conv_last = nn.Conv2d(nf, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.rrdb_blocks(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        feat = self.upsampling(feat)
        out = self.conv_last(feat)
        return out