import torch.nn as nn
from .base_model import ResnetBlock

class ArtResGenerator(nn.Module):
    """Generator with enhanced features input"""
    def __init__(self, input_nc=9, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 n_blocks=6, padding_type='reflect', scale_factor=1):
        super().__init__()
        
        # Initial convolution block - now accepting 9 channels (3 original + 6 enhanced)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # ResNet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]
        
        # Upsampling with efficient sub-pixel convolution
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                # Efficient sub-pixel convolution for upsampling
                nn.Conv2d(ngf * mult, int(ngf * mult / 2) * 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),  # Sub-pixel convolution
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Additional super-resolution scaling if needed
        if scale_factor > 1:
            sr_layers = []
            remaining_scale = scale_factor
            while remaining_scale > 1:
                if remaining_scale >= 4:
                    # 2x upscaling
                    sr_layers.extend([
                        nn.Conv2d(ngf, ngf * 4, kernel_size=3, padding=1, bias=True),
                        nn.PixelShuffle(2),
                        nn.ReLU(True)
                    ])
                    remaining_scale /= 2
                else:
                    # Fractional upscaling using interpolation for any remaining scale
                    sr_layers.append(nn.Upsample(scale_factor=remaining_scale, mode='bicubic', align_corners=False))
                    break
            model.extend(sr_layers)
        
        # Final output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        return self.model(input)