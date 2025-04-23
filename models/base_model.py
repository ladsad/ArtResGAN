import torch.nn as nn

class ResnetBlock(nn.Module):
    """Residual block with configurable padding type"""
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_bias=True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
            
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
            
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class VGGLoss(nn.Module):
    """Perceptual loss using VGG19"""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device)
        self.vgg = nn.Sequential()
        for i in range(36):  # Use up to relu5_3
            self.vgg.add_module(str(i), vgg[i])
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    def gram_matrix(self, x):
        # Handle both 3D and 4D tensors
        if x.dim() == 4:  # batch of features
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)
        elif x.dim() == 3:  # single feature map
            c, h, w = x.size()
            features = x.view(1, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram.div(c * h * w)
        else:
            raise ValueError(f"Unexpected tensor dimension: {x.dim()}")
        
    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        
        style_loss = 0
        # Make sure x_vgg and y_vgg are lists or tuples
        if not isinstance(x_vgg, (list, tuple)):
            x_vgg = [x_vgg]
        if not isinstance(y_vgg, (list, tuple)):
            y_vgg = [y_vgg]
        
        # Use only as many weights as we have feature maps
        num_features = min(len(x_vgg), len(self.weights))
        
        for i in range(num_features):
            style_loss += self.weights[i] * self.criterion(
                self.gram_matrix(x_vgg[i]), self.gram_matrix(y_vgg[i]))
        
        return style_loss
    
    def content_loss(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        
        # If the output is not a list/tuple, use it directly
        if not isinstance(x_vgg, (list, tuple)) and not isinstance(y_vgg, (list, tuple)):
            return self.criterion(x_vgg, y_vgg)
        
        # If the outputs are lists/tuples, use the last feature map
        if isinstance(x_vgg, (list, tuple)) and isinstance(y_vgg, (list, tuple)):
            return self.criterion(x_vgg[-1], y_vgg[-1])
        
        # Handle mixed cases (shouldn't normally happen)
        raise ValueError("Inconsistent VGG outputs: one is a list/tuple, the other is not")
