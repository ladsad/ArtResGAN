import cv2
import numpy as np
import torch
import pywt

class MachineVisionTechniques:
    @staticmethod
    def sobel_edge_detection(image_tensor):
        """Apply Sobel edge detection to image tensor."""
        # Convert tensor to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.sobel_edge_detection(img) for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            # Scale from [-1, 1] to [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_tensor
            
        # Convert to grayscale if it's RGB
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Apply Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to [0, 1]
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert back to tensor
        magnitude_tensor = torch.from_numpy(magnitude).float().unsqueeze(0)
        
        if isinstance(image_tensor, torch.Tensor):
            magnitude_tensor = magnitude_tensor.to(image_tensor.device)
            
        return magnitude_tensor
    
    @staticmethod
    def canny_edge_detection(image_tensor, low_threshold=100, high_threshold=200):
        """Apply Canny edge detection to image tensor."""
        # Convert tensor to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.canny_edge_detection(img, low_threshold, high_threshold) 
                                   for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            # Scale from [-1, 1] to [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_tensor
            
        # Convert to grayscale if it's RGB
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to tensor
        edges_tensor = torch.from_numpy(edges).float() / 255.0
        edges_tensor = edges_tensor.unsqueeze(0)
        
        if isinstance(image_tensor, torch.Tensor):
            edges_tensor = edges_tensor.to(image_tensor.device)
            
        return edges_tensor
    
    @staticmethod
    def morphological_operations(image_tensor, operation='dilate', kernel_size=3):
        """Apply morphological operations (dilate/erode) to image tensor."""
        # Convert tensor to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.morphological_operations(img, operation, kernel_size) 
                                   for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            # Scale from [-1, 1] to [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_tensor
            
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply morphological operation
        if operation == 'dilate':
            result = cv2.dilate(image_np, kernel, iterations=1)
        elif operation == 'erode':
            result = cv2.erode(image_np, kernel, iterations=1)
        elif operation == 'open':
            result = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            result = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
        # Convert back to tensor
        if result.ndim == 2:
            result = result[..., np.newaxis]
            
        result_tensor = torch.from_numpy(result).float().permute(2, 0, 1)
        # Scale back to [-1, 1]
        result_tensor = result_tensor / 127.5 - 1
        
        if isinstance(image_tensor, torch.Tensor):
            result_tensor = result_tensor.to(image_tensor.device)
            
        return result_tensor

    @staticmethod
    def local_binary_pattern(image_tensor, P=8, R=1):
        """Apply Local Binary Pattern for texture analysis."""
        from skimage.feature import local_binary_pattern
        
        # Convert tensor to numpy for scikit-image processing
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.local_binary_pattern(img, P, R) 
                                   for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            # Scale from [-1, 1] to [0, 1]
            image_np = (image_np + 1) / 2
        else:
            image_np = image_tensor
            
        # Convert to grayscale if it's RGB
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray / 255.0
        else:
            gray = image_np.squeeze()
            
        # Apply LBP
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        
        # Normalize to [0, 1]
        lbp = lbp / lbp.max()
        
        # Convert back to tensor
        lbp_tensor = torch.from_numpy(lbp).float().unsqueeze(0)
        
        if isinstance(image_tensor, torch.Tensor):
            lbp_tensor = lbp_tensor.to(image_tensor.device)
            
        return lbp_tensor
    
    @staticmethod
    def harris_corner_detection(image_tensor, block_size=2, ksize=3, k=0.04):
        """Apply Harris corner detection."""
        # Convert tensor to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.harris_corner_detection(img, block_size, ksize, k) 
                                   for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            # Scale from [-1, 1] to [0, 255]
            image_np = ((image_np + 1) * 127.5).astype(np.uint8)
        else:
            image_np = image_tensor
            
        # Convert to grayscale if it's RGB
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Apply Harris corner detection
        dst = cv2.cornerHarris(np.float32(gray), block_size, ksize, k)
        
        # Normalize to [0, 1]
        dst = cv2.normalize(dst, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert back to tensor
        dst_tensor = torch.from_numpy(dst).float().unsqueeze(0)
        
        if isinstance(image_tensor, torch.Tensor):
            dst_tensor = dst_tensor.to(image_tensor.device)
            
        return dst_tensor
    
    @staticmethod
    def haar_wavelet_transform(image_tensor):
        """Apply Haar wavelet transform for multi-scale analysis."""
        # Convert tensor to numpy
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                return torch.stack([MachineVisionTechniques.haar_wavelet_transform(img) 
                                   for img in image_tensor])
            
            image_np = image_tensor.detach().cpu().numpy()
            if image_np.shape[0] == 3:  # RGB image
                image_np = image_np.transpose(1, 2, 0)
        else:
            image_np = image_tensor
            
        # Convert to grayscale if it's RGB
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray / 255.0
        else:
            gray = image_np.squeeze()
            
        # Apply Haar wavelet transform
        coeffs = pywt.dwt2(gray, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # Normalize each component
        LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)
        LH = (LH - LH.min()) / (LH.max() - LH.min() + 1e-8)
        HL = (HL - HL.min()) / (HL.max() - HL.min() + 1e-8)
        HH = (HH - HH.min()) / (HH.max() - HH.min() + 1e-8)
        
        # Stack components
        wavelet_features = np.stack([LL, LH, HL, HH], axis=0)
        
        # Convert back to tensor
        wavelet_tensor = torch.from_numpy(wavelet_features).float()
        
        if isinstance(image_tensor, torch.Tensor):
            wavelet_tensor = wavelet_tensor.to(image_tensor.device)
            
        return wavelet_tensor
    
    @staticmethod
    def enhance_image_with_mv_techniques(image_tensor):
        """Apply multiple machine vision techniques and combine the results."""
        # Get edge maps
        sobel_edges = MachineVisionTechniques.sobel_edge_detection(image_tensor)
        canny_edges = MachineVisionTechniques.canny_edge_detection(image_tensor)
        
        # Get texture features
        lbp_features = MachineVisionTechniques.local_binary_pattern(image_tensor)
        harris_corners = MachineVisionTechniques.harris_corner_detection(image_tensor)
        
        # Get wavelet features (all 4 components)
        wavelet_features = MachineVisionTechniques.haar_wavelet_transform(image_tensor)
        
        # Combine features
        edge_map = (sobel_edges + canny_edges) / 2
        texture_map = (lbp_features + harris_corners) / 2
        
        # Create feature tensor
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:  # batch of images
                b, c, h, w = image_tensor.shape
                # Upsample all 4 wavelet components
                wavelet_upsampled = F.interpolate(wavelet_features[:, :4], size=(h, w), mode='bilinear', align_corners=False)
                enhanced_features = torch.cat([
                    image_tensor,          # 3 channels
                    edge_map,              # 1 channel
                    texture_map,           # 1 channel
                    wavelet_upsampled      # 4 channels
                ], dim=1)
            else:
                c, h, w = image_tensor.shape
                wavelet_upsampled = F.interpolate(wavelet_features[:4].unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
                enhanced_features = torch.cat([
                    image_tensor,
                    edge_map,
                    texture_map,
                    wavelet_upsampled
                ], dim=0)
        
        return enhanced_features