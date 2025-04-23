import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ArtworkDataset(Dataset):
    """Dataset for artwork restoration with synthetic damage"""
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all image files
        self.image_files = []
        for style_dir in os.listdir(root_dir):
            style_path = os.path.join(root_dir, style_dir)
            if os.path.isdir(style_path):
                for img_file in os.listdir(style_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_files.append(os.path.join(style_dir, img_file))
        
    def __len__(self):
        return len(self.image_files)
    
    def apply_random_damage(self, img):
        """Apply synthetic damage to artwork images."""
        img_np = np.array(img)
        
        # Choose random damage types
        damage_types = random.sample([
            'cracks', 'stains', 'scratches', 'fading', 'noise', 'missing_parts'
        ], k=random.randint(1, 3))
        
        damaged_img = img_np.copy()
        
        for damage_type in damage_types:
            if damage_type == 'cracks':
                # Simulate cracks using random lines
                for _ in range(random.randint(5, 20)):
                    pt1 = (random.randint(0, img_np.shape[1]-1), random.randint(0, img_np.shape[0]-1))
                    pt2 = (pt1[0] + random.randint(-100, 100), pt1[1] + random.randint(-100, 100))
                    cv2.line(damaged_img, pt1, pt2, (0, 0, 0), random.randint(1, 3))
                    
            elif damage_type == 'stains':
                # Simulate water stains or discoloration
                mask = np.zeros_like(img_np)
                for _ in range(random.randint(1, 5)):
                    center = (random.randint(0, img_np.shape[1]-1), random.randint(0, img_np.shape[0]-1))
                    radius = random.randint(20, 100)
                    color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                    cv2.circle(mask, center, radius, color, -1)
                
                # Blend the stain with the original image
                alpha = random.uniform(0.2, 0.5)
                damaged_img = cv2.addWeighted(damaged_img, 1-alpha, mask, alpha, 0)
                
            elif damage_type == 'scratches':
                # Simulate scratches
                for _ in range(random.randint(10, 30)):
                    pt1 = (random.randint(0, img_np.shape[1]-1), random.randint(0, img_np.shape[0]-1))
                    pt2 = (pt1[0] + random.randint(-50, 50), pt1[1] + random.randint(-50, 50))
                    cv2.line(damaged_img, pt1, pt2, (255, 255, 255), 1)
                    
            elif damage_type == 'fading':
                # Simulate color fading
                hsv = cv2.cvtColor(damaged_img, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.3, 0.7)  # Reduce saturation
                damaged_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            elif damage_type == 'noise':
                # Add noise
                noise = np.random.normal(0, random.uniform(5, 20), damaged_img.shape).astype(np.uint8)
                damaged_img = cv2.add(damaged_img, noise)
                
            elif damage_type == 'missing_parts':
                # Simulate missing parts
                for _ in range(random.randint(1, 5)):
                    x = random.randint(0, img_np.shape[1]-50)
                    y = random.randint(0, img_np.shape[0]-50)
                    w = random.randint(10, 50)
                    h = random.randint(10, 50)
                    damaged_img[y:y+h, x:x+w] = (255, 255, 255)
        
        return Image.fromarray(damaged_img)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # For training, create damaged version
        if self.is_train:
            damaged_image = self.apply_random_damage(image)
            target_image = image
        else:
            # For testing, we might have pairs of damaged/restored images
            # or we just use synthetic damage as in training
            damaged_image = self.apply_random_damage(image)
            target_image = image
        
        # Apply transformations
        if self.transform:
            damaged_image = self.transform(damaged_image)
            target_image = self.transform(target_image)
            
        return {'damaged': damaged_image, 'target': target_image}