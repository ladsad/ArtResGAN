import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import ArtResGAN
from utils import ArtworkDataset
from config import config
from config import config


def test(model_path, test_images):
    # Initialize model
    model = ArtResGAN(config.device, scale_factor=config.scale_factor)
    model.generator.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Testing logic
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Process test images
    for img_path in test_images:
        # Load and damage image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # Apply synthetic damage
        dataset = ArtworkDataset(root_dir="../input/wikiart", transform=None, is_train=True)
        damaged_image = dataset.apply_random_damage(image)
        
        # Transform to tensor
        damaged_tensor = transform(damaged_image).unsqueeze(0).to(config.device)
        
        # Generate restored and upscaled images
        with torch.no_grad():
            upscaled, restored = model(damaged_tensor, apply_upscaling=True)
        
        # Convert tensors to images
        damaged_np = ((damaged_tensor.squeeze().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        restored_np = ((restored.squeeze().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        original_np = np.array(image.resize((256, 256)))
        
        # Display results
        plt.figure(figsize=(20, 10))
        
        # Original
        plt.subplot(1, 3, 1)
        plt.imshow(original_np)
        plt.title("Original")
        plt.axis('off')
        
        # Damaged
        plt.subplot(1, 3, 2)
        plt.imshow(damaged_np)
        plt.title("Damaged")
        plt.axis('off')
        
        # Restored (before upscaling)
        plt.subplot(1, 3, 3)
        plt.imshow(restored_np)
        plt.title("Restored")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_images = [
        "../input/wikiart/Naive_Art_Primitivism/aldemir-martins_blue-cat.jpg",
        # ... other test images
    ]
    test(config.final_model_path, test_images)
