import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from models import ArtResGAN
from utils import ArtworkDataset
from config import config


def train_model():
    # Set device
    device = config.device
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Create data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    train_dataset = ArtworkDataset(
        root_dir=config.data_dir,
        transform=transform,
        is_train=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = ArtResGAN(
        device=device,
        lambda_adv=config.lambda_adv,
        lambda_content=config.lambda_content,
        lambda_style=config.lambda_style,
        lambda_tv=config.lambda_tv,
        scale_factor=config.scale_factor
    )
    
    # Create optimizers
    optimizer_G = optim.Adam(model.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    # Initialize starting epoch
    start_epoch = 0
    
    # Check for existing checkpoint
    if os.path.exists(config.checkpoint_path):
        try:
            print(f"Loading checkpoint from {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            if i >= config.max_steps_per_epoch:
                break
            
            # Transfer data to device
            damaged = batch['damaged'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            
            # ======== Train Discriminator ========
            optimizer_D.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                restored = model(damaged)
                loss_D, loss_D_dict = model.compute_discriminator_loss(damaged, restored, target)
            
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            
            # ======== Train Generator ========
            optimizer_G.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                restored = model(damaged)
                loss_G, loss_G_dict = model.compute_generator_loss(damaged, restored, target)
            
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            # ======== Log Progress ========
            if i % config.print_freq == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}] | Step [{i+1}/{len(train_loader)}] | "
                      f"G Loss: {loss_G_dict['total']:.4f} | D Loss: {loss_D_dict['total']:.4f} | "
                      f"Time: {time.time()-start_time:.2f}s")
                
                # Visualize sample results
                if i % (config.print_freq * 5) == 0:
                    with torch.no_grad():
                        sample = torch.cat([
                            damaged[:4],
                            restored[:4],
                            target[:4]
                        ], dim=0)
                        
                        grid = vutils.make_grid(sample, nrow=4, normalize=True)
                        plt.figure(figsize=(15, 15))
                        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                        plt.axis('off')
                        plt.title(f"Epoch {epoch+1} | Step {i+1}")
                        plt.show()
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"/kaggle/working/checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.generator.state_dict(), config.final_model_path)
    print("Training completed. Final model saved.")

        
if __name__ == "__main__":
    train_model()
