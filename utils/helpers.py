import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_checkpoint(epoch, model, optimizer, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer[0].state_dict(),
        'optimizer_D_state_dict': optimizer[1].state_dict(),
    }, path)

def load_checkpoint(path, device, model, optimizer):
    """Load training checkpoint with proper device handling"""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0, float('inf')  # Return default epoch and loss
    
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state with strict=False for backward compatibility
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load optimizer state and move tensors to device
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Manually move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # Get saved training progress
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Continuing with initial model weights")
        return 0, float('inf')
