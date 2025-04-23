import torch

class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model hyperparameters
        self.lambda_adv = 1.0
        self.lambda_content = 10.0
        self.lambda_style = 1000.0
        self.lambda_tv = 1.0
        self.scale_factor = 2
        
        # Training parameters
        self.batch_size = 4
        self.num_workers = 2
        self.num_epochs = 2
        self.lr = 0.0002
        self.print_freq = 100
        self.save_freq = 5
        self.max_steps_per_epoch = 4001
        
        # Paths
        self.data_dir = "../input/wikiart"
        self.checkpoint_path = "/kaggle/working/artresgan_latest_checkpoint.pth"
        self.final_model_path = "/kaggle/working/artresgan_generator_final.pth"

config = Config()
