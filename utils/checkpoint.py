import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import glob

# Kaggle persistent directory
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, defense_type):
    """Saves model state, optimizer state, and epoch number."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{defense_type}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"[*] Saved {defense_type} checkpoint to {checkpoint_path}")

def load_latest_checkpoint(model, optimizer, defense_type):
    """Finds the latest epoch checkpoint and loads it."""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, f"{defense_type}_epoch_*.pt"))
    if not checkpoints:
        print(f"[*] No existing {defense_type} checkpoints found. Starting from scratch.")
        return 0 # Start at epoch 0

    # Parse epoch numbers to find the latest
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('_epoch_')[1].split('.pt')[0]))
    checkpoint = torch.load(latest_checkpoint)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"[*] Resuming {defense_type} from epoch {start_epoch} (File: {latest_checkpoint})")
    return start_epoch
