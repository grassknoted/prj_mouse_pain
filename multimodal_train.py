"""
Training script for multimodal mouse action recognition model.
Combines visual and pose features.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
from datetime import datetime
from tqdm import tqdm

from multimodal_data_loader import create_multimodal_data_loaders
from multimodal_model import create_multimodal_model
from evaluation import evaluate, compute_metrics


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def find_free_port():
    """Find a free port for distributed training."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        # Set device BEFORE initializing process group
        torch.cuda.set_device(local_rank)

        # Find and set free port if MASTER_PORT not set
        if 'MASTER_PORT' not in os.environ:
            if rank == 0:
                port = find_free_port()
                os.environ['MASTER_PORT'] = str(port)
                print(f"Auto-selected port: {port}")

        # Use Gloo backend for virtualized GPUs (NCCL doesn't support MIG/vGPU)
        # Gloo is slower but works with virtualized/shared physical GPUs
        dist.init_process_group(
            backend='gloo',
            init_method='env://'
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch_multimodal(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: str,
    scaler=None
) -> Dict[str, float]:
    """
    Train for one epoch with multimodal data.

    Returns:
        Dictionary with loss and accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (visual, pose, labels) in enumerate(pbar):
        visual = visual.to(device)
        pose = pose.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(visual, pose)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(visual, pose)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics
        total_loss += loss.item() * visual.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    avg_loss = total_loss / total
    avg_acc = 100 * correct / total

    return {"loss": avg_loss, "accuracy": avg_acc}


def evaluate_multimodal(
    model: nn.Module,
    val_loader,
    criterion,
    device: str
) -> Tuple[float, float, Dict]:
    """
    Evaluate multimodal model on validation set.

    Returns:
        val_loss, val_accuracy, detailed_metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Create progress bar for validation
        pbar = tqdm(val_loader, desc="Validation", leave=False)

        for visual, pose, labels in pbar:
            visual = visual.to(device)
            pose = pose.to(device)
            labels = labels.to(device)

            logits = model(visual, pose)
            loss = criterion(logits, labels)

            total_loss += loss.item() * visual.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar with running loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_loss = total_loss / len(all_labels)
    val_acc = 100 * np.mean(all_preds == all_labels)

    metrics = compute_metrics(all_preds, all_labels)

    return val_loss, val_acc, metrics


def train_multimodal_model(
    video_dir: str,
    annotation_dir: str,
    dlc_dir: str,
    output_dir: str = "./checkpoints_multimodal",
    num_epochs: int = 100,
    batch_size: int = 32,
    clip_length: int = 17,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    model_type: str = "standard",
    use_amp: bool = True,
    random_seed: int = 42
):
    """
    Train multimodal action recognition model with multi-GPU support.

    Args:
        video_dir: Directory with video files
        annotation_dir: Directory with annotation CSVs
        dlc_dir: Directory with DLC coordinate CSVs
        output_dir: Where to save checkpoints
        num_epochs: Max training epochs
        batch_size: Batch size per GPU
        clip_length: Temporal clip length
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        model_type: "standard" or "light"
        use_amp: Use automatic mixed precision
        random_seed: Random seed
    """
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    # Only print from main process
    if is_main_process:
        print(f"Distributed training: {world_size} GPU(s)")

    # Setup
    torch.manual_seed(random_seed + rank)
    np.random.seed(random_seed + rank)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if is_main_process:
        print(f"Using device: {device}")

    output_dir = Path(output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    if is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-download DinoV2 on rank 0 to avoid conflicts
    if world_size > 1:
        if is_main_process:
            print("\n" + "="*60)
            print("PRE-DOWNLOADING DINOV2 MODEL (RANK 0 ONLY)")
            print("="*60)
            import os
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            try:
                _ = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=True, trust_repo=True)
                print("DinoV2 pre-download complete!")
            except Exception as e:
                print(f"Pre-download warning: {e}")
            print("="*60 + "\n")

        # Wait for rank 0 to finish downloading
        print(f"[Rank {rank}] Waiting at barrier for DinoV2 download...")
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed, DinoV2 is ready!")

    # Synchronize all processes
    if world_size > 1:
        print(f"[Rank {rank}] Waiting at barrier before data loading...")
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed, starting data loading...")

    # Data
    if is_main_process:
        print("\n" + "="*60)
        print("STARTING DATA LOADING")
        print("="*60)

    # Set up extracted frames directory
    extracted_frames_dir = str(Path(video_dir).parent / "extracted_frames")

    print(f"[Rank {rank}] Calling create_multimodal_data_loaders...")
    print(f"[Rank {rank}] Using extracted frames from: {extracted_frames_dir}")

    train_loader, val_loader, class_weights = create_multimodal_data_loaders(
        video_dir, annotation_dir, dlc_dir,
        batch_size=batch_size,
        clip_length=clip_length,
        stride=1,
        test_size=0.2,
        random_seed=random_seed,
        distributed=(world_size > 1),
        use_extracted_frames=True,
        extracted_frames_dir=extracted_frames_dir
    )
    print(f"[Rank {rank}] Data loaders created successfully!")

    if is_main_process:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("="*60)
        print("DATA LOADING COMPLETE")
        print("="*60 + "\n")

    # Model
    if is_main_process:
        print("Creating multimodal model with DinoV2...")

    print(f"[Rank {rank}] Creating model...")

    model = create_multimodal_model(
        num_classes=7,
        model_type=model_type,
        device=device,
        use_dinov2=True  # Use frozen DinoV2 + Bi-GRU + Attention
    )
    print(f"[Rank {rank}] Model created successfully!")

    # Wrap model in DDP for multi-GPU training
    if world_size > 1:
        print(f"[Rank {rank}] Wrapping model in DDP...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {rank}] DDP wrapper complete!")

    # Loss and optimizer
    print(f"[Rank {rank}] Setting up loss and optimizer...")
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    print(f"[Rank {rank}] Optimizer and scheduler ready!")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"[Rank {rank}] Mixed precision (AMP) enabled!")

    # Tensorboard
    writer = SummaryWriter(str(run_dir))

    # Training
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    best_val_f1 = 0.0

    print("Starting training...")

    # Epoch-level progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, disable=not is_main_process)

    for epoch in epoch_pbar:
        # Set epoch for distributed sampler
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train
        train_metrics = train_epoch_multimodal(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_acc, val_metrics = evaluate_multimodal(
            model, val_loader, criterion, device
        )

        # Update epoch progress bar with key metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{train_metrics["loss"]:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'val_f1': f'{val_metrics["f1_macro"]:.4f}'
        })

        # Print detailed epoch summary
        tqdm.write(
            f"\nEpoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_metrics['f1_macro']:.4f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/macro", val_metrics["f1_macro"], epoch)
        writer.add_scalar("F1/weighted", val_metrics["f1_weighted"], epoch)

        # Save checkpoint if best F1 (only on main process)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]

            if is_main_process:
                checkpoint_path = run_dir / f"best_model_epoch{epoch}.pt"

                # Unwrap DDP model if using distributed training
                model_to_save = model.module if isinstance(model, DDP) else model

                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": {
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "clip_length": clip_length,
                        "learning_rate": learning_rate,
                        "model_type": model_type
                    }
                }, checkpoint_path)
                tqdm.write(f"  ✓ Saved best model: {checkpoint_path.name} (F1: {best_val_f1:.4f})")

        scheduler.step()

        if early_stopping(val_loss):
            tqdm.write(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()

    # Save config
    config = {
        "video_dir": video_dir,
        "annotation_dir": annotation_dir,
        "dlc_dir": dlc_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "clip_length": clip_length,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_type": model_type,
        "use_amp": use_amp,
        "random_seed": random_seed,
        "best_val_f1": float(best_val_f1)
    }

    if is_main_process:
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nTraining complete. Results saved to {run_dir}")

    # Cleanup distributed training
    cleanup_distributed()

    return run_dir if is_main_process else None


if __name__ == "__main__":
    video_dir = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_mouse_pain/Dec24/REMY2/ALL_VIDEOS/"
    annotation_dir = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_mouse_pain/Dec24/REMY2/ALL_ANNOTATIONS/"
    dlc_dir = video_dir

    train_multimodal_model(
        video_dir,
        annotation_dir,
        dlc_dir,
        output_dir="./checkpoints_multimodal",
        num_epochs=20,
        batch_size=8,  # Conservative batch size for stability
        clip_length=9,  # Clip length for memory efficiency
        learning_rate=5e-4,
        weight_decay=1e-5,
        model_type="standard",
        use_amp=True
    )
