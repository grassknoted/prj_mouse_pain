"""
Action recognition models with support for both 3D CNN and DinoV2 (frozen) + temporal modeling.
DinoV2 model uses frozen pretrained features with Bi-GRU + Attention for temporal aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Conv3DBlock(nn.Module):
    """
    3D convolutional block with batch norm and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "relu"
    ):
        super(Conv3DBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Mouse3DCNN(nn.Module):
    """
    3D CNN for mouse action recognition.

    Architecture:
    - 4 3D conv blocks with increasing channels
    - Max pooling after each block
    - Temporal pooling to aggregate time dimension
    - FC layers for classification

    Input: (B, 1, T, H, W) grayscale video clips
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 7,
        clip_length: int = 17,
        input_height: int = 256,
        input_width: int = 256,
        dropout: float = 0.5
    ):
        super(Mouse3DCNN, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Input: (B, 1, T, H, W)

        # Block 1: Conv 3D + Pool
        self.layer1 = nn.Sequential(
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 32, T, H/2, W/2)
        )

        # Block 2: Conv 3D + Pool
        self.layer2 = nn.Sequential(
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # (B, 64, T/2, H/4, W/4)
        )

        # Block 3: Conv 3D + Pool
        self.layer3 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # (B, 128, T/4, H/8, W/8)
        )

        # Block 4: Conv 3D + Pool
        self.layer4 = nn.Sequential(
            Conv3DBlock(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # (B, 256, T/8, H/16, W/16)
        )

        # Calculate feature map size after convolutions
        # For clip_length=16, input H=256, W=256
        # After layer1: T=16, H=128, W=128
        # After layer2: T=8, H=64, W=64
        # After layer3: T=4, H=32, W=32
        # After layer4: T=2, H=16, W=16
        final_temporal = max(1, clip_length // 8)
        final_spatial = max(1, input_height // 16)

        self.feature_size = 256 * final_temporal * final_spatial * final_spatial

        # Global average pooling + FC layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # FC head
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 1, T, H, W) or (B, T, H, W)

        Returns:
            logits: (B, num_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 4:
            x = x.unsqueeze(1)

        # Convolutional blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)  # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        # FC layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)  # (B, num_classes)

        return x


class Mouse3DCNNLight(nn.Module):
    """
    Lightweight version of 3D CNN for action recognition.
    Useful if memory is constrained.

    Input: (B, 1, T, H, W) grayscale video clips
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 7,
        clip_length: int = 17,
        dropout: float = 0.4
    ):
        super(Mouse3DCNNLight, self).__init__()

        # Block 1
        self.layer1 = nn.Sequential(
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # Block 2
        self.layer2 = nn.Sequential(
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        # Block 3
        self.layer3 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_out(x)

        return x


class DinoV2TemporalModel(nn.Module):
    """
    Action recognition model using frozen DinoV2 + Bi-GRU + Temporal Attention.

    Architecture:
    1. Frozen DinoV2-large encoder (per-frame feature extraction)
    2. Bi-directional GRU for temporal modeling
    3. Temporal attention for frame-wise importance weighting
    4. FC classification head

    Input: (B, T, H, W) or (B, 1, T, H, W) grayscale video clips
    Output: (B, num_classes) action logits
    """

    def __init__(
        self,
        num_classes: int = 7,
        freeze_dinov2: bool = True,
        dropout: float = 0.4
    ):
        super(DinoV2TemporalModel, self).__init__()

        self.num_classes = num_classes
        self.freeze_dinov2 = freeze_dinov2

        # Lazy loading to avoid multi-process conflicts
        self.dinov2 = None
        self.dinov2_dim = 1024  # DinoV2-large output dimension

        # Preprocessing for DinoV2 (expects 224x224 RGB images, normalized)
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Bi-directional GRU for temporal modeling
        self.gru_hidden_dim = 512
        self.bi_gru = nn.GRU(
            input_size=self.dinov2_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.gru_hidden_dim * 2, 256),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Classification head
        self.fc1 = nn.Linear(self.gru_hidden_dim * 2, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, num_classes)

    def _load_dinov2(self):
        """Lazy load DinoV2 model to avoid multi-process conflicts."""
        if self.dinov2 is None:
            import os
            print("Loading DinoV2-large model for visual-only model...")
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=False, trust_repo=True)

            # Freeze DinoV2 parameters
            if self.freeze_dinov2:
                for param in self.dinov2.parameters():
                    param.requires_grad = False
                self.dinov2.eval()

            print("DinoV2 loaded successfully!")

    def preprocess_frames(self, frames):
        """
        Memory-efficient preprocessing of grayscale frames for DinoV2.

        Args:
            frames: (B, T, H, W) grayscale frames in [0, 1]

        Returns:
            (B, T, 3, 224, 224) preprocessed RGB frames
        """
        B, T, H, W = frames.shape

        # Process in a more memory-efficient way
        # Convert grayscale to RGB by repeating channels
        frames_rgb = frames.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # (B, T, 3, H, W) - uses expand instead of repeat

        # Resize and normalize all at once
        frames_flat = frames_rgb.reshape(B * T, 3, H, W)  # (B*T, 3, H, W)
        frames_resized = self.resize(frames_flat)  # (B*T, 3, 224, 224)
        frames_normalized = self.normalize(frames_resized)  # (B*T, 3, 224, 224)

        # Reshape back
        frames_processed = frames_normalized.reshape(B, T, 3, 224, 224)
        return frames_processed

    def extract_dinov2_features(self, frames):
        """
        Extract DinoV2 features for each frame (memory-safe).

        Args:
            frames: (B, T, 3, 224, 224) preprocessed frames

        Returns:
            (B, T, 1024) DinoV2 features
        """
        B, T, C, H, W = frames.shape

        # Reshape to process all frames in batch
        frames_flat = frames.reshape(B * T, C, H, W)  # (B*T, 3, 224, 224)

        # Extract features with frozen DinoV2 (always no_grad for frozen model)
        self.dinov2.eval()  # Ensure eval mode
        with torch.no_grad():
            features = self.dinov2(frames_flat)  # (B*T, 1024)

        # Reshape back to (B, T, 1024)
        features = features.reshape(B, T, self.dinov2_dim)

        return features

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, T, H, W) or (B, 1, T, H, W) grayscale frames in [0, 1]

        Returns:
            logits: (B, num_classes)
        """
        # Lazy load DinoV2 on first forward pass
        self._load_dinov2()

        # Remove channel dimension if present
        if x.dim() == 5:
            x = x.squeeze(1)  # (B, T, H, W)

        B, T, H, W = x.shape

        # Preprocess frames for DinoV2
        frames_processed = self.preprocess_frames(x)  # (B, T, 3, 224, 224)

        # Extract DinoV2 features per frame
        dinov2_features = self.extract_dinov2_features(frames_processed)  # (B, T, 1024)

        # Apply Bi-GRU for temporal modeling
        gru_out, _ = self.bi_gru(dinov2_features)  # (B, T, 1024) where 1024 = gru_hidden_dim * 2

        # Temporal attention
        attention_scores = self.temporal_attention(gru_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)

        # Weighted sum of GRU outputs
        attended_features = torch.sum(gru_out * attention_weights, dim=1)  # (B, 1024)

        # Classification head
        x = self.fc1(attended_features)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)  # (B, num_classes)

        return x


def create_model(
    num_classes: int = 7,
    clip_length: int = 16,
    model_type: str = "standard",
    device: str = "cuda",
    use_dinov2: bool = True
) -> nn.Module:
    """
    Create model instance with DinoV2 support.

    Args:
        num_classes: Number of action classes
        clip_length: Temporal clip length
        model_type: "standard" or "light" (ignored when use_dinov2=True)
        device: Device to place model on
        use_dinov2: Use DinoV2 temporal model (True) or 3D CNN (False)

    Returns:
        Model instance on specified device
    """
    if use_dinov2:
        # DinoV2-based temporal model (ignores model_type)
        model = DinoV2TemporalModel(num_classes=num_classes)
    elif model_type == "standard":
        model = Mouse3DCNN(num_classes=num_classes, clip_length=clip_length)
    elif model_type == "light":
        model = Mouse3DCNNLight(num_classes=num_classes, clip_length=clip_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(num_classes=7, clip_length=16, model_type="standard", device=device)
    print(model)

    # Test with random input
    x = torch.randn(4, 16, 256, 256).to(device)  # (B, T, H, W)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
