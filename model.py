"""
3D CNN model for action recognition.
Designed to capture spatiotemporal features from grayscale video clips.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        clip_length: int = 16,
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
        clip_length: int = 16,
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


def create_model(
    num_classes: int = 7,
    clip_length: int = 16,
    model_type: str = "standard",
    device: str = "cuda"
) -> nn.Module:
    """
    Create model instance.

    Args:
        num_classes: Number of action classes
        clip_length: Temporal clip length
        model_type: "standard" or "light"
        device: Device to place model on

    Returns:
        Model instance on specified device
    """
    if model_type == "standard":
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
