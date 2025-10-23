"""
Multimodal model for action recognition combining visual and pose features.
Architecture: Visual stream (3D CNN) + Pose stream (MLP) -> Fusion -> Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D convolutional block with batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
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
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class PoseStream(nn.Module):
    """
    Pose feature processing stream.
    Takes pose features and extracts meaningful representations.
    """

    def __init__(self, input_dim: int = 18, output_dim: int = 256):
        """
        Args:
            input_dim: Dimension of pose features per frame (18: 8 lengths + 10 angles)
            output_dim: Output dimension for fusion
        """
        super(PoseStream, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Temporal processing of pose features
        # Process each frame's pose features
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        # Temporal attention over frames
        self.temporal_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final projection
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, 18) pose features over temporal dimension

        Returns:
            (B, output_dim) pose representation
        """
        batch_size, seq_len, pose_dim = x.shape

        # Process each frame's pose
        x = x.reshape(batch_size * seq_len, pose_dim)  # (B*T, 18)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)  # (B*T, 256)
        x = self.dropout2(x)

        x = x.reshape(batch_size, seq_len, 256)  # (B, T, 256)

        # Temporal attention pooling
        attention_weights = self.temporal_attention(x)  # (B, T, 1)
        x = torch.sum(x * attention_weights, dim=1)  # (B, 256)

        # Final projection
        x = self.fc_out(x)  # (B, output_dim)

        return x


class VisualStream(nn.Module):
    """
    Visual feature processing stream using 3D CNN.
    """

    def __init__(self, output_dim: int = 256):
        super(VisualStream, self).__init__()

        # Visual stream (similar to original 3D CNN)
        self.layer1 = nn.Sequential(
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.layer2 = nn.Sequential(
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.layer3 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.layer4 = nn.Sequential(
            Conv3DBlock(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Projection to output dim
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T, H, W) visual frames

        Returns:
            (B, output_dim) visual representation
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        x = self.fc(x)  # (B, output_dim)

        return x


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model combining visual and pose features.

    Architecture:
    - Visual stream: 3D CNN processing video frames
    - Pose stream: MLP with temporal attention on pose features
    - Fusion: Concatenation + MLP
    - Classification head: FC layers

    Input:
    - Visual: (B, 1, T, H, W) grayscale video clips
    - Pose: (B, T, 18) pose features (8 lengths + 10 angles)

    Output:
    - (B, num_classes) action predictions
    """

    def __init__(
        self,
        num_classes: int = 7,
        visual_dim: int = 256,
        pose_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.5
    ):
        super(MultimodalFusionModel, self).__init__()

        # Streams
        self.visual_stream = VisualStream(output_dim=visual_dim)
        self.pose_stream = PoseStream(input_dim=18, output_dim=pose_dim)

        # Fusion
        self.fusion_fc1 = nn.Linear(visual_dim + pose_dim, fusion_dim)
        self.fusion_bn1 = nn.BatchNorm1d(fusion_dim)
        self.fusion_dropout1 = nn.Dropout(dropout)

        self.fusion_fc2 = nn.Linear(fusion_dim, 256)
        self.fusion_bn2 = nn.BatchNorm1d(256)
        self.fusion_dropout2 = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, visual, pose):
        """
        Forward pass.

        Args:
            visual: (B, 1, T, H, W) or (B, T, H, W) visual frames
            pose: (B, T, 18) pose features

        Returns:
            logits: (B, num_classes)
        """
        # Add channel dimension if needed
        if visual.dim() == 4:
            visual = visual.unsqueeze(1)

        # Process each stream
        visual_feat = self.visual_stream(visual)  # (B, 256)
        pose_feat = self.pose_stream(pose)  # (B, 256)

        # Fusion
        fused = torch.cat([visual_feat, pose_feat], dim=1)  # (B, 512)

        fused = self.fusion_fc1(fused)
        fused = self.fusion_bn1(fused)
        fused = F.relu(fused)
        fused = self.fusion_dropout1(fused)

        fused = self.fusion_fc2(fused)
        fused = self.fusion_bn2(fused)
        fused = F.relu(fused)
        fused = self.fusion_dropout2(fused)

        # Classification
        logits = self.classifier(fused)  # (B, num_classes)

        return logits


class MultimodalFusionModelLight(nn.Module):
    """
    Lightweight multimodal model (memory-efficient variant).
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super(MultimodalFusionModelLight, self).__init__()

        # Visual stream (lightweight)
        self.visual_layer1 = nn.Sequential(
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.visual_layer2 = nn.Sequential(
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.visual_layer3 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.visual_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.visual_fc = nn.Linear(128, 128)

        # Pose stream (lightweight)
        self.pose_fc1 = nn.Linear(18, 64)
        self.pose_dropout1 = nn.Dropout(dropout)

        self.pose_temporal_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        self.pose_fc2 = nn.Linear(64, 128)

        # Fusion
        self.fusion_fc = nn.Linear(128 + 128, 256)
        self.fusion_dropout = nn.Dropout(dropout)

        # Classification
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, visual, pose):
        """
        Args:
            visual: (B, 1, T, H, W) or (B, T, H, W)
            pose: (B, T, 18)

        Returns:
            logits: (B, num_classes)
        """
        # Add channel if needed
        if visual.dim() == 4:
            visual = visual.unsqueeze(1)

        # Visual stream
        visual = self.visual_layer1(visual)
        visual = self.visual_layer2(visual)
        visual = self.visual_layer3(visual)
        visual = self.visual_avgpool(visual)
        visual = visual.view(visual.size(0), -1)
        visual = F.relu(self.visual_fc(visual))  # (B, 128)

        # Pose stream
        batch_size, seq_len, _ = pose.shape
        pose = pose.reshape(batch_size * seq_len, 18)  # (B*T, 18)
        pose = self.pose_fc1(pose)
        pose = F.relu(pose)
        pose = self.pose_dropout1(pose)
        pose = pose.reshape(batch_size, seq_len, 64)  # (B, T, 64)

        # Temporal attention
        attn = self.pose_temporal_attention(pose)  # (B, T, 1)
        pose = torch.sum(pose * attn, dim=1)  # (B, 64)
        pose = F.relu(self.pose_fc2(pose))  # (B, 128)

        # Fusion
        fused = torch.cat([visual, pose], dim=1)  # (B, 256)
        fused = self.fusion_fc(fused)
        fused = F.relu(fused)
        fused = self.fusion_dropout(fused)

        # Classification
        logits = self.classifier(fused)

        return logits


def create_multimodal_model(
    num_classes: int = 7,
    model_type: str = "standard",
    device: str = "cuda"
) -> nn.Module:
    """
    Create multimodal model instance.

    Args:
        num_classes: Number of action classes
        model_type: "standard" or "light"
        device: Device to place model on

    Returns:
        Model instance on specified device
    """
    if model_type == "standard":
        model = MultimodalFusionModel(num_classes=num_classes)
    elif model_type == "light":
        model = MultimodalFusionModelLight(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test standard model
    model = create_multimodal_model(num_classes=7, model_type="standard", device=device)
    print(model)

    # Test forward pass
    visual = torch.randn(4, 16, 256, 256).to(device)  # (B, T, H, W)
    pose = torch.randn(4, 16, 18).to(device)  # (B, T, 18)

    output = model(visual, pose)
    print(f"Visual shape: {visual.shape}")
    print(f"Pose shape: {pose.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
