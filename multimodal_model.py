"""
Multimodal model for action recognition combining visual and pose features.
Architecture: Visual stream (DinoV2 + Bi-GRU + Attention) + Pose stream (MLP + Attention) -> Fusion -> Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


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


class DinoV2VisualStream(nn.Module):
    """
    Visual feature processing stream using frozen DinoV2 + Bi-GRU + Temporal Attention.

    Architecture:
    1. Frozen DinoV2-large encoder (per-frame feature extraction)
    2. Bi-directional GRU for temporal modeling
    3. Temporal attention for frame-wise importance weighting
    4. FC projection to output dimension
    """

    def __init__(self, output_dim: int = 256, freeze_dinov2: bool = True):
        super(DinoV2VisualStream, self).__init__()

        self.output_dim = output_dim
        self.freeze_dinov2 = freeze_dinov2
        self.dinov2 = None  # Lazy loading
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
            dropout=0.3
        )

        # Temporal attention mechanism
        # Attention over GRU outputs
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.gru_hidden_dim * 2, 256),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Final projection to output dimension
        self.fc_out = nn.Linear(self.gru_hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def _load_dinov2(self):
        """Lazy load DinoV2 model to avoid multi-process conflicts."""
        if self.dinov2 is None:
            import os
            print("Loading DinoV2-large model (this may take a moment)...")
            # Force use of local cache and avoid download conflicts
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
            # Process frames and immediately move to same device as model
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
            (B, output_dim) aggregated visual representation
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

        # Final projection
        output = self.fc_out(attended_features)  # (B, output_dim)
        output = self.dropout(output)

        return output


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
    Multimodal model combining visual and pose features with DinoV2.

    Architecture:
    - Visual stream: DinoV2 (frozen) + Bi-GRU + Temporal Attention
    - Pose stream: MLP with temporal attention on pose features
    - Fusion: Concatenation + MLP
    - Classification head: FC layers

    Input:
    - Visual: (B, 1, T, H, W) or (B, T, H, W) grayscale video clips
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
        dropout: float = 0.5,
        use_dinov2: bool = True
    ):
        super(MultimodalFusionModel, self).__init__()

        # Streams
        if use_dinov2:
            self.visual_stream = DinoV2VisualStream(output_dim=visual_dim)
        else:
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
    Lightweight multimodal model (memory-efficient variant) with DinoV2 support.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4, use_dinov2: bool = True):
        super(MultimodalFusionModelLight, self).__init__()

        self.use_dinov2 = use_dinov2

        if use_dinov2:
            # Use DinoV2 visual stream (lighter version with smaller GRU)
            self.dinov2 = None  # Lazy loading
            self.dinov2_loaded = False

            self.resize = transforms.Resize((224, 224), antialias=True)
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            # Lightweight GRU for temporal modeling
            self.gru = nn.GRU(1024, 256, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
            self.temporal_attention = nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.visual_fc = nn.Linear(512, 128)
        else:
            # Visual stream (lightweight 3D CNN)
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

    def _load_dinov2_light(self):
        """Lazy load DinoV2 for light model."""
        if self.use_dinov2 and not self.dinov2_loaded:
            import os
            print("Loading DinoV2-large for light model...")
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', verbose=False, trust_repo=True)
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
            self.dinov2_loaded = True
            print("DinoV2 loaded for light model!")

    def forward(self, visual, pose):
        """
        Args:
            visual: (B, 1, T, H, W) or (B, T, H, W)
            pose: (B, T, 18)

        Returns:
            logits: (B, num_classes)
        """
        # Visual stream
        if self.use_dinov2:
            # Lazy load DinoV2
            self._load_dinov2_light()
            # Remove channel dimension if present
            if visual.dim() == 5:
                visual = visual.squeeze(1)  # (B, T, H, W)

            B, T, H, W = visual.shape

            # Preprocess for DinoV2 (memory-efficient)
            visual_rgb = visual.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # (B, T, 3, H, W) - expand instead of repeat
            frames_flat = visual_rgb.reshape(B * T, 3, H, W)
            frames_resized = self.resize(frames_flat)
            frames_processed = self.normalize(frames_resized)  # (B*T, 3, 224, 224)

            # Extract DinoV2 features
            with torch.no_grad():
                dinov2_feat = self.dinov2(frames_processed)  # (B*T, 1024)
            dinov2_feat = dinov2_feat.reshape(B, T, 1024)

            # Apply GRU and attention
            gru_out, _ = self.gru(dinov2_feat)  # (B, T, 512)
            attn_scores = self.temporal_attention(gru_out)  # (B, T, 1)
            attn_weights = F.softmax(attn_scores, dim=1)
            visual = torch.sum(gru_out * attn_weights, dim=1)  # (B, 512)
            visual = F.relu(self.visual_fc(visual))  # (B, 128)
        else:
            # 3D CNN path
            if visual.dim() == 4:
                visual = visual.unsqueeze(1)

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
    device: str = "cuda",
    use_dinov2: bool = True
) -> nn.Module:
    """
    Create multimodal model instance with DinoV2 support.

    Args:
        num_classes: Number of action classes
        model_type: "standard" or "light"
        device: Device to place model on
        use_dinov2: Use DinoV2 visual encoder (True) or 3D CNN (False)

    Returns:
        Model instance on specified device
    """
    if model_type == "standard":
        model = MultimodalFusionModel(num_classes=num_classes, use_dinov2=use_dinov2)
    elif model_type == "light":
        model = MultimodalFusionModelLight(num_classes=num_classes, use_dinov2=use_dinov2)
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
