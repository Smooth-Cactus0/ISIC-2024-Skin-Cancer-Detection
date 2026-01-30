"""
ConvNeXt classifier for ISIC 2024.

ConvNeXt is a modernized CNN that incorporates design principles from ViT
while maintaining the efficiency and inductive biases of convolutions.

Key advantages over ViT for medical imaging:
- Better at capturing local patterns (borders, texture)
- More efficient inference (no quadratic attention)
- Strong hierarchical features (multi-scale)

Used in 3rd place Kaggle solution (ConvNeXtV2 Nano).
"""

import torch
import torch.nn as nn
import timm

from .vit_classifier import FocalLoss  # Reuse focal loss


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt classifier for skin lesion classification.

    Supports multiple ConvNeXt variants via timm:
    - ConvNeXtV2 (recommended): Modern version with improved training
    - ConvNeXt V1: Original version
    - Different sizes: nano, tiny, small, base, large
    """

    def __init__(
        self,
        model_name: str = 'convnextv2_nano.fcmae_ft_in22k_in1k',
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0
    ):
        """
        Initialize ConvNeXt classifier.

        Args:
            model_name: timm model name
            pretrained: Load pretrained weights
            num_classes: Number of output classes (1 for binary)
            dropout: Dropout rate for classifier head
            drop_path_rate: Stochastic depth rate (regularization)
            freeze_backbone: Freeze backbone initially
            freeze_epochs: Number of epochs to keep backbone frozen
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg',  # Global average pooling
            drop_path_rate=drop_path_rate
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            Logits, shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for GBDT ensemble).

        Args:
            x: Input images, shape (B, 3, H, W)

        Returns:
            Feature embeddings, shape (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def update_epoch(self, epoch: int):
        """
        Update current epoch and unfreeze backbone if needed.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        if epoch >= self.freeze_epochs and self.freeze_epochs > 0:
            self.unfreeze_backbone()


class HybridModel(nn.Module):
    """
    Hybrid model combining ConvNeXt and ViT.

    Uses ConvNeXt for local features and ViT for global context.
    Can be used for late fusion or feature concatenation.
    """

    def __init__(
        self,
        convnext_model: ConvNeXtClassifier,
        vit_model: 'ViTClassifier',  # Forward reference
        fusion_type: str = 'concat'
    ):
        """
        Initialize hybrid model.

        Args:
            convnext_model: ConvNeXt classifier
            vit_model: ViT classifier
            fusion_type: 'concat', 'average', or 'weighted'
        """
        super().__init__()

        self.convnext = convnext_model
        self.vit = vit_model
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # Concatenate features before classification
            combined_dim = self.convnext.feature_dim + self.vit.feature_dim
            self.fusion_head = nn.Linear(combined_dim, 1)
        elif fusion_type == 'weighted':
            # Learnable weights for each model
            self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fusion.

        Args:
            x: Input images

        Returns:
            Fused logits
        """
        if self.fusion_type == 'concat':
            # Extract features from both models
            convnext_feat = self.convnext.backbone(x)
            vit_feat = self.vit.backbone(x)

            # Concatenate and classify
            combined = torch.cat([convnext_feat, vit_feat], dim=1)
            logits = self.fusion_head(combined)
            return logits

        elif self.fusion_type == 'average':
            # Average predictions
            convnext_logits = self.convnext(x)
            vit_logits = self.vit(x)
            return (convnext_logits + vit_logits) / 2

        elif self.fusion_type == 'weighted':
            # Weighted combination
            convnext_logits = self.convnext(x)
            vit_logits = self.vit(x)

            # Normalize weights
            w = torch.softmax(self.weights, dim=0)
            return w[0] * convnext_logits + w[1] * vit_logits


def create_convnext_model(
    model_name: str = 'convnextv2_nano.fcmae_ft_in22k_in1k',
    pretrained: bool = True,
    **kwargs
) -> ConvNeXtClassifier:
    """
    Factory function to create ConvNeXt models.

    Args:
        model_name: Model architecture name
        pretrained: Load pretrained weights
        **kwargs: Additional arguments for ConvNeXtClassifier

    Returns:
        ConvNeXtClassifier instance

    Example:
        >>> model = create_convnext_model('convnextv2_nano.fcmae_ft_in22k_in1k')
        >>> logits = model(images)
        >>> embeddings = model.extract_features(images)
    """
    return ConvNeXtClassifier(model_name=model_name, pretrained=pretrained, **kwargs)


# Model configurations
CONVNEXT_CONFIGS = {
    'convnextv2_nano': {
        'model_name': 'convnextv2_nano.fcmae_ft_in22k_in1k',
        'image_size': 224,
        'feature_dim': 640,
        'description': 'ConvNeXtV2 Nano - Used in 3rd place solution'
    },
    'convnextv2_tiny': {
        'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        'image_size': 224,
        'feature_dim': 768,
        'description': 'ConvNeXtV2 Tiny - Good balance'
    },
    'convnextv2_base': {
        'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'image_size': 224,
        'feature_dim': 1024,
        'description': 'ConvNeXtV2 Base - High performance'
    },
    'convnext_small': {
        'model_name': 'convnext_small.fb_in22k_ft_in1k',
        'image_size': 224,
        'feature_dim': 768,
        'description': 'ConvNeXt V1 Small'
    },
    'convnext_base': {
        'model_name': 'convnext_base.fb_in22k_ft_in1k',
        'image_size': 224,
        'feature_dim': 1024,
        'description': 'ConvNeXt V1 Base'
    }
}


if __name__ == "__main__":
    # Test the model
    print("Testing ConvNeXt Classifier...\n")

    # Create model
    model = create_convnext_model('convnextv2_nano.fcmae_ft_in22k_in1k', pretrained=False)
    print(f"✓ Model created: {model.model_name}")
    print(f"  Feature dimension: {model.feature_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    logits = model(dummy_input)
    print(f"\n✓ Forward pass: {dummy_input.shape} -> {logits.shape}")

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"✓ Feature extraction: {dummy_input.shape} -> {features.shape}")

    # Test focal loss (reused from ViT)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.randint(0, 2, (batch_size,)).float()
    loss = focal_loss(logits.squeeze(), targets)
    print(f"\n✓ Focal Loss: {loss.item():.4f}")

    # Test model configs
    print(f"\n{'='*60}")
    print("AVAILABLE CONVNEXT CONFIGURATIONS")
    print(f"{'='*60}")
    for name, config in CONVNEXT_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Feature dim: {config['feature_dim']}")
        print(f"  {config['description']}")

    # Compare with ViT
    print(f"\n{'='*60}")
    print("CONVNEXT vs ViT COMPARISON")
    print(f"{'='*60}")
    print("\nArchitectural Differences:")
    print("  ViT:      Global context via self-attention (quadratic complexity)")
    print("  ConvNeXt: Local patterns via convolutions (linear complexity)")
    print("\nBest For:")
    print("  ViT:      Symmetry, global shape, holistic patterns")
    print("  ConvNeXt: Borders, texture, color gradients, fine details")
    print("\nEnsemble Value:")
    print("  Combining both captures complementary information")
    print("  Error correlation reduced → better ensemble performance")

    print(f"\n{'='*60}")
    print("✓ All tests passed")
