"""
Vision Transformer (ViT/EVA02) classifier for ISIC 2024.

EVA02 was chosen by top solutions for its:
- Strong ImageNet-21K pretraining
- Good balance of performance and inference speed
- Effective transfer learning to medical images
"""

import torch
import torch.nn as nn
import timm


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Focal loss down-weights easy examples and focuses on hard negatives.
    Critical for ISIC 2024's 1000:1 imbalance.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for class imbalance (typical: 0.25-0.75)
            gamma: Focusing parameter (typical: 2.0)
                   - γ=0: equivalent to BCE
                   - γ>0: down-weights easy examples
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (logits), shape (N,) or (N, 1)
            targets: Ground truth labels (0 or 1), shape (N,)

        Returns:
            Focal loss value
        """
        # Ensure correct shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier for skin lesion classification.

    Supports multiple ViT variants via timm library:
    - EVA02 (recommended): eva02_base_patch14_224.mim_in22k
    - Standard ViT: vit_base_patch16_224.augreg_in21k
    - Tiny variants for faster training: vit_tiny_patch16_224.augreg_in21k
    """

    def __init__(
        self,
        model_name: str = 'eva02_base_patch14_224.mim_in22k',
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0
    ):
        """
        Initialize ViT classifier.

        Args:
            model_name: timm model name
            pretrained: Load pretrained weights
            num_classes: Number of output classes (1 for binary)
            dropout: Dropout rate for classifier head
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
            global_pool='avg'  # Global average pooling
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


class EnsembleViT(nn.Module):
    """
    Ensemble multiple ViT models for improved robustness.

    Used in top solutions to combine predictions from models
    with different architectures or training strategies.
    """

    def __init__(self, models: list):
        """
        Initialize ensemble.

        Args:
            models: List of ViTClassifier instances
        """
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average predictions from all models.

        Args:
            x: Input images

        Returns:
            Averaged logits
        """
        logits = torch.stack([model(x) for model in self.models])
        return logits.mean(dim=0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenate features from all models.

        Args:
            x: Input images

        Returns:
            Concatenated features
        """
        features = [model.extract_features(x) for model in self.models]
        return torch.cat(features, dim=1)


def create_vit_model(
    model_name: str = 'eva02_base_patch14_224.mim_in22k',
    pretrained: bool = True,
    **kwargs
) -> ViTClassifier:
    """
    Factory function to create ViT models.

    Args:
        model_name: Model architecture name
        pretrained: Load pretrained weights
        **kwargs: Additional arguments for ViTClassifier

    Returns:
        ViTClassifier instance

    Example:
        >>> model = create_vit_model('eva02_base_patch14_224.mim_in22k')
        >>> logits = model(images)
        >>> embeddings = model.extract_features(images)
    """
    return ViTClassifier(model_name=model_name, pretrained=pretrained, **kwargs)


# Model variants for experimentation
MODEL_CONFIGS = {
    'eva02_base': {
        'model_name': 'eva02_base_patch14_224.mim_in22k',
        'image_size': 224,
        'feature_dim': 768,
        'description': 'EVA02 Base - Best balance (1st/2nd place used this)'
    },
    'eva02_small': {
        'model_name': 'eva02_small_patch14_224.mim_in22k',
        'image_size': 224,
        'feature_dim': 384,
        'description': 'EVA02 Small - Faster training'
    },
    'vit_base': {
        'model_name': 'vit_base_patch16_224.augreg_in21k',
        'image_size': 224,
        'feature_dim': 768,
        'description': 'Standard ViT Base'
    },
    'vit_small': {
        'model_name': 'vit_small_patch16_224.augreg_in21k',
        'image_size': 224,
        'feature_dim': 384,
        'description': 'Standard ViT Small - Good for quick experiments'
    },
    'vit_tiny': {
        'model_name': 'vit_tiny_patch16_224.augreg_in21k',
        'image_size': 224,
        'feature_dim': 192,
        'description': 'Tiny ViT - Very fast, good for debugging'
    }
}


if __name__ == "__main__":
    # Test the model
    print("Testing ViT Classifier...\n")

    # Create model
    model = create_vit_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False)
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

    # Test focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.randint(0, 2, (batch_size,)).float()
    loss = focal_loss(logits.squeeze(), targets)
    print(f"\n✓ Focal Loss: {loss.item():.4f}")

    # Test model configs
    print(f"\n{'='*60}")
    print("AVAILABLE MODEL CONFIGURATIONS")
    print(f"{'='*60}")
    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Feature dim: {config['feature_dim']}")
        print(f"  {config['description']}")

    print(f"\n{'='*60}")
    print("✓ All tests passed")
