"""
Image augmentation strategies for ISIC 2024.

Augmentations are designed to:
1. Increase diversity for the minority class (malignant)
2. Respect medical validity (avoid unrealistic transformations)
3. Account for 3D TBP image characteristics
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, augmentation_strength: str = 'medium'):
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target image size
        augmentation_strength: 'light', 'medium', or 'heavy'

    Returns:
        Albumentations Compose object

    Notes:
        - Skin has no canonical orientation → safe to flip/rotate
        - Moderate color jitter accounts for lighting variation
        - Heavy color changes may alter diagnostic features (use cautiously)
    """
    if augmentation_strength == 'light':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

    elif augmentation_strength == 'medium':
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

    elif augmentation_strength == 'heavy':
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 70), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.GridDistortion(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")


def get_val_transforms(image_size: int = 224):
    """
    Get validation transforms (no augmentation, only resize + normalize).

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_tta_transforms(image_size: int = 224, n_tta: int = 5):
    """
    Get test-time augmentation transforms.

    TTA averages predictions over multiple augmented versions of the same image.
    Typically improves performance by 0.5-1% but increases inference time.

    Args:
        image_size: Target image size
        n_tta: Number of TTA variants (common: 4 flips + 1 original = 5)

    Returns:
        List of Albumentations Compose objects

    Example:
        >>> tta_transforms = get_tta_transforms(224, n_tta=5)
        >>> for tfm in tta_transforms:
        >>>     augmented = tfm(image=image)
        >>>     predictions.append(model(augmented['image']))
        >>> final_pred = np.mean(predictions)
    """
    if n_tta == 1:
        # No TTA, just normal validation transform
        return [get_val_transforms(image_size)]

    # Standard TTA: original + 4 flips
    tta_list = []

    # 1. Original (no flip)
    tta_list.append(A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ]))

    if n_tta >= 2:
        # 2. Horizontal flip
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if n_tta >= 3:
        # 3. Vertical flip
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if n_tta >= 4:
        # 4. Both flips
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if n_tta >= 5:
        # 5. 90 degree rotation
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if n_tta >= 6:
        # 6. 180 degree rotation
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=180, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if n_tta >= 7:
        # 7. 270 degree rotation
        tta_list.append(A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=270, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    return tta_list[:n_tta]


def get_hair_removal_transform():
    """
    Hair removal preprocessing using morphological operations.

    Hair artifacts can obscure lesion borders. This transform attempts to remove them.

    Note: This is a simplified version. For production, consider DullRazor algorithm.

    Returns:
        Function that applies hair removal
    """
    def remove_hair(image, **kwargs):
        """Apply hair removal to image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Black hat filter to detect dark hair
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold to create mask
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Inpaint to remove hair
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result

    return A.Lambda(image=remove_hair, p=1.0)


def get_train_transforms_with_hair_removal(image_size: int = 224):
    """
    Training transforms with hair removal preprocessing.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        get_hair_removal_transform(),
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


class MixUp:
    """
    MixUp augmentation for training.

    Mixes two images and their labels:
        mixed_image = alpha * image1 + (1 - alpha) * image2
        mixed_label = alpha * label1 + (1 - alpha) * label2

    Can help with overfitting on small datasets.

    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.

        Args:
            alpha: Beta distribution parameter (typical: 0.2-0.4)
        """
        self.alpha = alpha

    def __call__(self, image1, image2, label1, label2):
        """
        Apply MixUp to a pair of images.

        Args:
            image1, image2: Input images (tensors)
            label1, label2: Labels (scalars or one-hot)

        Returns:
            mixed_image, mixed_label
        """
        import numpy as np

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix images and labels
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    print("Testing augmentation pipelines...\n")

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Test each augmentation level
    for strength in ['light', 'medium', 'heavy']:
        transforms = get_train_transforms(image_size=224, augmentation_strength=strength)
        augmented = transforms(image=dummy_image)
        print(f"✓ {strength.capitalize()} augmentation: {augmented['image'].shape}")

    # Test validation transform
    val_transform = get_val_transforms(224)
    val_aug = val_transform(image=dummy_image)
    print(f"✓ Validation transform: {val_aug['image'].shape}")

    # Test TTA
    tta_transforms = get_tta_transforms(224, n_tta=5)
    print(f"✓ TTA transforms: {len(tta_transforms)} variants")

    # Test MixUp
    mixup = MixUp(alpha=0.2)
    img1 = np.random.randn(3, 224, 224)
    img2 = np.random.randn(3, 224, 224)
    mixed, mixed_label = mixup(img1, img2, 0, 1)
    print(f"✓ MixUp: {mixed.shape}")

    print("\n✓ Augmentation modules tested successfully")
