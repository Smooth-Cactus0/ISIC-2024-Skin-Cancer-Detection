# Experiment Configurations (`configs/`)

YAML configuration files for reproducible experiments.

| Config | Model | Image Size | Key Settings |
|--------|-------|-----------|-------------|
| `eva02.yaml` | EVA02 ViT-Small | 336 | Quick experiments, 384-d embeddings |
| `eva02_production.yaml` | EVA02 ViT-Base | 448 | Production training, 768-d embeddings |
| `vit_base.yaml` | ViT-Base | 224 | Standard ViT baseline |
| `convnext_base.yaml` | ConvNeXtV2-Base | 224 | ConvNeXt with stochastic depth |

Each config specifies: model architecture, learning rates (differential for backbone/head), augmentation level, batch size, number of folds, and early stopping criteria.
