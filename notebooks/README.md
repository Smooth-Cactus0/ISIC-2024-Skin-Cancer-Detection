# Notebooks

Interactive Jupyter notebooks for analysis and experimentation.

| Notebook | Description |
|----------|-------------|
| [`01_eda.ipynb`](01_eda.ipynb) | **Exploratory Data Analysis** — Class distribution, patient-level patterns, demographic analysis, TBP feature exploration, correlation structure, sample images, CV strategy design |
| [`02_baseline_tabular.ipynb`](02_baseline_tabular.ipynb) | **Baseline Tabular Model** — LightGBM on raw + engineered features, patient normalization impact, ABCDE clinical features, SHAP feature importance, out-of-fold predictions |

### Running the Notebooks

```bash
conda activate isic
jupyter notebook notebooks/
```

**Data requirement**: Both notebooks expect competition data in `data/raw/`. Download from Kaggle:
```bash
kaggle competitions download -c isic-2024-challenge -p data/raw/
```
