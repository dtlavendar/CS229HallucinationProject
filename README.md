# Multi-Dataset Hallucination Pipeline

**File:** `multi_dataset_pipeline.ipynb`

This notebook runs LLaMA-2-13B on 3 factual True/False datasets, extracts hidden state activations (layers 12–16), and produces a combined visualization and analysis of hallucination geometry.

## Expected Datasets

Upload these 3 CSV files **one at a time** when prompted:

| # | File | Source | Description |
|---|------|--------|-------------|
| 1 | `cities.csv` | Marks et al. "The Geometry of Truth" | "The city of X is in Y." — 1496 city/country statements |
| 2 | `companies_true_or_false.csv` | Marks et al. | Factual claims about companies |
| 3 | `common_claim_true_false.csv` | Marks et al. | Common factual claims |

Each CSV must have at minimum:
- A `statement` column (the text fed to the model)
- A `label` column (`0`/`1`, `True`/`False`, or `"true"`/`"false"`)

Additional columns (e.g. `city`, `country`) are kept but not required.

## Pipeline Steps

1. **Cells 2–3**: Install dependencies, import libraries
2. **Cell 4**: Upload 3 datasets one at a time, convert each to `.parquet`
3. **Cell 5**: Config — model name, batch size, quantization, etc.
4. **Cell 6**: HuggingFace login (requires access token for LLaMA-2)
5. **Cell 8**: Load model **once**, run `model.generate` on all 3 datasets. Per dataset saves:
   - `outputs/<name>/model_outputs.csv` — prompt_num, answer (ground truth), output (model prediction), hallucination (Yes/No/N/A), confidence
   - `outputs/<name>/hs_12_to_17.npz` — hidden state activations for layers 12–16
6. **Cell 9**: 3×4 PCA grid (rows = datasets, cols = Only Hallucinations / All / Non-Hallucinations / Hallucination Model Confidence), saved as `outputs/combined_pca_grid.png`
7. **Cell 10**: Confidence statistics (mean, median, IQR, std) per dataset
8. **Cell 11**: Logistic regression on hallucination-only data — tests if "said True (wrong)" vs "said False (wrong)" are linearly separable at layer 15

## Requirements

- Google Colab with **GPU runtime** (T4 or better)
- HuggingFace account with LLaMA-2 access
- Set `USE_4BIT = True` in Cell 5 if using T4 (16 GB VRAM)

## Saving Results

After running, use Cell 1 (at the top of the notebook) to copy all outputs to Google Drive.
