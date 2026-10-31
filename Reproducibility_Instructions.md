# Reproducibility Instructions
**CAI421 — Class-Wise Foreground-Aware Occlusion Evaluation on Fashion-MNIST**

This document explains how to inspect the team's recorded results, rerun the provided evaluation code, and distinguish those procedures from training a new CNN. It accompanies `Final_FashionMnist.ipynb`.

## 1. What is required

- **Environment:** Google Colab, Python 3, preferably a T4 GPU (CPU may be slower).
- **Code:** `Final_FashionMnist.ipynb`.
- **Dependencies:** packages listed in `Dependency_List.txt`: `numpy`, `pandas`, `pyarrow`, `Pillow`, `matplotlib`, `scikit-learn`, `torch`, `torchvision`, and `ipython`. The original package versions were not recorded, so bit-for-bit reproduction across different environments is not guaranteed.
- **Random seed:** `42`. Cell 4 uses `random_state=42` for the stratified training/validation split, and Cell 7 seeds Python, NumPy, PyTorch, and CUDA and sets deterministic cuDNN options. `Fixed_Random_Seeds.py` documents the same settings; it does not have to be executed separately.
- **Inputs:** the Fashion-MNIST train/test Parquet files, and—when reproducing the team's *recorded* results—the original `best_fashion_cnn.pth` checkpoint plus the relevant saved CSV files.

The notebook **does not retrain automatically**: Cell 11 defines a training function without calling it, and Cell 12 loads the existing checkpoint.

## 2. Prepare the files

The notebook expects the following layout inside **your own Google Drive My Drive**:

```text
MyDrive/
└── DLproject1/
    ├── train-00000-of-00001 (1).parquet
    ├── test-00000-of-00001 (1).parquet
    ├── best_fashion_cnn.pth
    └── results/
        ├── training_history.csv
        ├── baseline_predictions.csv
        ├── per_image_occlusion_results.csv
        ├── occlusion_summary.csv
        ├── occlusion_classwise.csv
        ├── fixed_vs_foreground_per_image.csv
        ├── fixed_vs_foreground_comparison.csv
        ├── classwise_method_comparison.csv
        ├── bootstrap_confidence_intervals.csv
        └── [other analysis CSVs and PNGs]
```

1. Obtain the Fashion-MNIST data from the original source or the Hugging Face Parquet distribution used for this project. The two files must have an `image` column containing image bytes and a `label` column, with **60,000** training images and **10,000** test images.
2. Obtain the team's original trained checkpoint and saved result files from the project's authorized Google Drive folder. They are **not included automatically** just because the notebook is uploaded to GitHub.
3. Keep the exact filenames above, or adjust Cells 2–3 to match your own copies.
4. If you access a folder shared from another Google account, it may not appear under `/content/drive/MyDrive/DLproject1`. Make a copy in your own My Drive or change `dataset_path` accordingly.
5. **Work on a writable copy of `DLproject1`.** Several analysis cells overwrite existing CSV and PNG files when executed. Do not run the entire notebook against the group's sole shared results folder.

Original Fashion-MNIST project and license: https://github.com/zalandoresearch/fashion-mnist (MIT). The supplied notebook expects the **Parquet** representation, not the source repository's raw binary format.

## 3. Run the existing model and reproduce the reported tables/figures

1. Open `Final_FashionMnist.ipynb` in **Google Colab**.
2. Choose **Runtime → Change runtime type → T4 GPU** if available.
3. Make the project files available in your Drive copy under the layout above.
4. Mount Google Drive by running **Cell 1**, and confirm that **Cell 2** points to the right `dataset_path`.
5. Run Cells **3–14 in numerical order**:
   - Cells 3–8: load data, reproduce the stratified 54,000/6,000 split, apply `ToTensor()` (values in `[0,1]`), set the seed and build DataLoaders.
   - Cells 9–11: define the model, optimizer settings and **optional** training function. Cell 11 does **not** train.
   - Cell 12: load `best_fashion_cnn.pth`.
   - Cell 13: evaluate the original 10,000 test images and verify `baseline_predictions.csv` if it exists.
   - Cell 14: display the saved 10-epoch training history from `training_history.csv`.
6. Run Cells **15–28 in numerical order** to display masks, load or compute occlusion evaluations and comparisons, verify record counts, create summary tables/plots, and calculate paired bootstrap confidence intervals.
7. Run Cells **29–31** for training/validation curves, clean-test confusion matrix and classification report, and the class-substitution summary under Center–Large occlusion.

When all previously saved detailed result files are present, **Cell 18** and **Cell 26** reuse those files instead of recomputing every model prediction. This reproduces the *saved analyses*, not a new independent inference run.

### Expected results using the team's existing checkpoint

| Result | Expected value / output |
| --- | --- |
| Dataset split | Train **54,000**, Validation **6,000**, Test **10,000** |
| Best saved validation accuracy | **92.95%** |
| Original clean-test accuracy | **91.91%** |
| Foreground-aware conditions | 3 positions × 3 sizes = **9** |
| Foreground-aware per-image rows | **90,000** |
| Class-wise occlusion summary rows | **90** |
| Fixed-center per-image comparison rows | **30,000** |
| Paired bootstrap | **1,000** resamples per class/size; see `bootstrap_confidence_intervals.csv` |

Key outputs under `DLproject1/results/` include `training_curves.png`, `confusion_matrix_clean.png`, `classwise_heatmap.png`, `foreground_vs_accuracy_drop.png`, and `bootstrap_ci_small.png`, `bootstrap_ci_medium.png`, `bootstrap_ci_large.png`.

## 4. Independently recompute the occlusion inference (optional)

**Use a separate, writable copy; preserve the original files.**

1. Make sure that the original input Parquet files and `best_fashion_cnn.pth` are present. Run Cells 1–17 in order; do **not** invoke `train_cnn()`.
2. In **the copied** `results/` folder only, remove or move `per_image_occlusion_results.csv` to a backup location. This causes Cell **18** to perform the 9 × 10,000 foreground-aware predictions rather than load the existing detailed CSV.
3. Cell 18 compares reconstructed summary/class-wise values with existing saved summary files. If assertions fail, inspect the cause rather than overwriting the originals or bypassing verification.
4. For fresh fixed-center inference, in the **copied** folder remove or move `fixed_vs_foreground_per_image.csv` before running Cell **26**. The cell needs the foreground-aware per-image data generated by Cell 18.
5. Run Cells 19–31 in order as needed for the accompanying analyses and plots. Cell **27** recalculates bootstrap intervals with `np.random.default_rng(42)` and **1,000** resamples.
6. Compare the outputs against the recorded CSV files. Differences may indicate altered inputs, code, package versions, or compute environment.

**Do not delete files from the original shared Drive folder.** Cells 18, 21–22, 24–28 and 29–30 can save or overwrite derived CSV/PNG files in the working directory; Cell 27 also rewrites the copied paired comparison CSV.

## 5. Training a new model is a separate experiment

The published **91.91% test** and **92.95% validation** accuracies are associated with the team's previously saved `best_fashion_cnn.pth`.

- **Cell 11 defines** `train_cnn(...)` but deliberately does **not call** it.
- Training a new model requires explicitly creating a **fresh `FashionCNN` and optimizer** and invoking the function with a **new, unused checkpoint path**. The function rejects an existing `save_path` to protect saved weights.
- It returns an in-memory `history` and training duration. It **does not automatically export a new `training_history.csv`**. A separate history-export step is required for new training runs.
- Retraining on a different machine or software environment can produce different results despite the seed. Therefore, this repository documents how to reproduce the team's **saved-model evaluation** and how to recompute its tests; it does **not** claim that the exact historical checkpoint can be regenerated bit-for-bit with a single Run all.

Do **not** overwrite `best_fashion_cnn.pth` or the team's original `results/training_history.csv` during demonstrations.

## 6. Notes for reviewers

- The project studies **evaluation robustness** under masked clothing regions. It does not retrain or improve the CNN after applying occlusion.
- `Foreground-aware` uses an image-intensity threshold (>0.1) as a foreground heuristic, not manual segmentation or semantic masks.
- `Cell 29` reads `training_history.csv`, `Cell 30` reads `baseline_predictions.csv`, and `Cell 31` reads `per_image_occlusion_results.csv`. Their outputs are derived from those saved records, without retraining.
- If the repository is **Private**, reviewers need access permission. Public visibility makes the code accessible but does not automatically provide access to external private Google Drive files.
