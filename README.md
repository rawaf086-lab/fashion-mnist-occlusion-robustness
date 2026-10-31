# Class-Wise Foreground-Aware Occlusion Evaluation on Fashion-MNIST

CAI421 Deep Learning group project. The project evaluates how the predicted classes and classification accuracy of a trained convolutional neural network (CNN) change when foreground regions of Fashion-MNIST images are hidden at different positions and mask sizes.

> **Project contribution:** evaluation methodology, not a modified CNN architecture or a robustness-training method. The project does **not** claim that the original CNN becomes more robust after occlusion.

## Overview

- **Dataset:** Fashion-MNIST, 70,000 grayscale images (28 × 28), 10 classes.
- **Data split:** 54,000 training / 6,000 validation / 10,000 official test images.
- **Random seed:** `42` (Python, NumPy, PyTorch; with deterministic cuDNN settings).
- **Preprocessing:** `torchvision.transforms.ToTensor()` (pixel values scaled to `[0, 1]`); no mean/std normalization or data augmentation is performed in the supplied notebook.
- **Model:** two 3 × 3 convolutional layers (32 and 64 channels), ReLU, 2 × 2 max pooling after each convolution, followed by fully connected layers (128, 10).
- **Training configuration:** cross-entropy, Adam (`lr=0.001`), batch size 128, 10 epochs.
- **Saved-model test accuracy:** **91.91%** on the 10,000 original test images.
- **Best saved validation accuracy:** **92.95%**.

These reported accuracies refer to the team's **existing, saved model weights**; training a new model may give different values.

## Occlusion evaluation

The provided notebook defines `apply_occlusion` and evaluates nine settings:

| Position | Small | Medium | Large |
| --- | --- | --- | --- |
| Top | 4 × 4 px | 8 × 8 px | 12 × 12 px |
| Center | 4 × 4 px | 8 × 8 px | 12 × 12 px |
| Bottom | 4 × 4 px | 8 × 8 px | 12 × 12 px |

The method identifies a foreground using the heuristic `image > 0.1`, selects an occlusion center near the target vertical location (20%, 50%, or 80% of the foreground bounding range), and calculates the fraction of detected foreground pixels covered by the mask. This is a **threshold-based foreground estimate, not semantic segmentation**.

Analyses include overall and per-class accuracy drops, comparisons at approximately matched foreground coverage, fixed-image-center versus foreground-aware masking, paired bootstrap confidence intervals, training curves, a confusion matrix, and class substitutions after large center occlusion.

## Files

- `Final_FashionMnist.ipynb`: complete Colab notebook, with code cells numbered 1–31 and saved outputs.
- `README.md`: project description and instructions.
- `Dependency_List.txt`: Python dependency list needed by the notebook.
- `Fixed_Random_Seeds.py`: documents and reproduces the same Python, NumPy, PyTorch, and CUDA seed configuration as Cell 7.

The instructions specify **a dependency list** and **fixed random seeds**, but do not prescribe exact filenames. This repository uses descriptive names `Dependency_List.txt` and `Fixed_Random_Seeds.py` to make these requirements explicit. The notebook already sets `SEED=42` in Cell 7 (and `random_state=42` for data splitting in Cell 4). Do not run the standalone seed script in a separate shell process expecting it to change the Colab kernel: run the notebook's Cell 7 instead.

The repository may also include the existing `best_fashion_cnn.pth` checkpoint and selected `results/` CSV/PNG outputs if the group decides to distribute them. They are **not embedded in the notebook** and are **not automatically included** just by uploading the `.ipynb` file.

## Required input files and Drive layout

The notebook uses **Google Colab** and expects the following paths, which are currently hard-coded in code cell 2:

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
        └── ... additional saved figures and analysis CSVs
```

The dataset was obtained as **Parquet** files from the Hugging Face Fashion-MNIST dataset. Obtain the correct train/test Parquet splits, and use the **exact filenames expected by code cell 3**, or update the two filename strings before running. The original dataset is available from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) (MIT licensed); the notebook, however, specifically expects Parquet records containing `image` (with image bytes) and `label` columns.

**Important for shared Google Drive folders:** a folder shared with your Google account does not necessarily appear at `/content/drive/MyDrive/DLproject1`. Copy/add it to your own My Drive, or change `dataset_path` in cell 2. If the shared directory is read-only, use a **writable copy** for runs that save outputs; some cells use `to_csv` or `savefig` and may fail on read-only Drive folders.

## Run the project in Google Colab

1. Open `Final_FashionMnist.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** if available. CPU operation is supported but may be slower.
3. Ensure the paths and inputs above exist in your accessible Google Drive folder. Install packages if Colab is missing them (for example, `!pip install -r "Dependency_List.txt"` after uploading `Dependency_List.txt`, or use the Colab-installed dependencies).
4. The notebook includes its own random seed setup in Cell 7. The separate `Fixed_Random_Seeds.py` is optional reference code; running it outside the Colab kernel does not seed the notebook. Execute cells **1 through 31 in order** only on a **writable copy** of the project folder. The training function in cell 11 is **defined but not called automatically**; cell 12 loads the saved `.pth` model.
5. The baseline is checked in cell 13; training history is shown in cell 14. The later cells analyze occlusion and saved results.
6. Cells **29**, **30**, and **31** display training curves, clean-test confusion matrix/classification report, and class substitution analysis, respectively.

### Reusing saved results versus recomputing

- With the saved CSV files available, some analysis cells **load the precomputed per-image data** rather than repeating image-by-image model inference. This can reproduce the saved analyses quickly but **does not constitute a fresh independent rerun of all experiments**.
- To recompute the occlusion experiments, use a **separate writable copy** of the Drive directory and remove only the *copied* detailed result files that trigger loading (such as `per_image_occlusion_results.csv` and `fixed_vs_foreground_per_image.csv`). Keep the original data and saved model; review the CSV verification logic before regenerating derived outputs.
- Some cells overwrite summary CSVs or PNGs when rerun. Do **not** use `Run all` against the team's sole shared results folder.
- The notebook currently expects an existing `best_fashion_cnn.pth` and `training_history.csv`. The optional training function in cell 11 can train a new CNN **only when explicitly called** with a new, unused `save_path`. It does **not** automatically reproduce the original model or the original training-history file. Before claiming full clean-room reproducibility, make the original weights/history available or document and execute a separate training-and-export workflow.

## Expected saved outputs

When using the original shared model and results, the verified outcomes include:

- Clean-image test accuracy: **91.91%**.
- Best validation accuracy: **92.95%**.
- Full foreground-aware occlusion evaluation: 9 conditions × 10,000 test images = **90,000 records**.
- Fixed-center comparison: 3 sizes × 10,000 images = **30,000 records**.
- Per-class paired bootstrap confidence intervals using **1,000 resamples** for each size/class combination.

The notebook produces or reads figures such as `training_curves.png`, `confusion_matrix_clean.png`, `classwise_heatmap.png`, `foreground_vs_accuracy_drop.png`, and `bootstrap_ci_small.png` / `bootstrap_ci_medium.png` / `bootstrap_ci_large.png` in `DLproject1/results/`.

## Data citation and license

Fashion-MNIST: Han Xiao, Kashif Rasul, and Roland Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms* (2017), arXiv:1708.07747. Official repository and dataset license: <https://github.com/zalandoresearch/fashion-mnist> (MIT License).

## Notes

- The notebook and its output figures document a specific evaluation study. Accuracy differences from fixed versus foreground-aware masking reflect **different test occlusions**, not retraining or improving the neural network itself.
- This repository is part of the group's CAI421 coursework. Include the required course acknowledgment / AI-use disclosure in the **course submission** as instructed by the instructor.
