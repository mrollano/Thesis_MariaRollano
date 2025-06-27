# Design, Implementation and Analysis of a Human-Robot Trust Methodology Using Electroencephalography and Artificial Intelligence Models

## General Description

This project is part of a Master's Thesis conducted by **María Rollano Corroto**. It aims to replicate and extend the study presented in the paper:

> **"Trust Recognition in Human-Robot Cooperation Using EEG"**  
> Caiyue Xu, Changming Zhang, Yanmin Zhou, Zhipeng Wang, Ping Lu, Bin He  
> [arXiv:2403.05225](https://doi.org/10.48550/arXiv.2403.05225)

The main objective of this thesis is to infer trust levels using electroencephalography (EEG) signals combined with artificial intelligence (AI) models in a human-robot cooperation task, using the publicly available EEGTrust dataset provided by the original authors.

To that end, we replicate the reference study, following the same methodology, dataset, and experimental pipeline. Furthermore, we extend their work by proposing six systematic experimental approaches that vary the validation mode, number of input features, and data segmentation strategy. Each configuration serves as a distinct pipeline on which a consistent set of models is trained and evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)
- Convolutional Neural Network (CNN)
- Vision Transformer (ViT)

The table below summarizes the six approaches:

| Approach | Validation Type | Feature Type       | Data Split Type |
|----------|------------------|--------------------|------------------|
| 1        | Within-subject   | One characteristic | Slice-wise       |
| 2        | Within-subject   | One characteristic | Trial-wise       |
| 3        | Within-subject   | Multiple features  | Slice-wise       |
| 4        | Within-subject   | Multiple features  | Trial-wise       |
| 5        | Between-subject  | One characteristic |       |
| 6        | Between-subject  | Multiple features  |       |


## Table of Contents

1. [Project Structure](#-project-structure)
2. [Requirements](#-requirements)
3. [Authors](#-authors)
4. [License](#-license)

## Project Structure
```
Thesis_MariaRollano-main/
├── Data/
│   ├── Preprocessing/
│   └── Scripts/
│       ├── 1. Load_data/
│       ├── 2. Transformations/
│       ├── 3. Split_data/
│       └── 4. Load/
├── Models/
│   ├── 1. knn/
│   ├── 2. nb/
│   ├── 3. svm/
│   ├── 4. forest/
│   ├── 5. XGBoost/
│   ├── 6. cnn/
│   └── 7. transformer/
├── README.md
```

---

### Script Overview
Although each script includes a docstring describing its purpose and functionality, the following section provides a high-level overview of the content and structure of each folder. This summary is intended to help readers quickly understand the role of each component before diving into the source code.

- `Data/Preprocessing/final preprocessing TFM.mlx`: raw EEG signal preprocessing in MATLAB using EEGLAB.

- `Data/Scripts/1. Load_data/`  
  - `load_data.py`: Loads preprocessed EEG data and organizes it into structured formats for further analysis.

- `Data/Scripts/2. Transformations/`
  - `a. One_characteristic/`:  
    - `transform_one_characteristic.py`: Extracts Band Differential Entropy (BDE) features for classical machine learning models.  
    - `transform_toGrid.py`: Converts BDE features into a spatial grid format suitable for CNNs and ViTs.  
  - `b. Multiple_characteristics/`:  
    - `transform_multiple_characteristic.py`: Extracts multiple types of features from EEG segments.  
    - `transform_cnn_transformer_multiple.py`: Applies grid transformation to multi-feature inputs for CNNs and ViTs.

- `Data/Scripts/3. Split_data/`
  - `a. one_characteristic/`:  
    - `slice_wise/split_slice_wise.py`: Performs slice-wise splitting for single-feature data (classical models).  
    - `slice_wise/transformer_split_slice_wise.py`: Same as above, adapted for CNN/ViT input format.  
    - `trial_wise/split_trial_wise.py`: Performs trial-wise splitting for single-feature data (classical models).  
    - `trial_wise/transformer_split_trial_wise.py`: Same as above, adapted for CNN/ViT input format.
  - `b. multiple_characteristic/`:  
    - `slice_wise/split_slice_wise.py`: Slice-wise splitting for multiple features.  
    - `slice_wise/transformer_split_slice_wise.py`: CNN/ViT-compatible slice-wise splitting.  
    - `trial_wise/split_trial_wise.py`: Trial-wise splitting for multiple features.  
    - `trial_wise/transformer_split_trial_wise.py`: CNN/ViT-compatible trial-wise splitting.

- `Data/Scripts/4. Load/`  
  - `load_within.py`: Loads datasets for within-subject evaluation (classical format).  
  - `load_between.py`: Loads datasets for between-subject evaluation (classical format).  
  - `toGrid_load_within.py`: Loads datasets for within-subject evaluation in grid format (for CNN/ViT).  
  - `toGrid_load_between.py`: Same as above, for between-subject evaluation.

---

### Model Scripts

- `Models/`: Contains implementation scripts for each classification model. For each model:

  - `knn/`, `nb/`, `svm/`, `forest/`, `XGBoost/`:  
    - Each contains two scripts:  
      - `*_within.py`: Training and evaluation for within-subject configuration.  
      - `*_between.py`: Training and evaluation for between-subject configuration.

  - `cnn/`:  
    - Contains four scripts to handle both one- and multi-feature inputs under within- and between-subject configurations:  
      - `ccnn_within_one_characteristic.py`  
      - `ccnn_within_multiple_characteristic.py`  
      - `ccnn_between_one_characteristic.py`  
      - `ccnn_between_multiple_characteristic.py`

  - `transformer/` (Vision Transformer):  
    - Also contains four scripts matching the same schema as CNN:  
      - `transformer_within.py`  
      - `transformer_within_multiple_characteristics.py`  
      - `transformer_between.py`  
      - `transformer_between_multiple_characteristics.py`
---

## Requirements

- **Python 3.11**
- **MATLAB** (for EEG preprocessing via EEGLAB)

Python dependencies:

```bash
# EEG processing
pip install mne

# Scientific libraries
pip install numpy pandas==2.2.2 scikit-learn==1.4.2 joblib==1.4.2 tensorboard==2.16.2 torchmetrics==1.3.2

# PyTorch with GPU support + torcheeg
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torcheeg==1.1.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Other tools
pip install packaging==23.2 xgboost lmdb
```

## Authors

- **María Rollano Corroto**  
  *Master's Thesis in Artificial Intelligence*  
  Universidad Politécnica de Madrid

## License

© 2025 María Rollano Corroto. Academic project submitted as part of a Master's Thesis.  
Reuse of this code is allowed for academic, educational and research purposes only.
