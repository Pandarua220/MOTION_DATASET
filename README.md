Here is the complete English version of the `README.md` file. You can click the "Copy" button in the top right corner to save it as an `.md` file:

```markdown
# Sleep Management & Motion Analysis System (sleep_git)

This project focuses on the automatic monitoring and classification of infant sleep stages using motion signals captured by contactless sensors (e.g., cameras) combined with Deep Learning (DL) and Machine Learning (ML) technologies. The system supports both binary classification (Sleep vs. Wake) and three-class classification (NREM, REM, Wake) tasks.

## 📁 Project Structure

\`\`\`text
sleep_git
├─ code
│  ├─ Feature_Extraction      # Feature extraction module: processes raw motion signals
│  │  ├─ kp_displacement_calculate.py         # Calculates inter-frame keypoint displacement
│  │  ├─ extract_features_opticalflow.py      # Extracts optical flow statistical features
│  │  ├─ threshold_calculate_displacement.py  # Calculates displacement threshold statistics
│  │  └─ ... (Includes MATLAB preprocessing scripts and utilities)
│  ├─ ML_Classifier           # Traditional machine learning classifiers
│  │  ├─ classify.py          # Supports SVM, LDA, and Balanced Random Forest (BRF)
│  │  └─ utils.py             # Data loading and validation utilities
│  └─ DL_Classifier           # Deep learning classifiers
│     ├─ model.py             # Contains GRU, LSTM, Transformer, and MLP models
│     ├─ dataloader.py        # Enhanced DataLoader with sequence masking support
│     └─ classify.py          # Batch training script with 9-Fold Cross-Validation
└─ dataset                    # Dataset directory (2024-8 and 2025-8 batches)
   ├─ psg_sig                 # PSG (Polysomnography) labels and aligned data
   ├─ motion_signal           # Raw motion signals (AggPose results, optical flow results)
   └─ interference.xlsx       # Records video segments with interference to be excluded
\`\`\`

## 🚀 Core Features

1. **Multi-modal Feature Extraction**:
   * **Displacement**: Calculates inter-frame displacement for 21 keypoints and global average displacement.
   * **Optical Flow**: Extracts optical flow signals based on ROI and calculates 8 statistical features such as mean (mACT), maximum, and standard deviation.
   * **Temporal Feature (pSLP)**: Calculates the time distance between the current moment and the most recent high-intensity movement epoch.

2. **Robust Data Processing**:
   * **Interference Exclusion**: Automatically filters out segments containing environmental interference using `interference.xlsx`.
   * **Sequence Masking**: Deep learning models support processing long sequences containing invalid time steps, which are automatically masked out during calculations.
   * **Automated Alignment**: Synchronizes 20fps motion features with PSG sleep labels automatically.

3. **Multi-Model Evaluation System**:
   * **ML Analysis**: Provides SHAP interpretability analysis to identify key body parts (e.g., torso, limbs) influencing sleep stage classification.
   * **DL Training**: Implements mainstream temporal models (LSTM/GRU/Transformer) alongside an MLP baseline, and supports dimensionality reduction via PCA.

## 🛠️ Usage Instructions

### 1. Feature Preparation
First, calculate the statistical thresholds and extract features based on different batches of data:
\`\`\`bash
cd code/Feature_Extraction
# Calculate thresholds
python threshold_calculate_opticalflow.py
python threshold_calculate_displacement.py
# Extract features
python extract_features_opticalflow.py
python extract_features_displacement.py
\`\`\`

### 2. Traditional Machine Learning Training
Run the scripts under `ML_Classifier` for classification evaluation (includes SHAP feature importance analysis):
\`\`\`bash
cd code/ML_Classifier
python classify.py
\`\`\`
* Executing this script generates confusion matrix images, evaluation metrics, and `.mat` result files in the `../result/` directory.

### 3. Deep Learning Experiments
Use the batch training script to explore different model architectures:
\`\`\`bash
cd code/DL_Classifier
python classify.py --models lstm transformer --num_epochs 100 --batch_size 256
\`\`\`

## 📊 Experimental Setup
* **Cross-Validation**: Employs a subject-grouped 9-fold cross-validation strategy (7 folds train, 1 fold validation, 1 fold test).
* **Sampling & Window**: Default video framerate is 20 fps, and a single sleep analysis segment is typically 30 seconds long (corresponding to 600 frames).
* **Classification Tasks**:
    * `binary`: Sleep (NREM+REM) vs. Wake.
    * `multi`: NREM vs. REM vs. Wake.

## ⚖️ License
This project operates under the **MIT License**, permitting free use, modification, and distribution of the software, provided the original copyright notice is included.

---
```
