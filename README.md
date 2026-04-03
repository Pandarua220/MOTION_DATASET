# Video-actigraphy-based Sleep Staging
This is code repository for the paper "Can Video-Actigraphy Alone be used for Neonatal Sleep Staging? A Clinical Study".

To address the lack of accessible data in this field, we have released a de-identified version of our clinical dataset recorded in NICU (download link:https://drive.google.com/file/d/1_aDnGmU_h4L54n1BQ5sKoBoR9UXnI18G/view?usp=drive_link). The dataset provides **pure video-actigraphy data**, which consists of two parts: 

First, skeleton landmarks (21 keypoints) were extracted using the AggPose framework, providing a 2D skeletal representation for each frame along with their respective detection confidences.

Second, to quantify local motion dynamics, we calculated the mean optical flow magnitude within a $41 \times 41$ pixel Region of Interest (ROI) centered at each keypoint.

### Data Statistics:
| Metric | Value |
| :--- | :--- |
| Number of Subjects | 54 |
| Total Recording Hours | 177 hours |
| Sampling Rate | 20 FPS |
| Labels | NREM, REM, Wake |

### Code Structure

```text
sleep_git
├─ code
│  ├─ Feature_Extraction                      # Preprocesses raw motion signals and extract features
│  │  ├─ keypoint_preprocess.mlx              # Step 1: Preprocess keypoint coordinates
│  │  ├─ kp_displacement_calculate.py         # Step 2: Calculates inter-frame keypoint displacement
│  │  ├─ threshold_calculate_displacement.py  # Step 3: Calculates displacement threshold statistics
│  │  ├─ extract_features_displacement.py     # Step 4: Extracts keypoint displacement features
│  │  └─ ...
│  ├─ ML_Classifier              # Machine learning classifiers, including SVM, LDA, and Balanced Random Forest (BRF)
│  └─ DL_Classifier              # Deep learning classifiers, including MLP, LSTM, GRU and Transformer
└─ dataset                       # Dataset directory
│  ├─ 2024-8
│  │  ├─ psg_sig                 # PSG (Polysomnography) labels
│  │  ├─ motion_signal           # Raw motion signals
│  │  │  ├─ optical_flow_41      # Optical flow results
│  │  │  ├─ aggpose_result       # Keypoints' coordinates
│  │  ├─ cam_delay.mat           # delay between the PSG and camera
│  │  ├─ interference.xlsx       # Records video segments with interference to be excluded
│  ├─ 2025-8
│  │  └─ ...
```

### Usage Instructions

First set the working directory to the `code`.

Then to extract keypoint-displacement features, run Step 1–4 sequentially. For optical-flow features, run the optical-flow version of Step 3–4 (Python files suffixed with `opticalflow`).

Finally, ML/DL classifiers can be tested with code in `ML_Classifier` and `DL_Classifier`, respectively.
