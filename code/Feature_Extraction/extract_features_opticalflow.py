import numpy as np
import os
from scipy.io import loadmat, savemat
from bisect import bisect_left
from tqdm import tqdm

from utils import load_interference_map, load_psg_files_with_data

'''
Extract optical flow features
'''

# Configuration
fps = 20
window_duration = 30 # Single segment duration (seconds)
nan_threshold = 0.5  # NaN value threshold, if nan ratio > nan_threshold, then the feature is set to NaN

data_collection_time = 2024 # 2024 / 2025
# Set paths
if data_collection_time == 2024:
    base_dir = rf'../dataset/2024-8'
    result_dir = os.path.join(base_dir, '2024-8-result')
else:
    base_dir = rf'../dataset/2025-8'
    result_dir = os.path.join(base_dir, '2025-8-result')
    
# Global Configuration
data_dir = os.path.join(base_dir, 'motion_signal','optical_flow_41')
summary_file = os.path.join(data_dir, 'summary_opticalflow.mat')
output_dir = os.path.join(result_dir, 'optical_flow_features')
psg_folder = os.path.join(base_dir, 'psg_sig')
interference_file_path = os.path.join(base_dir, 'interference.xlsx')

interference_map = load_interference_map(interference_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    
# Window radii iteration
# window_radii = [0, 1, 2, 3, 4, 5, 9]
window_radii = [2]

for window_radius in tqdm(window_radii, desc="Window Radii", position=0, leave=False):
    # Load global statistics
    summary_data = loadmat(summary_file)
    global_percentiles = summary_data['percentile_95']  # 95th percentile for n subjects, 22 channels
    motion_threshold = summary_data['percentile_75']  # 75th percentile for n subjects, 22 channels

    # Find PSG files
    psg_files = load_psg_files_with_data(psg_folder)

    # Subject progress bar
    for vid_idx in tqdm(range(1, len(psg_files) + 1), desc=f"Subjects (R={window_radius})", position=1, leave=False):
        
        video_info = psg_files[vid_idx-1]
        subj_id = video_info['vid_sub']
        subj_interf = interference_map.get(vid_idx, set())
        
        # Initialize paths
        subj_dir = os.path.join(data_dir, subj_id)
        output_file = os.path.join(output_dir, f'{subj_id}_features_r{window_radius}.mat')

        # Load label data
        psg_data = psg_files[vid_idx - 1]['data']
        labels = psg_data['stageValues'].item().ravel()
        labels = np.array(np.int32(labels))
        
        # Get and sort data files
        mat_files = [f for f in os.listdir(subj_dir) if f.startswith(f'subj_{vid_idx}') and f.endswith('.mat')]
        mat_files.sort(key=lambda x: int(x.split('_seg_')[1].split('.')[0]))
        
        # Verify data consistency
        if len(labels) != len(mat_files):
            min_len = min(len(labels), len(mat_files))
            if min_len < 10:
                    print("CHECK LABELS", len(labels), len(mat_files))
            labels = labels[:min_len]
            mat_files = mat_files[:min_len]
        
        total_epochs = len(mat_files)

        # ================== Preprocessing Stage: Identify high activity epochs ==================
        # Create a list of high activity epoch markers for each channel
        high_activity_epochs = [[] for _ in range(22)]  # 22 channels
        
        # Store the mean of each channel for each epoch (mACT)
        epoch_means = np.full((total_epochs, 22), np.nan)
        
        for window_idx, mat_file in enumerate(mat_files):
            # Load data
            mat_path = os.path.join(subj_dir, mat_file)
            mat_data = loadmat(mat_path)
            data = np.zeros((22, 599))
            
            for i in range(22):
                temp = mat_data.get(f'kp_{i+1}')
                if temp is not None:
                    data[i, :] = temp.flatten()
            
            for ch in range(22):
                channel_data = data[ch, :]
                nan_count = np.isnan(channel_data).sum()
                nan_ratio = nan_count / channel_data.size
                
                if nan_ratio <= nan_threshold:
                    # Use NaN-safe function to calculate mean (mACT)
                    mean_val = np.nanmean(channel_data)
                    epoch_means[window_idx, ch] = mean_val
                    
                    # Get the 95th percentile threshold for the subject's specific channel
                    ch_threshold = global_percentiles[vid_idx-1, ch]
                    
                    # If the mean of the current epoch exceeds the threshold, mark as high activity epoch
                    if mean_val > ch_threshold:
                        high_activity_epochs[ch].append(window_idx)
        
        # ================== Feature Processing Stage ==================
        all_features = []
        pSLP_values = np.full((total_epochs, 22), np.nan)  # Temporarily store unsmoothed pSLP

        for center_idx in range(total_epochs):
            # Determine sliding window range
            start_idx = max(0, center_idx - window_radius)
            end_idx = min(total_epochs-1, center_idx + window_radius)
            window_indices = range(start_idx, end_idx+1)

            # Merge window data
            merged_data = np.zeros((22, 0))  # Merge along time dimension
            has_valid_segment = False  # Whether there is at least one valid segment in the window

            for win_idx in window_indices:
                if win_idx in subj_interf:
                    continue
                mat_path = os.path.join(subj_dir, mat_files[win_idx])
                mat_data = loadmat(mat_path)
                
                # Load data, fill missing channels with NaN
                seg_data = np.full((22, 599), np.nan)  # Initialize with NaN instead of 0
                missing_channels = []  # Record missing channels
                
                for ch in range(22):
                    temp = mat_data.get(f'kp_{ch+1}')
                    if temp is not None:
                        seg_data[ch, :] = temp.flatten()
                    else:
                        # Do not break, just record missing channel (keeps NaN)
                        missing_channels.append(ch+1)
                
                # Print warning if there are missing channels but continue
                if missing_channels:
                    print(f"Warning: Missing channels {missing_channels} in file {mat_files[win_idx]}. Filled with NaN.")
                
                # Concatenate data regardless of missing channels
                merged_data = np.concatenate([merged_data, seg_data], axis=1)
                has_valid_segment = True  # Mark segment as present (even if partial channels are NaN)

            # Check: If no valid segments in the entire window (all are interference segments)
            if not has_valid_segment:
                all_features.append([[np.nan]*8 for _ in range(22)])
                print(f"Warning: No valid segments in window centered at {center_idx}. Filling NaNs.")
                continue

            current_label = labels[center_idx]
            window_features = []
            
            for ch in range(22):
                channel_data = merged_data[ch, :]
                nan_count = np.isnan(channel_data).sum()
                nan_ratio = nan_count / channel_data.size
                
                if nan_ratio > nan_threshold:
                    # NaN value exceeds threshold, fill all features with NaN
                    print(f"Warning: NaN ratio ({nan_ratio:.4f}) exceeds threshold ({nan_threshold:.4f}) for channel {ch+1} in window {win_idx}. Filling NaNs.")
                    features = [np.nan] * 8  # All 8 features are NaN
                    window_features.append(features)
                    continue
                
                # ===== Use NaN-safe function to calculate features =====
                # Basic statistical features
                mean_val = np.nanmean(channel_data)
                max_val = np.nanmax(channel_data)
                min_val = np.nanmin(channel_data)
                std_val = np.nanstd(channel_data)
                median_val = np.nanmedian(channel_data)
                
                # Interquartile range
                q1 = np.nanpercentile(channel_data, 25)
                q3 = np.nanpercentile(channel_data, 75)
                iqr_val = q3 - q1
                
                # Motion time ratio
                motion_thresh = motion_threshold[vid_idx-1, ch] 
                motion_count = np.sum(channel_data[~np.isnan(channel_data)] > motion_thresh)
                motion_ratio = motion_count / len(channel_data)
                
                # ===== Calculate pSLP features =====
                high_act_epochs = high_activity_epochs[ch]
                
                if not high_act_epochs:  # No high activity epoch
                    time_diff = total_epochs  # Use total epochs as max time difference
                else:
                    # Find the nearest high activity epoch
                    pos = bisect_left(high_act_epochs, center_idx)
                    
                    # Calculate front and back time differences
                    prev_diff = center_idx - high_act_epochs[pos-1] if pos > 0 else np.inf
                    next_diff = high_act_epochs[pos] - center_idx if pos < len(high_act_epochs) else np.inf
                    time_diff = min(prev_diff, next_diff)
                
                # Calculate pSLP (using epoch quantity)
                pSLP = np.log(1 + time_diff)
                pSLP_values[center_idx, ch] = pSLP
                
                # Create feature vector (pSLP will be added after smoothing)
                features = [
                    float(current_label),
                    mean_val,
                    max_val,
                    min_val,
                    std_val,
                    median_val,
                    iqr_val,
                    motion_ratio
                    # pSLP will be added after smoothing
                ]
                window_features.append(features)
            
            all_features.append(window_features)
        
        # ================== pSLP Smoothing ==================
        # Smooth the pSLP series for each channel
        for ch in range(22):
            pSLP_series = pSLP_values[:, ch]
            
            # Process valid values only
            valid_idx = ~np.isnan(pSLP_series)
            
            if np.any(valid_idx):
                # Create temporary array for smoothing
                temp_series = pSLP_series.copy()
                
                # Fill NaN values with the series mean to facilitate smoothing
                mean_val = np.nanmean(pSLP_series)
                temp_series[~valid_idx] = mean_val
                
                smoothed_pSLP = temp_series
                # Restore original NaN positions
                smoothed_pSLP[~valid_idx] = np.nan
                
                # Update feature matrix
                for window_idx in range(total_epochs):
                    if valid_idx[window_idx]:
                        all_features[window_idx][ch].append(smoothed_pSLP[window_idx])
                    else:
                        all_features[window_idx][ch].append(np.nan)
            else:
                # All values are NaN
                for window_idx in range(total_epochs):
                    all_features[window_idx][ch].append(np.nan)
        
        # ================== Save Results ==================
        features_array = np.array(all_features, dtype=np.float64)
        
        savemat(output_file, {
            'features': features_array,
            'feature_labels': [
                'label', 'mean', 'max', 'min', 'std', 'median',
                'iqr', 'motion_ratio', 'pSLP'
            ],
            'high_activity_epochs': high_activity_epochs
        })