import numpy as np
import os
from scipy.io import loadmat, savemat
from bisect import bisect_left
from tqdm import tqdm

# import caffeine

from utils import load_interference_map, load_psg_files_with_data

"""
Used to extract features from displacement data and integrate data features.
"""

start_subj_idx = 1
end_subj_idx = -1
roi_size = 41
fps = 20
window_duration = 30
nan_threshold = 0.5

# Feature parameters
# window_radii = [0, 1, 2, 3, 4, 5, 9]
window_radii = [2]

data_collection_time = 2025 # 2024 / 2025
# Set paths
if data_collection_time == 2024:
    base_dir = rf'../dataset/2024-8'
    result_dir = os.path.join(base_dir, '2024-8-result')
else:
    base_dir = rf'../dataset/2025-8'
    result_dir = os.path.join(base_dir, '2025-8-result')



displacement_dir = os.path.join(result_dir, 'keypoint_displacement')
summary_file = os.path.join(displacement_dir, 'summary_displacement.mat')
output_base = os.path.join(result_dir, 'keypoint_displacement_features')
psg_dir = os.path.join(base_dir, 'psg_sig')
interference_file = os.path.join(base_dir, 'interference.xlsx')

def extract_features(window_radii):
    """
    Extract features for specific data type.
    Hardcoded to 'displacement', roi_size 41, and smoothing_window 0.
    
    Args:
        window_radii: List of time window radii
    """
    global end_subj_idx
    interference_map = load_interference_map(interference_file)
    
    # Find PSG files
    psg_files = load_psg_files_with_data(psg_dir)
    if end_subj_idx == -1:
        end_subj_idx = len(psg_files)

    if not os.path.exists(displacement_dir):
        assert False, f"Displacement directory does not exist: {displacement_dir}, CHECK"
        
    if not os.path.exists(summary_file):
        assert False, f"Summary file does not exist: {summary_file}, CHECK"
    
    os.makedirs(output_base, exist_ok=True)
    
    # Load global statistics
    summary_data = loadmat(summary_file)
    global_percentiles = summary_data['percentile_95']
    motion_threshold = summary_data['percentile_75']
    
    for window_radius in tqdm(window_radii, desc="Window Radii", position=0, leave=False):
        
        for vid_idx in tqdm(range(start_subj_idx, end_subj_idx+1), desc=f"Subjects (R={window_radius})", position=1, leave=False):
            
            subj_id = psg_files[vid_idx - 1]['vid_sub']
            
            subj_interf = interference_map.get(vid_idx, set())
            subj_dir = os.path.join(displacement_dir, str(subj_id))
            
            if not os.path.exists(subj_dir):
                assert False, f"Subject directory does not exist: {subj_dir}"
            
            # Output file
            output_file = os.path.join(output_base, 
                f'{subj_id}_features_r{window_radius}.mat')
            
            psg_data = psg_files[vid_idx - 1]['data']
            labels = psg_data['stageValues'].item().ravel()
            labels = np.array(np.int32(labels))
            
            # Get and sort data files
            mat_files = sorted([f for f in os.listdir(subj_dir) if f.startswith('vid') and f.endswith('.mat')],
                             key=lambda x: int(x.split('_seg_')[-1].split('.')[0]))
            
            # Verify data consistency
            if len(labels) != len(mat_files):
                if len(labels) < 10:
                    print("CHECK LABELS", len(labels))
                min_len = min(len(labels), len(mat_files))
                labels = labels[:min_len]
                mat_files = mat_files[:min_len]
            
            total_epochs = len(mat_files)
            
            # ================== Preprocessing Stage ==================
            high_activity_epochs = [[] for _ in range(22)]
            epoch_means = np.full((total_epochs, 22), np.nan)
            
            for window_idx, mat_file in enumerate(mat_files):
                if window_idx in subj_interf:
                    continue
                    
                mat_path = os.path.join(subj_dir, mat_file)
                mat_data = loadmat(mat_path)
                
                # Extract data
                data = np.zeros((22, 599))
                
                for i in range(21):
                    key = f'kp_{i+1}_mag'
                    if key in mat_data:
                        temp = mat_data[key].flatten()
                        # Ensure length is 599
                        if len(temp) >= 599:
                            data[i, :] = temp[:599]
                        else:
                            data[i, :len(temp)] = temp
                            data[i, len(temp):] = np.nan
                
                if 'kp_22_mag' in mat_data:
                    temp = mat_data['kp_22_mag'].flatten()
                    if len(temp) >= 599:
                        data[21, :] = temp[:599]
                    else:
                        data[21, :len(temp)] = temp
                        data[21, len(temp):] = np.nan
                
                # Calculate mean and identify high activity epochs
                for ch in range(22):
                    channel_data = data[ch, :]
                    nan_ratio = np.isnan(channel_data).sum() / channel_data.size
                    
                    if nan_ratio <= nan_threshold:
                        mean_val = np.nanmean(channel_data)
                        epoch_means[window_idx, ch] = mean_val
                        
                        ch_threshold = global_percentiles[vid_idx-1, ch]
                        if mean_val > ch_threshold:
                            high_activity_epochs[ch].append(window_idx)
            
            # ================== Feature Extraction Stage ==================
            all_features = []
            pSLP_values = np.full((total_epochs, 22), np.nan)
            
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
                        temp = mat_data.get(f'kp_{ch+1}_mag')
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
                    nan_ratio = np.isnan(channel_data).sum() / channel_data.size
                    
                    if nan_ratio > nan_threshold:
                        features = [np.nan] * 8
                        window_features.append(features)
                        continue
                    
                    # Calculate features
                    mean_val = np.nanmean(channel_data)
                    max_val = np.nanmax(channel_data)
                    min_val = np.nanmin(channel_data)
                    std_val = np.nanstd(channel_data)
                    median_val = np.nanmedian(channel_data)
                    
                    q1 = np.nanpercentile(channel_data, 25)
                    q3 = np.nanpercentile(channel_data, 75)
                    iqr_val = q3 - q1
                    
                    motion_thresh = motion_threshold[vid_idx-1, ch]
                    motion_count = np.sum(channel_data[~np.isnan(channel_data)] > motion_thresh)
                    motion_ratio = motion_count / len(channel_data)
                    
                    # Calculate pSLP
                    high_act_epochs = high_activity_epochs[ch]
                    if not high_act_epochs:
                        time_diff = total_epochs
                    else:
                        pos = bisect_left(high_act_epochs, center_idx)
                        prev_diff = center_idx - high_act_epochs[pos-1] if pos > 0 else np.inf
                        next_diff = high_act_epochs[pos] - center_idx if pos < len(high_act_epochs) else np.inf
                        time_diff = min(prev_diff, next_diff)
                    
                    pSLP = np.log(1 + time_diff)
                    pSLP_values[center_idx, ch] = pSLP
                    
                    features = [
                        float(current_label),
                        mean_val,
                        max_val,
                        min_val,
                        std_val,
                        median_val,
                        iqr_val,
                        motion_ratio
                    ]
                    window_features.append(features)
                
                all_features.append(window_features)
            
            # ================== pSLP Smoothing ==================
            for ch in range(22):
                pSLP_series = pSLP_values[:, ch]
                valid_idx = ~np.isnan(pSLP_series)
                
                if np.any(valid_idx):
                    temp_series = pSLP_series.copy()
                    mean_val = np.nanmean(pSLP_series)
                    temp_series[~valid_idx] = mean_val
                    
                    smoothed_pSLP = temp_series
                        
                    smoothed_pSLP[~valid_idx] = np.nan
                    
                    for window_idx in range(total_epochs):
                        if valid_idx[window_idx]:
                            all_features[window_idx][ch].append(smoothed_pSLP[window_idx])
                        else:
                            all_features[window_idx][ch].append(np.nan)
                else:
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
                'high_activity_epochs': high_activity_epochs,
                'roi_size': roi_size,
                'window_radius': window_radius
            })

def main():
    # Extract displacement data features
    extract_features(window_radii)
    
    print("\nAll feature extraction completed!")

if __name__ == "__main__":
    # caffeine.on(display=False)
    main()
    # caffeine.off()