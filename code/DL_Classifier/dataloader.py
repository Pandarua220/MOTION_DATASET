"""
Enhanced Sleep Stage DataLoader (Unified Data)
- Loads data using the unified DATASETS configuration from utils.py
- Implements Label Processing (process_label_2)
- Detailed statistics reporting
- Sequence generation with masking
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import pandas as pd
import os
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from collections import Counter

# Import utils to ensure consistency with the newly unified classification logic
try:
    from utils import (
        DATASETS, 
        load_interference_map, 
        get_psg_file_info, 
        load_psg_label,
        FEATURE_DELETE_IDX
    )
except ImportError:
    raise ImportError("Please ensure the updated utils.py is in the same directory.")

class SleepSequenceDataset(Dataset):
    """Sleep Stage Sequence Dataset for Unified Data Configuration"""
    
    def __init__(
        self,
        subjects: Optional[List[int]] = None, # If None, loads all available
        mode: str = "concatenated",
        channels: List[int] = list(range(22)),
        window_radius: int = 2,
        sequence_length: int = 30,
        stride: int = 1,
        task: str = "multi",
        roi_size: int = 41,
        normalize: str = "standard",  # "standard", "minmax", "robust", None
        pca_components: Union[int, float, None] = None,
        augment: bool = False,
        is_training: bool = True,
        scaler: Optional[object] = None,
        pca_model: Optional[PCA] = None
    ):
        self.subjects = subjects
        self.mode = mode
        self.channels = channels
        self.window_radius = window_radius
        self.sequence_length = sequence_length
        self.stride = stride
        self.task = task
        self.roi_size = roi_size
        self.normalize = normalize
        self.pca_components = pca_components
        self.augment = augment and is_training
        self.is_training = is_training
        self.scaler = scaler
        self.pca_model = pca_model

        # Internal storage
        self.sequences = []
        self.labels = []
        self.masks = []
        self.subject_ids = [] # Mapped Group IDs
        self.original_ids = [] # String descriptors (e.g., "2024-8_vid1")
        
        # Statistics storage
        self.stats_data = []

        # Load Data
        self._load_combined_data()
        
        # Preprocessing: Fit Scaler
        if self.normalize and self.scaler is None and self.is_training:
            self._fit_scaler()
        
        # Preprocessing: Apply Scaler
        if self.normalize and self.scaler is not None:
            self.sequences = self._normalize_features(self.sequences)
        
        # Preprocessing: Fit & Apply PCA
        if self.pca_components is not None and self.pca_components > 0:
            if self.pca_model is None and self.is_training:
                self._fit_pca()
            if self.pca_model is not None:
                self.sequences = self._apply_pca(self.sequences)
        
        # Convert to Tensors
        self.sequences = torch.FloatTensor(self.sequences)
        self.labels = torch.LongTensor(self.labels)
        self.masks = torch.BoolTensor(self.masks)

    def _load_combined_data(self):
        """Loads data dynamically from the unified DATASETS list."""
        
        global_group_counter = 0
        # Map unique string (dataset_name + sub_id) to int group_id 
        # This prevents merging sub_id '1' from 2024 with sub_id '1' from 2025
        sub_id_to_group_map = {} 
        
        for ds in DATASETS:
            dataset_name = ds["name"]
            
            bad_map = load_interference_map(ds["interference_file"])
            opt_path = os.path.join(ds["result_base"], ds["optical_pattern"])
            disp_path = os.path.join(ds["result_base"], ds["displacement_pattern"])
            
            psg_files = get_psg_file_info(ds["psg_base"])
            
            for psg_info in psg_files:
                vid_id = psg_info['vid_id']
                sub_id = psg_info['sub_id']
                vid_sub = psg_info['vid_sub']
                
                # Load Label
                label_all = load_psg_label(psg_info['filepath'])
                if label_all is None: continue
                
                # Unified Paths
                opt_file = os.path.join(opt_path, f"{vid_sub}_features_r{self.window_radius}.mat")
                disp_file = os.path.join(disp_path, f"{vid_sub}_features_r{self.window_radius}.mat")
                
                if not (os.path.exists(opt_file) and os.path.exists(disp_file)):
                    print(f"Warning: {opt_file} or {disp_file} does not exist. Skipping subject {vid_sub}.")
                    continue

                opt_feat = loadmat(opt_file)["features"]
                disp_feat = loadmat(disp_file)["features"]
                
                # 【CRITICAL FIX】 Create a unique key using BOTH dataset name and sub_id
                unique_sub_key = f"{dataset_name}_{sub_id}"
                
                # Determine Group ID:
                # Same unique_sub_key (e.g., "2025-8_1") gets the same group.
                # "2024-8_1" and "2025-8_1" will be treated as different subjects.
                if unique_sub_key in sub_id_to_group_map:
                    current_group = sub_id_to_group_map[unique_sub_key]
                else:
                    global_group_counter += 1
                    current_group = global_group_counter
                    sub_id_to_group_map[unique_sub_key] = current_group
                
                subj_identifier = f"{dataset_name}_{vid_sub}"
                
                self._process_subject_data(
                    subj_identifier, current_group,
                    opt_feat, disp_feat, label_all,
                    bad_map.get(vid_id, set())
                )

        # Convert lists to numpy arrays
        if len(self.sequences) > 0:
            self.sequences = np.array(self.sequences)
            self.labels = np.array(self.labels)
            self.masks = np.array(self.masks, dtype=bool)
            self.subject_ids = np.array(self.subject_ids)
        else:
            self.sequences = np.array([])
            self.labels = np.array([])
            self.masks = np.array([])
            self.subject_ids = np.array([])

    def _process_subject_data(self, subj_name, group_id, opt_feat, disp_feat, label_all, bad_set):
        """
        Extracts features, calculates stats, and builds sequences for a single subject.
        """
        # Check alignment
        min_len = min(opt_feat.shape[0], disp_feat.shape[0], label_all.shape[0])
        
        # --- STATISTICS CALCULATION ---
        # 1. Total Segments
        total_segments = min_len
        
        # 2. Interference Segments (Bad Set) within valid range
        valid_bad_indices = {x for x in bad_set if x < min_len}
        count_interference = len(valid_bad_indices)
        
        # 3. Label Counts (process_label_2 already applied in loading)
        labels_subset = label_all[:min_len]
        c = Counter(labels_subset)
        
        stats_entry = {
            "Subject": subj_name,
            "GroupID": group_id,
            "Total_Segments": total_segments,
            "Interference": count_interference,
            "T": c.get(3, 0),
            "NREM": c.get(4, 0),
            "REM": c.get(5, 0),
            "Wake": c.get(6, 0),
            "Other": sum(count for lbl, count in c.items() if lbl not in [3, 4, 5, 6])
        }
        self.stats_data.append(stats_entry)
        
        # --- FEATURE EXTRACTION & SEQUENCE BUILDING ---
        
        # Check if we should include this subject based on self.subjects filter
        if self.subjects is not None and group_id not in self.subjects:
            return

        window_features = []
        window_labels = []
        window_valid = []
        
        for win in range(min_len):
            # Validity Check
            is_bad_window = win in bad_set
            
            # Strict Label Filter: Only keep 4, 5, 6 for training/inference
            raw_label = label_all[win]
            has_valid_label = raw_label in [4, 5, 6]
            
            is_valid = (not is_bad_window) and has_valid_label
            
            if is_valid:
                # Extract Features
                opt_features = []
                disp_features = []
                
                for ch in self.channels:
                    # Remove min feature (index 2) as per utils.py
                    opt_vec = opt_feat[win, ch, 1:] 
                    opt_vec = np.delete(opt_vec, FEATURE_DELETE_IDX)
                    opt_features.extend(opt_vec)
                    
                    disp_vec = disp_feat[win, ch, 1:]
                    disp_vec = np.delete(disp_vec, FEATURE_DELETE_IDX)
                    disp_features.extend(disp_vec)
                
                concatenated = np.concatenate([opt_features, disp_features])
                concatenated = np.nan_to_num(concatenated, nan=0.0)
                
                # Transform Label
                if self.task == "binary":
                    # Sleep (4,5) vs Wake (6)
                    label = 0 if raw_label in [4, 5] else 1
                else:
                    # NREM(4)->0, REM(5)->1, Wake(6)->2
                    label_map = {4: 0, 5: 1, 6: 2}
                    label = label_map[raw_label]
                
                window_features.append(concatenated)
                window_labels.append(label)
                window_valid.append(True)
            else:
                # Invalid window (pad with zeros, use dummy label)
                feat_dim_per_ch = (opt_feat.shape[2] - 1) - 1
                total_dim = len(self.channels) * feat_dim_per_ch * 2
                
                window_features.append(np.zeros(total_dim))
                window_labels.append(-1)
                window_valid.append(False)

        # Create Sequences (Sliding Window)
        for i in range(0, min_len - self.sequence_length + 1, self.stride):
            seq_features = window_features[i : i + self.sequence_length]
            seq_labels = window_labels[i : i + self.sequence_length]
            seq_valid = window_valid[i : i + self.sequence_length]
            
            # Condition: The anchor window (usually the last one) must be valid
            if seq_valid[-1] and seq_labels[-1] != -1:
                self.sequences.append(seq_features)
                self.labels.append(seq_labels[-1])
                self.masks.append(seq_valid)
                self.subject_ids.append(group_id)
                self.original_ids.append(subj_name)

    def _print_statistics(self):
        """Prints the collected statistics as a dataframe"""
        df = pd.DataFrame(self.stats_data)
        
        # Calculate Totals
        total_row = df.sum(numeric_only=True)
        total_row["Subject"] = "TOTAL"
        total_row["GroupID"] = "-"
        
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        print("\n" + "="*50)
        print("DATASET STATISTICS (Unified)")
        print("="*50)
        # Reorder columns for readability
        cols = ["Subject", "GroupID", "Total_Segments", "Interference", "T_3", "NREM_4", "REM_5", "Wake_6", "Other"]
        print(df[cols].to_string(index=False))
        print("="*50 + "\n")
        
    def _print_valid_statistics(self):
        """
        Generates a table of valid sequences actually used for training/testing.
        Matches the format: Infant ID | Segment Num | NREM(n/%) | REM(n/%) | Wake(n/%)
        """
        if len(self.labels) == 0:
            print("No valid sequences found.")
            return

        unique_groups = np.unique(self.subject_ids)
        rows = []
        
        grand_total = 0
        total_nrem = 0
        total_rem = 0
        total_wake = 0
        
        for grp in unique_groups:
            idx = np.where(self.subject_ids == grp)[0]
            curr_labels = self.labels[idx].numpy()
            
            n_samples = len(idx)
            grand_total += n_samples
            
            n_nrem = np.sum(curr_labels == 0)
            n_rem = np.sum(curr_labels == 1)
            n_wake = np.sum(curr_labels == 2)

            total_nrem += n_nrem
            total_rem += n_rem
            total_wake += n_wake
            
            row = {
                "Infant ID": int(grp),
                "Segment Num.": n_samples,
                "NREM (n/%)": f"{n_nrem} ({n_nrem/n_samples*100:.1f}%)",
                "REM (n/%)": f"{n_rem} ({n_rem/n_samples*100:.1f}%)",
                "Wake (n/%)": f"{n_wake} ({n_wake/n_samples*100:.1f}%)"
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        overall = {
            "Infant ID": "Overall",
            "Segment Num.": grand_total,
            "NREM (n/%)": f"{total_nrem} ({total_nrem/grand_total*100:.1f}%)",
            "REM (n/%)": f"{total_rem} ({total_rem/grand_total*100:.1f}%)",
            "Wake (n/%)": f"{total_wake} ({total_wake/grand_total*100:.1f}%)"
        }
        df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
        
        print("\n" + "="*80)
        print(f"Sleep Stage Distribution Across Subjects (Task: {self.task.upper()})")
        print("Note: Valid Sequences only (valid anchor + no interference)")
        if self.task == 'binary':
            print("Note: Binary Task -> 'NREM' col is Sleep (N+R), 'REM' is 0.")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
    def _print_load_summary(self):
        print(f"Loaded {len(self.sequences)} sequences.")
        print(f"Unique Subjects (Groups): {len(set(self.subject_ids))}")
        print(f"Shape: {self.sequences.shape}")
        if len(self.labels) > 0:
            print(f"Label Dist: {np.bincount(self.labels.numpy())}")

    def _fit_scaler(self):
        """Fits scaler only on valid windows in training data"""
        if self.normalize == "standard":
            self.scaler = StandardScaler()
        elif self.normalize == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalize == "robust":
            self.scaler = RobustScaler()
        else:
            return
        
        valid_data = []
        for i in range(len(self.sequences)):
            mask = self.masks[i]
            seq = self.sequences[i]
            valid_data.append(seq[mask]) # Only take valid timesteps
            
        if valid_data:
            flat_data = np.concatenate(valid_data, axis=0)
            self.scaler.fit(flat_data)
            print(f"Scaler ({self.normalize}) fitted on {flat_data.shape[0]} valid timestamps.")

    def _normalize_features(self, sequences):
        if self.scaler is None: return sequences
        orig_shape = sequences.shape
        reshaped = sequences.reshape(-1, orig_shape[-1])
        norm = self.scaler.transform(reshaped)
        return norm.reshape(orig_shape)

    def _fit_pca(self):
        self.pca_model = PCA(n_components=self.pca_components, random_state=42)
        valid_data = []
        for i in range(len(self.sequences)):
            mask = self.masks[i]
            seq = self.sequences[i]
            valid_data.append(seq[mask])
            
        if valid_data:
            flat_data = np.concatenate(valid_data, axis=0)
            self.pca_model.fit(flat_data)
            print(f"PCA fitted. Components: {self.pca_model.n_components_}, Var: {self.pca_model.explained_variance_ratio_.sum():.4f}")

    def _apply_pca(self, sequences):
        if self.pca_model is None: return sequences
        orig_shape = sequences.shape
        reshaped = sequences.reshape(-1, orig_shape[-1])
        transformed = self.pca_model.transform(reshaped)
        return transformed.reshape(orig_shape[0], orig_shape[1], -1)
    
    def get_scaler(self):
        return self.scaler
        
    def get_pca_model(self):
        return self.pca_model

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        
        if self.augment:
            sequence = self._augment_sequence(sequence, mask)
        
        return sequence, label, mask

    def _augment_sequence(self, sequence, mask):
        augmented = sequence.clone() if torch.is_tensor(sequence) else sequence.copy()
        noise_level = 0.01
        noise = torch.randn_like(augmented) * noise_level
        
        expanded_mask = mask.unsqueeze(-1).expand_as(augmented)
        augmented = torch.where(expanded_mask, augmented + noise, augmented)
        
        return augmented

def main():
    print("Scanning dataset for groups...")
    # Initialize with minimal overhead just to scan IDs
    temp_ds = SleepSequenceDataset(
        subjects=None, # Load all
        channels=list(range(21)),
        window_radius=2,
        sequence_length=1, 
        task="multi",
        normalize=None, 
        pca_components=None,
        augment=False,
        is_training=False
    )
    
    unique_groups = np.unique(temp_ds.subject_ids)
    print(f"Found {len(unique_groups)} unique subject groups: {unique_groups}")
    
    temp_ds._print_valid_statistics()

if __name__ == "__main__":
    main()