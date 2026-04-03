import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== Data Source Configuration =====================
# Unified data source configuration list
DATASETS = [
    {
        "name": "2024-8",
        "interference_file": r"../dataset/2024-8/interference.xlsx",
        "psg_base": r"../dataset/2024-8/psg_sig",
        "result_base": r"../dataset/2024-8/2024-8-result",
        "optical_pattern": "optical_flow_features",
        "displacement_pattern": "keypoint_displacement_features",
    },
    {
        "name": "2025-8",
        "interference_file": r"../dataset/2025-8/interference.xlsx",
        "psg_base": r"../dataset/2025-8/psg_sig",
        "result_base": r"../dataset/2025-8/2025-8-result",
        "optical_pattern": "optical_flow_features",
        "displacement_pattern": "keypoint_displacement_features",
    }
]

FEATURE_DELETE_IDX = 2  # Index of the 'min' feature to be deleted

# ===================== Basic IO Functions =====================

def load_interference_map(xlsx_path: str) -> dict[int, set[int]]:
    """Read the interference mapping file (Excel)."""
    if not os.path.exists(xlsx_path):
        print(f"Warning: Interference file not found: {xlsx_path}")
        return {}
    
    df = pd.read_excel(xlsx_path, header=None)
    mapping: dict[int, set[int]] = {}
    for _, row in df.iterrows():
        sid_raw = str(row.iloc[0]).strip()
        if sid_raw.lower().startswith("subj"):
            sid = int(sid_raw.lower().replace("subj", ""))
        elif sid_raw.lower().startswith("vid"):
            sid = int(sid_raw.lower().replace("vid_", ""))
        else:
            try:
                sid = int(float(sid_raw))
            except ValueError:
                continue
        mapping[sid] = {int(v) for v in row.iloc[1:].dropna().tolist()}
    return mapping

def process_label_2(labels: np.ndarray) -> np.ndarray:
    """Process label array, handling and replacing segments with value 2."""
    processed = labels.copy()
    n = len(processed)
    i = 0
    while i < n:
        if processed[i] == 2:
            start = i
            while i < n and processed[i] == 2:
                i += 1
            end = i - 1
            length = end - start + 1
            
            prev_label = processed[start - 1] if start > 0 else None
            next_label = processed[end + 1] if end < n - 1 else None
            
            if prev_label is not None and next_label is not None:
                if prev_label == next_label and prev_label in [4, 5]:
                    replacement = prev_label
                else:
                    replacement = 3
            else:
                replacement = 3
                
            if length >= 7:
                processed[start:end+1] = 1 # Wake
            else:
                processed[start:end+1] = replacement
        else:
            i += 1
    return processed

def get_psg_file_info(folder_path):
    """Retrieve list of PSG files in the unified naming format."""
    pattern = os.path.join(folder_path, "psg*.mat")
    mat_files = glob.glob(pattern)
    
    if not mat_files:
        print(f"Warning: No .mat files starting with 'psg' found in {folder_path}")
        return []
    
    regex = re.compile(r'(vid_(\d+)-sub_(\w+))')
    file_info_list = []
    
    for filepath in mat_files:
        filename = os.path.basename(filepath)
        match = regex.search(filename)
        if match:
            file_info_list.append({
                'filepath': filepath,
                'filename': filename,
                'vid_sub': match.group(1),
                'vid_id': int(match.group(2)),
                'sub_id': match.group(3)
            })
    
    file_info_list.sort(key=lambda x: x['vid_id'])
    return file_info_list

def load_psg_label(filepath) -> np.ndarray:
    """Load and process PSG labels from a .mat file."""
    try:
        label = loadmat(filepath)["label_algn"]
        label = label['stageValues']
        return process_label_2(np.array(label.item())).ravel()
    except Exception as e:
        print(f"Failed to load PSG file: {filepath}, Error: {e}")
        return None

def process_windows(opt_feat, disp_feat, label_all, bad_set, chs, classification_type,
                    xs, ys_binary, ys_three, gps, group_id):
    """Process windowed data for a single subject."""
    total_samples = 0
    for win in range(opt_feat.shape[0]):
        if win in bad_set: continue
        if win >= len(label_all): continue
            
        label_val = label_all[win]
        if label_val not in [4, 5, 6]: continue # Keep only NREM(4), REM(5), Wake(6)
        
        # Label Mapping
        if classification_type == "binary":
            label_bin = 0 if label_val in [4, 5] else 1 # Sleep (0) vs Wake (1)
            label_three = None
        else:
            label_map = {4: 0, 5: 1, 6: 2} # NREM(0), REM(1), Wake(2)
            label_three = label_map.get(label_val)
            label_bin = None
        
        # Feature Extraction
        opt_features = []
        disp_features = []
        
        for c in chs:
            # Delete Min feature (index 2) after slicing out the anchor (index 0)
            opt_vec = np.delete(opt_feat[win, c, 1:], FEATURE_DELETE_IDX)
            disp_vec = np.delete(disp_feat[win, c, 1:], FEATURE_DELETE_IDX)
            opt_features.extend(opt_vec)
            disp_features.extend(disp_vec)
        
        concatenated = np.concatenate([opt_features, disp_features])
        concatenated = np.nan_to_num(concatenated, nan=0.0)
        
        xs.append(concatenated)
        gps.append(group_id)
        
        if classification_type == "binary":
            ys_binary.append(label_bin)
        else:
            ys_three.append(label_three)
        
        total_samples += 1
    return total_samples

# ===================== Core Loading Logic =====================

def load_concatenated_features_combined(window_radius, chs, classification_type="three_class"):
    """Load and merge data from all years, grouping by sub_id consistently."""
    xs, ys_binary, ys_three, gps = [], [], [], []
    total = excluded = 0
    global_subj_counter = 0
    
    for ds in DATASETS:
        sub_id_to_group_map = {}
        dataset_name = ds["name"]
        print(f"\n--- Loading {dataset_name} data ---")
        
        bad_map = load_interference_map(ds["interference_file"])
        opt_path = os.path.join(ds["result_base"], ds["optical_pattern"])
        disp_path = os.path.join(ds["result_base"], ds["displacement_pattern"])
        
        psg_files = get_psg_file_info(ds["psg_base"])
        
        for psg_info in psg_files:
            vid_id = psg_info['vid_id']
            sub_id = psg_info['sub_id']
            vid_sub = psg_info['vid_sub']
                
            label_all = load_psg_label(psg_info['filepath'])
            if label_all is None: 
                print(f"Warning: {vid_sub} label file is None. CHECK!!!\n")
                continue
            
            # Unified filename formatting
            opt_file = os.path.join(opt_path, f"{vid_sub}_features_r{window_radius}.mat")
            disp_file = os.path.join(disp_path, f"{vid_sub}_features_r{window_radius}.mat")
            
            if not (os.path.exists(opt_file) and os.path.exists(disp_file)):
                print(f"Warning: {vid_sub} {opt_file} or {disp_file} does not exist. CHECK!!!\n")
                continue

            opt_feat = loadmat(opt_file)["features"]
            disp_feat = loadmat(disp_file)["features"]
            
            # Grouping Logic: Assign unique Group ID based on Subject ID across datasets
            if sub_id in sub_id_to_group_map:
                current_group = sub_id_to_group_map[sub_id]
            else:
                global_subj_counter += 1
                current_group = global_subj_counter
                sub_id_to_group_map[sub_id] = current_group
                
            added = process_windows(opt_feat, disp_feat, label_all, bad_map.get(vid_id, set()), 
                                    chs, classification_type, xs, ys_binary, ys_three, gps, current_group)
            
            total += opt_feat.shape[0]
            excluded += (opt_feat.shape[0] - added)
            print(f"  {dataset_name} {vid_sub}: {added} samples (Group {current_group})")

    stats = {
        "Total": total, "Excluded": excluded, 
        "Valid": len(ys_binary) if classification_type == "binary" else len(ys_three),
        "NumGroups": global_subj_counter
    }
    
    print(f"\n--- Summary: {stats['Valid']} valid samples from {stats['NumGroups']} unique subjects ---")
    
    labels = np.array(ys_binary if classification_type == "binary" else ys_three)
    return np.array(xs), labels, np.array(gps), stats


# ===================== Statistics & Visualization =====================

def _format_percentages(n_nrem, n_rem, n_wake, total):
    """Helper: Calculate and format percentage strings."""
    if total == 0:
        return "0 (0.0%)", "0 (0.0%)", "0 (0.0%)"
    
    p_nrem = (n_nrem / total) * 100
    p_rem = (n_rem / total) * 100
    p_wake = (n_wake / total) * 100
    
    return f"{n_nrem} ({p_nrem:.1f}%)", f"{n_rem} ({p_rem:.1f}%)", f"{n_wake} ({p_wake:.1f}%)"

def print_and_plot_valid_statistics(labels: np.ndarray, subject_ids: np.ndarray, task: str = 'three_class'):
    """Print statistical table and plot violin charts for sleep stage distribution."""
    if len(labels) == 0:
        print("No valid sequences found.")
        return

    unique_groups = np.unique(subject_ids)
    rows = []
    plot_data = [] 
    
    grand_total = 0
    total_nrem = total_rem = total_wake = 0
    
    for grp in unique_groups:
        idx = np.where(subject_ids == grp)[0]
        curr_labels = labels[idx]
        
        n_samples = len(idx)
        if n_samples == 0: continue
            
        grand_total += n_samples
        
        if task == 'binary':
            n_nrem = np.sum(curr_labels == 0)
            n_rem = 0
            n_wake = np.sum(curr_labels == 1)
        else:
            n_nrem = np.sum(curr_labels == 0)
            n_rem = np.sum(curr_labels == 1)
            n_wake = np.sum(curr_labels == 2)

        total_nrem += n_nrem
        total_rem += n_rem
        total_wake += n_wake
        
        nrem_str, rem_str, wake_str = _format_percentages(n_nrem, n_rem, n_wake, n_samples)
        
        rows.append({
            "Infant ID": int(grp),
            "Segment Num.": n_samples,
            "NREM (n/%)": nrem_str,
            "REM (n/%)": rem_str,
            "Wake (n/%)": wake_str
        })
        
        if task == 'binary':
            plot_data.append({"Subject ID": int(grp), "Stage": "Sleep", "Percentage (%)": (n_nrem / n_samples) * 100})
            plot_data.append({"Subject ID": int(grp), "Stage": "Wake", "Percentage (%)": (n_wake / n_samples) * 100})
        else:
            plot_data.append({"Subject ID": int(grp), "Stage": "NREM", "Percentage (%)": (n_nrem / n_samples) * 100})
            plot_data.append({"Subject ID": int(grp), "Stage": "REM", "Percentage (%)": (n_rem / n_samples) * 100})
            plot_data.append({"Subject ID": int(grp), "Stage": "Wake", "Percentage (%)": (n_wake / n_samples) * 100})
        
    df = pd.DataFrame(rows)
    overall_nrem_str, overall_rem_str, overall_wake_str = _format_percentages(
        total_nrem, total_rem, total_wake, grand_total
    )
    overall = {
        "Infant ID": "Overall",
        "Segment Num.": grand_total,
        "NREM (n/%)": overall_nrem_str,
        "REM (n/%)": overall_rem_str,
        "Wake (n/%)": overall_wake_str
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    
    print("\n" + "="*80)
    print(f"Sleep Stage Distribution Across Subjects (Task: {task.upper()})")
    print("Note: Valid Sequences only (valid anchor + no interference)")
    if task == 'binary':
        print("Note: Binary Task -> 'NREM' col represents Sleep (NREM+REM).")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_df, x="Stage", y="Percentage (%)", palette="Set2", inner="quartile")
    
    title_suffix = "Sleep vs Wake" if task == 'binary' else "NREM vs REM vs Wake"
    plt.title(f"Distribution of Sleep Stage Segment Percentages ({title_suffix})", fontsize=14, pad=15)
    plt.ylabel("Percentage of Total Segments (%)", fontsize=12)
    plt.xlabel("Sleep Stage", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Configure Data Loading Parameters
    roi_size = "41" 
    feature_type = "v4"
    window_radius = 2 
    smoothing_window = 0 
    chs = range(22)
    task_type = "three_class"  # Choice: "three_class" or "binary"

    # 2. Load and Merge Feature Data
    print(f"Starting data load for task: {task_type} ...")
    X, y, groups, stats = load_concatenated_features_combined(
        window_radius=window_radius,
        chs=chs,
        classification_type=task_type
    )

    print("\n--- Data Check ---")
    print(f"Feature matrix (X) shape: {X.shape}")
    print(f"Labels array (y) shape: {y.shape}")
    print(f"Groups array (groups) shape: {groups.shape}")
    print(f"Statistics: {stats}")

    # 3. Statistics & Visualization
    if len(y) > 0:
        print("\nGenerating distribution table and violin plot...")
        print_and_plot_valid_statistics(labels=y, subject_ids=groups, task=task_type)
    else:
        print("\nWarning: No valid data loaded. Plotting aborted.")

if __name__ == "__main__":
    main()