import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
from glob import glob
import re

def load_interference_map(xlsx_path: str) -> dict[int, set[int]]:
    """Read xlsx, return {subj_id: {win_idx, ...}}"""
    df = pd.read_excel(xlsx_path, header=None)
    mapping: dict[int, set[int]] = {}
    for _, row in df.iterrows():
        subj_col = str(row.iloc[0]).strip()
        if subj_col.lower().startswith("vid"):
            subj_col = subj_col.lower().replace("vid_", "")
        elif subj_col.lower().startswith("subj"):
            subj_col =subj_col.lower().replace("subj", "")
        else:
            raise ValueError(f"Unknown subject ID format: {subj_col}")
        subj_id = int(subj_col)
        mapping[subj_id] = {
            int(val) for val in row.iloc[1:].dropna().tolist()
        }
    return mapping

def load_psg_files(folder_path):
    """
    Read mat files starting with 'psg' in the specified folder and sort them by video ID
    
    Args:
        folder_path: Folder path
    
    Returns:
        list of dict: A list containing file information, sorted by vid number
            - 'filepath': Full file path
            - 'filename': File name
            - 'vid_sub': Extracted vid_X-sub_XX string
            - 'vid_id': Video ID (int)
            - 'sub_id': Subject ID (str)
            - 'data': mat file content
    """
    # Find all mat files starting with 'psg'
    pattern = os.path.join(folder_path, "psg*.mat")
    mat_files = glob(pattern)
    
    if not mat_files:
        print(f"Warning: No mat files starting with 'psg' found in {folder_path}")
        return []
    
    # Regular expression to extract vid_X-sub_XX
    # Match format: vid_number-sub_number or letter
    regex = re.compile(r'(vid_(\d+)-sub_(\w+))')
    
    file_info_list = []
    
    for filepath in mat_files:
        filename = os.path.basename(filepath)
        match = regex.search(filename)
        
        if match:
            vid_sub = match.group(1)      # vid_2-sub_02
            vid_id = int(match.group(2))  # 2
            sub_id = match.group(3)       # 02
            
            file_info_list.append({
                'filepath': filepath,
                'filename': filename,
                'vid_sub': vid_sub,
                'vid_id': vid_id,
                'sub_id': sub_id,
                'data': None  # Lazy loading, or direct load
            })
        else:
            print(f"Warning: Cannot extract 'vid_sub' info from filename '{filename}'")
    
    # Sort by vid_id in ascending order
    file_info_list.sort(key=lambda x: x['vid_id'])
    
    return file_info_list


def load_psg_files_with_data(folder_path):
    """
    Version that reads and loads the mat file data
    """
    file_info_list = load_psg_files(folder_path)
    
    for info in file_info_list:
        try:
            info['data'] = loadmat(info['filepath'])
            info['data'] = info['data']['label_algn']
        except Exception as e:
            print(f"Failed to load file {info['filename']}: {e}")
            info['data'] = None
    
    return file_info_list