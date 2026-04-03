import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from utils import load_interference_map, load_psg_files_with_data
"""
Calculate the threshold (95% percentile and 75% percentile) for future optical flow features extraction
"""


# ========================= Configuration =========================
data_collection_time = 2025 # 2024 / 2025
# Set paths
if data_collection_time == 2024:
    base_dir = rf'../dataset/2024-8'
else:
    base_dir = rf'../dataset/2025-8'

data_path = os.path.join(base_dir, 'motion_signal', 'optical_flow_41')  # raw data directory
output_path = data_path  # output directory
psg_dir = os.path.join(base_dir, 'psg_sig')  # psg directory
interference_file_path = os.path.join(base_dir, 'interference.xlsx')  # interference file path

# ========================= Main =========================

def main():
    all_data = []

    stats = {"Total": 0, "Excluded": 0}
    psg_file = load_psg_files_with_data(psg_dir)
    interference_map = load_interference_map(interference_file_path)
    
    pSLP_threshold = np.zeros((len(psg_file),22))  # 每位被试的 22 个关键点的 95% 分位数
    motion_threshold = np.ones((len(psg_file),22))  # 每位被试的 22 个关键点的 75% 分位数
    # 遍历每位被试
    for vid_idx in range(1, len(psg_file)+1):
        video_info = psg_file[vid_idx-1]
        mACT_subj = []
        subj_dir = video_info['vid_sub']
        subj_folder = os.path.join(data_path, subj_dir)
        bad_set = interference_map.get(vid_idx, set())
        
        print(f"\nProcessing Subject {subj_dir}...")
        
        for file_name in os.listdir(subj_folder):
            if not file_name.endswith(".mat") or not file_name.startswith(f"subj"):
                continue

            # 提取窗口编号
            parts = file_name[:-4].split('_')
            win_idx = int(parts[-1])-1 if parts[-1] else None
            stats["Total"] += 1
            # print(f"  Processing file: {file_name} (Window Index: {win_idx})")
            # 跳过干扰片段
            if win_idx is not None and win_idx in bad_set:
                stats["Excluded"] += 1
                continue

            # 读取 .mat
            file_path = os.path.join(subj_folder, file_name)
            mat_data = loadmat(file_path)
            data_block = np.zeros((22, 599))

            for i in range(22): # 22个通道（关键点）
                temp = mat_data.get(f"kp_{i+1}")
                if temp is not None:
                    if np.all(np.isnan(temp)):
                        # print(f"Warning: {file_name} contains all NaN for kp_{i+1}")
                        continue
                    data_block[i, :] = temp.flatten()
                else:
                    print(f"Warning: {file_name} missing kp_{i+1}")
            mean_block = np.nanmean(data_block, axis=1)
            if mean_block[21] == 0:
                print(f"Warning: {file_name} has zero mean for the 22th keypoint")
                continue
            mACT_subj.append(mean_block)
            all_data.append(data_block)

        pSLP_threshold[vid_idx-1, :] = np.nanpercentile(mACT_subj, 95, axis=0)
        motion_threshold[vid_idx-1, :] = np.nanpercentile(mACT_subj, 75, axis=0)
        
        if motion_threshold[vid_idx-1, 21]==1:
            print(f"Warning: Video {vid_idx} has NaN in motion_threshold for keypoint 22")
            

    if not all_data:
        print("No valid data after exclusion! Check interference table or file names.")
        return

    # ------------------ 合并 & 统计 ------------------
    all_data = np.concatenate(all_data, axis=1)  # shape: (22, N_frames)

    summary_data = {
        "percentile_95": pSLP_threshold,
        "percentile_75": motion_threshold,
    }

    out_file = os.path.join(output_path, "summary_opticalflow.mat")
    savemat(out_file, summary_data)

    print("\n======= 统计完成 =======")
    print("样本统计:", stats)
    print(f"均值 & 标准差已保存至 {out_file}")


if __name__ == "__main__":
    main()