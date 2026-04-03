import os
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm

from utils import load_interference_map, load_psg_files

"""
Similar with threshold_calculate_opticalflow.py
"""
data_collect_time = 2024 # 2024 / 2025
if data_collect_time == 2024:
    main_folder ='../dataset/2024-8'
    result_dir = os.path.join(main_folder, '2024-8-result')
else:
    main_folder ='../dataset/2025-8'
    result_dir = os.path.join(main_folder, '2025-8-result')

displacement_dir = os.path.join(result_dir, 'keypoint_displacement')
psg_dir = os.path.join(main_folder, 'psg_sig')
interference_file = os.path.join(main_folder, 'interference.xlsx')
# interference_file = r"E:\637\sleep\2025-8\interference.xlsx"
# displacement_base = r"E:\637\sleep\2025-8\2025-8-result\keypoint_displacement_41"

def process_displacement_data():
    roi_size = 41
    print(f"\n{'='*60}")
    print(f"Processing displacement data (ROI={roi_size})")
    print(f"{'='*60}")
    
    interference_map = load_interference_map(interference_file)
    
    if not os.path.exists(displacement_dir):
        print(f"Path does not exist: {displacement_dir}, exiting...")
        return
    
    
    files = load_psg_files(psg_dir)
    
    percentile_95 = np.zeros((len(files), 22))  # 95% percentile
    percentile_75 = np.zeros((len(files), 22))  # 75% percentile
    stats = {"Total": 0, "Excluded": 0, "Valid": 0}
    
    
    # Process each subject
    for subj_idx in tqdm(range(1, len(files)+1), desc="Subjects", leave=False):
        if subj_idx - 1 >= len(files):
            continue
            
        subj_str_id = files[subj_idx - 1]['vid_sub']
        vid_id = files[subj_idx - 1]['vid_id']
        subj_folder = os.path.join(displacement_dir, subj_str_id)
        
        if not os.path.exists(subj_folder):
            continue
            
        bad_set = interference_map.get(vid_id, set())
        
        mat_files = sorted([f for f in os.listdir(subj_folder) if f.startswith('vid') and f.endswith('.mat')],
                         key=lambda x: int(x.split('_seg_')[-1].split('.')[0]))
        
        subj_data = [] 
        
        # load every segments
        for file_idx, file_name in enumerate(mat_files):
            stats["Total"] += 1
            if file_idx in bad_set:
                stats["Excluded"] += 1
                continue
            
            file_path = os.path.join(subj_folder, file_name)
            mat_data = loadmat(file_path)
            
            data_block = []
            for i in range(21):
                key = f'kp_{i+1}_mag'
                if key in mat_data:
                    data_block.append(mat_data[key].flatten())
                else:
                    data_block.append(np.full(599, np.nan))
            
            if 'kp_22_mag' in mat_data:
                data_block.append(mat_data['kp_22_mag'].flatten())
            else:
                data_block.append(np.full(599, np.nan))
            
            data_block = np.array(data_block)
            mean_block = np.nanmean(data_block, axis=1)
            
            if np.all(np.isnan(mean_block)):
                continue
                
            stats["Valid"] += 1
            subj_data.append(mean_block)
        
        # find the threshold
        if subj_data:
            subj_data = np.array(subj_data)
            percentile_95[subj_idx-1, :] = np.nanpercentile(subj_data, 95, axis=0)
            percentile_75[subj_idx-1, :] = np.nanpercentile(subj_data, 75, axis=0)
    
    # save the result
    summary_data = {
        "percentile_95": percentile_95,
        "percentile_75": percentile_75
    }
    
    out_file = os.path.join(displacement_dir, "summary_displacement.mat")
    savemat(out_file, summary_data)
    
    print(f"\nFinished:")
    print(f"  Total segments: {stats['Total']}")
    print(f"  Excluded: {stats['Excluded']}")
    print(f"  Valid: {stats['Valid']}")
    print(f"  Results saved to: {out_file}")

if __name__ == "__main__":
    process_displacement_data()