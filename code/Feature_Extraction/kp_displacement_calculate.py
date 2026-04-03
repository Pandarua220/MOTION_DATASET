import os
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm
from utils import load_psg_files_with_data
"""
计算关键点帧间位移 (displacement)
"""

data_collect_time = 2024 # 2024 / 2025
if data_collect_time == 2024:
    main_folder ='../dataset/2024-8'
    output_base = os.path.join(main_folder, '2024-8-result', 'keypoint_displacement')
else:
    main_folder ='../dataset/2025-8'
    output_base = os.path.join(main_folder, '2025-8-result', 'keypoint_displacement')

    
base_keypoint_path = os.path.join(main_folder, 'motion_signal', 'aggpose_result')
psg_dir = os.path.join(main_folder, 'psg_sig')


start_subj = 1
end_subj = -1
SEG_LEN = 600 # 20fps * 30s

def calculate_keypoint_displacement(x, y, interp=True):
    if interp:
        # interpolate NaN values
        valid = ~np.isnan(x)
        if np.any(~valid) and np.any(valid):
            indices = np.arange(len(x))
            x = np.interp(indices, indices[valid], x[valid])
            y = np.interp(indices, indices[valid], y[valid])
    
    dx = np.diff(x)
    dy = np.diff(y)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    return dx, dy, magnitude


def main():
    global start_subj, end_subj, SEG_LEN, data_collect_time 
    
    # create output directory
    if not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
        
    files = load_psg_files_with_data(psg_dir)
    
    if end_subj == -1:
        end_subj = len(files)
        
    for subj_idx in tqdm(range(start_subj, end_subj+1), desc="Video", leave=False):

        subj_id = files[subj_idx - 1]['vid_sub']
            
        psg_label = files[subj_idx - 1]['data']['stageValues'].item().ravel()
        psg_label = np.array(np.int32(psg_label))
        # create output directory
        subj_output_disp = os.path.join(output_base, subj_id)
        
        if not os.path.exists(subj_output_disp):
            os.makedirs(subj_output_disp, exist_ok=True)

        # path setup
        keypoint_dir = os.path.join(base_keypoint_path, subj_id)
        
        # preload all keypoint data
        kp_data_list = []
        for kp in range(21):
            kp_file = os.path.join(keypoint_dir, f'preprocessed_keypoints_{kp+1}.mat')
            if not os.path.exists(kp_file):
                kp_data_list.append((None, None, None))
                assert False, f"Keypoint data for keypoint {kp+1} not found in subject {subj_id}."
            
            kp_data = loadmat(kp_file)
            x = kp_data['keypoint_data'][:, 0].flatten()
            y = kp_data['keypoint_data'][:, 1].flatten()
            kp_data_list.append((x, y))
        
        seg_num = x.shape[0] // SEG_LEN
        if psg_label.shape[0] - seg_num > 1 or psg_label.shape[0] - seg_num < 0:
            print(f" Warning: {subj_id} Segment number {seg_num} not equal to psg label number {psg_label.shape[0]}.")
        # process each segment
        for seg_idx in tqdm( range(1, seg_num+1), total=seg_num, desc="Segment", leave=False):

            # Calculate segment range
            seg_start = (seg_idx - 1) * SEG_LEN
            seg_end = seg_idx * SEG_LEN
            
            # initialize storage array for displacement data
            displacement_data = {}
            
            # process each keypoint
            for kp in range(21):
                x, y = kp_data_list[kp]
                
                # extract current segment data from keypoint data
                start = max(0, seg_start)
                end = min(len(x), seg_end)
                
                x_seg = x[start:end]
                y_seg = y[start:end]                    
                
                # calculate keypoint displacement for current segment data
                dx, dy, kp_magnitude = calculate_keypoint_displacement(
                    x_seg, y_seg
                )
                
                displacement_data[f'kp_{kp+1}_dx'] = dx
                displacement_data[f'kp_{kp+1}_dy'] = dy
                displacement_data[f'kp_{kp+1}_mag'] = kp_magnitude

            
            # process global channel
            all_kp_mags = []
            for kp in range(21):
                if f'kp_{kp+1}_mag' in displacement_data:
                    all_kp_mags.append(displacement_data[f'kp_{kp+1}_mag'])
            
            if all_kp_mags:
                # calculate global displacement
                if not np.isnan(np.array(all_kp_mags)).all():
                    global_displacement = np.nanmean(np.array(all_kp_mags), axis=0)
                    displacement_data['kp_22_mag'] = global_displacement
                else:
                    print(f"Warning: NaN values found in kp_{kp+1}_mag for segment {seg_idx} in subject {subj_id}.")
                    displacement_data['kp_22_mag'] = 0  
                    


            # save displacement data
            disp_output_file = os.path.join(
                subj_output_disp, 
                f'{subj_id}_displacement_seg_{seg_idx}.mat'
            )
            savemat(disp_output_file, displacement_data)
                
    
    print("\nFinish！")
    print("Displacement data saved in: keypoint_displacement folder")

if __name__ == "__main__":
    # caffeine.on(display=False)
    main()
    # caffeine.off()