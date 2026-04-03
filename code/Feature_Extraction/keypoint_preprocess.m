% Script 2: Process individual .mat files
clear; clc;

% --- Parameter settings ---
mainFolder = '../../dataset/2024-8';
% mainFolder = '../../dataset/2024-8';
AggPoseFolder = fullfile(mainFolder, 'motion_signal/aggpose_result');
kpDir = dir(fullfile(AggPoseFolder, 'vid*'));

% Load camera delay
delay_data = load(fullfile(mainFolder, 'cam_delay.mat'));
delay = delay_data.cam_delay;

% Preprocessing parameters
fps = 20;
confidence_threshold = 0.5;
window_size = 11;
numKeypoints = 21;

% --- Folder sorting ---
vid_idx = zeros(length(kpDir), 1);
for i = 1:length(kpDir)
    tokens = regexp(kpDir(i).name, 'vid_(\d+)-', 'tokens');
    if ~isempty(tokens)
        vid_idx(i) = str2double(tokens{1}{1});
    end
end
[~, sortedIndices] = sort(vid_idx);
kpDir = kpDir(sortedIndices);

% --- Preprocessing loop ---
for subj = 1:length(kpDir)
    subj_folder = fullfile(AggPoseFolder, kpDir(subj).name);
    fprintf('Preprocessing: %s\n', kpDir(subj).name);
    
    for i = 0:numKeypoints-1
        matPath = fullfile(subj_folder, sprintf('kp_%d.mat', i));
        
        if exist(matPath, 'file')
            % Load data, variable name is raw_table
            load(matPath); 
            
            % 1. Data conversion and extraction
            keypoint_array = table2array(kp_data);
            coordinates = keypoint_array(:, 1:2);  % x, y
            confidence = keypoint_array(:, 3);      % confidence
            
            % 2. Confidence filtering
            low_conf_idx = confidence < confidence_threshold;
            coordinates(low_conf_idx, :) = NaN;
            
            % 3. Filtering
            filtered_coords = medfilt1(coordinates, window_size, 'omitnan');
            keypoint_data = movmean(filtered_coords, window_size, 'omitnan');
            
            % 4. Time alignment
            start_idx = round(delay(subj) * fps) + 1;
            if start_idx <= size(keypoint_data, 1)
                keypoint_data = keypoint_data(start_idx:end, :);
            else
                keypoint_data = []; 
            end

            % 5. Save preprocessed result (keeping original naming convention)
            outputFilePath = fullfile(subj_folder, sprintf('preprocessed_keypoints_%d.mat', i));
            save(outputFilePath, 'keypoint_data');
        end
    end
end
fprintf('All preprocessing completed.\n');
