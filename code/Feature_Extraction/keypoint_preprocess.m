% This program performs median filtering and mean filtering on the raw keypoint data,
% then performs alignment and saves as preprocessed_keypoints_1,2,3(kp_idx)
clear; clc;


% parameter setting
mainFolder = '../../dataset/2024-8';
% mainFolder = '../../dataset/2025-8';
AggPoseFolder = fullfile(mainFolder, 'motion_signal/aggpose_result');
kpDir = dir(fullfile(AggPoseFolder, 'vid*'));

delay = load(fullfile(mainFolder, 'cam_delay.mat'));
delay = delay.cam_delay;

fps = 20;
confidence_threshold = 0.5;
window_size = 11;

% Extract the number after 'vid' and convert to numericvid_idx = zeros(length(kpDir), 1);
for i = 1:length(kpDir)
    name = kpDir(i).name;
    tokens = regexp(name, 'vid_(\d+)-', 'tokens');
    if ~isempty(tokens)
        vid_idx(i) = str2double(tokens{1}{1});
    end
end

% sort by vid_idx
[~, sortedIndices] = sort(vid_idx);
kpDir = kpDir(sortedIndices);

for subj = 1:length(kpDir)
     disp("processing" + kpDir(subj).name);
    
    % read raw data from Excel
    subj_folder = fullfile(AggPoseFolder, kpDir(subj).name);
    data = read_xlsx(subj_folder);
    
   % process each key points
    for k = 1:length(data)
        % read keypoint data（N×3: x, y, confidence）
        keypoint_table = data{k};
        keypoint_array = table2array(keypoint_table);
        coordinates = keypoint_array(:, 1:2);  % x, y
        confidence = keypoint_array(:, 3);      % confidence
        
        % confidence filter
        low_conf_idx = confidence < confidence_threshold;
        coordinates(low_conf_idx, :) = NaN;
        
        % preprocess: median + mean filter
        filtered_coords = medfilt1(coordinates, window_size, 'omitnan');
        keypoint_data = movmean(filtered_coords, window_size, 'omitnan');
        
        % algn the coordinates and the camera delay
        keypoint_data = keypoint_data(delay(subj)*fps+1:end, :);

        % save the preprocessed data
        outputFilePath = fullfile(subj_folder, sprintf('preprocessed_keypoints_%d.mat', k));
        save(outputFilePath, 'keypoint_data');
    end
end



%%
function data = read_xlsx(folderPath)
% Function: Reads the xlsx file for each keypoint 
% from the specified folder (range: C:E)
    numFiles = 21;
    data = cell(1, numFiles);
    for i = 0:numFiles-1
        filePattern = fullfile(folderPath, sprintf('*_%d.xlsx', i));
        fileList = dir(filePattern);
        if ~isempty(fileList)
            fileName = fileList(1).name;
            fullFileName = fullfile(folderPath, fileName);
            try
                data{i+1} = readtable(fullFileName, 'Range', 'C:E');
            catch ME
                fprintf('Erro when loading %s: %s\n', fullFileName, ME.message);
            end
        else
            fprintf('Cannot find file in format %s \n', filePattern);
        end
    end
end