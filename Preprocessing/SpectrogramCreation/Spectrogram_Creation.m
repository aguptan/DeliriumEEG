tic; clc; close all; clear all;
set(0,'DefaultFigureWindowStyle','normal'); 

% Add function path
addpath('D:\AmalScripts2\Functions');
% addpath('E:\Delerium-EEG\AmalScripts\Functions');

% Define the directory containing the .mat files
data_dir = 'E:\Delerium-EEG\MatData'; % Change this to your directory
mat_files = dir(fullfile(data_dir, '*.mat')); % Get all .mat files

% Where all the spectrograms are saved
%parent_dir = 'E:\Delerium-EEG\Spectrograms\';

time_windows = sort([2, 15, 30, 120], 'descend'); % in minutes

% Store processed filenames for summary printing
processed_files = {};

logLines = {};

spectrogram_root = 'E:\Delerium-EEG\Spectrograms';
summary_log_path = fullfile(spectrogram_root, 'processing_summary.txt');


for numMat = 1:length(mat_files)
    
    
    % Get full file path
    file_path = fullfile(data_dir, mat_files(numMat).name);
    
    % Extract file name without extension for dynamic naming
    [~, base_filename, ~] = fileparts(mat_files(numMat).name);
    logLines{end+1} = sprintf('%s', base_filename);  % Person name header

    % Load EEG data
    load(file_path, 'eegStruct');
    
    % Get the number of datasets using arrayfun
    num_datasets = numel(arrayfun(@(x) size(x.Data), eegStruct, 'UniformOutput', false));

    for dataset_idx = 1:num_datasets
        % Create a unique folder name for each dataset
        folder_name = sprintf('%s_%d', base_filename, dataset_idx);
        logPrefix = sprintf('  Dataset %d:', dataset_idx);
        % Print confirmation message before processing
        fprintf('%s loaded successfully.\n', folder_name);
        
        % Variables for preprocessing
        fs = 200; % Sampling frequency (Hz)
        dat = eegStruct(dataset_idx).Data; % Extract dataset from structure
        
        % Define EEG channels
        eeg_channels = [1:19]; 
        
        % Extract EEG data 
        eeg_data = dat(:, eeg_channels)'; 
        [num_electrodes, num_timepoints] = size(eeg_data); 
        
        % Extract labels
        labels = eegStruct(dataset_idx).Labels; 
        selected_labels = labels(eeg_channels); 
        
        % Clear raw struct to free memory
        clear dat 
        
        %% (2) Remove NaNs via Interpolation
        try
            [eeg_data, nanFlag] = removeNaN(eeg_data);
            if nanFlag
                logLines{end+1} = sprintf('%s skipped - too many NaNs', logPrefix);
                continue;
            end
        catch ME
            logLines{end+1} = sprintf('%s error in removeNaN2 - %s', logPrefix, ME.message);
            continue;
        end
  
        %% (3) Apply Notch Filtering
        notchFreqs = [20, 60, 80]; % Notch frequencies to remove
        Q = 75; % Quality factor (higher Q = narrower notch)
        electrode_idx = 10; % Electrode for PSD visualization
        plt = 0; % Set to 1 to plot PSD, 0 to skip plotting        
        try
            [nf_eeg_data, ~, ~, ~] = applyNotchFilter(eeg_data, fs, notchFreqs, Q, electrode_idx, plt);
        catch ME
            logLines{end+1} = sprintf('%s error in applyNotchFilter - %s', logPrefix, ME.message);
            continue;
        end

        
        %% (4) Apply High-Pass Butterworth Filtering
        try
            hp_cutoff = 0.5;       % High-pass cutoff frequency in Hz
            filter_order = 4;      % Filter order (typically 2–4 is reasonable)
            [filtered_eeg_data, ~, ~, ~] = applyButter(nf_eeg_data, fs, hp_cutoff, filter_order, electrode_idx, plt);
        
        catch ME
            logLines{end+1} = sprintf('%s error in applyButter - %s', logPrefix, ME.message);
        continue;
        end

        
        %% (5) Re-referencing to the Common Mean
        try
            rreeg_data = reReference(filtered_eeg_data);
        catch ME
            logLines{end+1} = sprintf('%s error in reReference - %s', logPrefix, ME.message);
        continue;
        end

        
        %% (6) Compute and Plot PSD for One Electrode 
        try
            [~, ~, ~] = CaP_PSD(eeg_data, rreeg_data, fs, electrode_idx, plt);
        catch ME
            logLines{end+1} = sprintf('%s error in CaP_PSD - %s', logPrefix, ME.message);
            continue;
        end

        
        %% (7) Loop over Time Window Durations
        

        for w = 1:length(time_windows)
            window_duration = time_windows(w);

            % Create output directory for this window size
            current_parent_dir = fullfile(spectrogram_root, sprintf('%dmin', window_duration));
            if ~exist(current_parent_dir, 'dir'); mkdir(current_parent_dir); end
            fprintf('\n=== Processing for %d-minute windows ===\n', window_duration);
            
            log_folder = fullfile(current_parent_dir, folder_name);
            if ~exist(log_folder, 'dir'); mkdir(log_folder); end
            log_file = fullfile(log_folder, sprintf('%s_%dmin_log.txt', folder_name, window_duration));
            diary(log_file);

            % Apply time windowing
            try
                [tw_eeg_data, ~, skipFlag] = timeWindowEEG(rreeg_data, fs, window_duration);
                if skipFlag
                    fprintf('Skipping dataset %s: too short for time window of %d min.\n', folder_name, window_duration);
                    logLines{end+1} = sprintf('%s %dmin - skipped (too short)', logPrefix, window_duration);
                    continue;
                end
            catch ME
                    logLines{end+1} = sprintf('%s %dmin - error in timeWindowEEG - %s', logPrefix, window_duration, ME.message);
            continue;
            end

            %% (8) Save Spectrograms
            try
                saveAllElectrodeSpectrograms(tw_eeg_data, fs, eeg_channels, selected_labels, current_parent_dir, folder_name, window_duration);
                logLines{end+1} = sprintf('%s %dmin - success', logPrefix, window_duration);
            catch ME
                logLines{end+1} = sprintf('%s %dmin - error in saveAllElectrodeSpectrograms - %s', logPrefix, window_duration, ME.message);
            end

            % Track successful processing
            processed_files{end+1} = sprintf('%s_%dmin', folder_name, window_duration);
            diary off;
            
        end

        clearvars -except spectrogram_root logLines summary_log_path data_dir mat_files fs time_windows w window_duration numMat processed_files eegStruct num_datasets base_filename

    end
    % Clear eegStruct after processing all datasets in a file
    % Write per-person summary to the summary log
    fid = fopen(summary_log_path, 'a');  % Append mode
    fprintf(fid, '%s\n', logLines{:});
    fclose(fid);
    logLines = {};  % Clear for next person

    clear eegStruct
end

% Print summary of all processed datasets
fprintf('\nAll processed datasets:\n');
fprintf('%s\n', processed_files{:}); 

toc