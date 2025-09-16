%% This script is on the Samsung USB

function [tw_eeg_data, num_tw, skipFlag] = timeWindowEEG2(rreeg_data, fs, window_duration)
    % TIMEWINDOWEEG - Segments EEG data into fixed-length time windows.
    % Now includes a check for insufficient data length.
    
    [num_electrodes, num_timepoints] = size(rreeg_data);
    window_size = window_duration * 60 * fs; % Samples per window
    skipFlag = false; % Default: process as normal

    if num_timepoints < window_size
        disp('Dataset too short for one full time window — skipping.');
        tw_eeg_data = [];
        num_tw = 0;
        skipFlag = true;
        return;
    end

    % Compute full windows
    num_windows = floor(num_timepoints / window_size);
    trimmed_length = num_windows * window_size;
    excess_samples = num_timepoints - trimmed_length;
    
    trim_start = floor(excess_samples / 2);
    trim_end = excess_samples - trim_start;
    
    eeg_data_trimmed = rreeg_data(:, trim_start + 1 : end - trim_end);
    tw_eeg_data = reshape(eeg_data_trimmed, num_electrodes, window_size, num_windows);
    num_tw = size(tw_eeg_data, 3);

    % Display info
    disp('Reshaping Complete');
    disp(['Original Data Size: ', num2str(num_electrodes), ' x ', num2str(num_timepoints)]);
    disp(['Trimmed Data Start Index: ', num2str(trim_start + 1)]);
    disp(['Trimmed Data End Index: ', num2str(num_timepoints - trim_end)]);
    disp(['New Data Size (Channels x Timepoints/Window x Windows): ', ...
        num2str(size(tw_eeg_data,1)), ' x ', num2str(size(tw_eeg_data,2)), ' x ', num2str(size(tw_eeg_data,3))]);
    
    clear eeg_data_trimmed excess_samples num_windows trim_start trim_end trimmed_length window_size;
end
