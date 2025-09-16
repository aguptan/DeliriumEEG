function [tw_eeg_data, skipFlag, logStruct] = timeWindowEEG2(rreeg_data, fs, window_duration, logStruct, verbose)
    format long g
    % TIMEWINDOWEEG2 - Segments EEG data and embeds metadata into logStruct

    if nargin < 5
        verbose = true;
    end

    % Ensure logStruct.timeWindowing exists
    if ~isfield(logStruct, 'timeWindowing')
        logStruct.timeWindowing = struct();
    end

    [num_electrodes, num_timepoints] = size(rreeg_data);
    window_size = window_duration * 60 * fs;
    skipFlag = false;

    try
        % Case: dataset too short
        if num_timepoints < window_size
            tw_eeg_data = [];
            skipFlag = true;

            logStruct.timeWindowing.status              = 'skipped (too short)';
            logStruct.timeWindowing.window_duration_min = window_duration;
            logStruct.timeWindowing.samples_per_window  = window_size;
            logStruct.timeWindowing.num_input_samples   = num_timepoints;
            logStruct.timeWindowing.num_output_windows  = 0;
            logStruct.timeWindowing.trim                = 0;
            logStruct.timeWindowing.timestamp           = datetime('now');
            logStruct.timeWindowing.notes               = '';
            logStruct.timeWindowing.windows             = struct([]);


            if verbose
                disp('Dataset too short for one full time window — skipping.');
            end
            return;
        end

        % Compute trimming
        num_windows     = floor(num_timepoints / window_size);
        trimmed_length  = num_windows * window_size;
        excess_samples  = num_timepoints - trimmed_length;
        trim_start      = floor(excess_samples / 2);
        trim_end        = excess_samples - trim_start;

        % Segment into windows
        eeg_data_trimmed = rreeg_data(:, trim_start + 1 : end - trim_end);
        tw_eeg_data      = reshape(eeg_data_trimmed, num_electrodes, window_size, num_windows);

        % Metadata for each window
        windows = struct([]);
        for i = 1:num_windows
            s_start = trim_start + 1 + (i - 1) * window_size;
            s_end   = s_start + window_size - 1;
            t_start = s_start / fs;
            t_end   = s_end / fs;

            windows(i).index        = i;
            windows(i).start_sample = s_start;
            windows(i).end_sample   = s_end;
            windows(i).start_time   = t_start;
            windows(i).end_time     = t_end;
            windows(i).spectrograms = [];
        end

        % Update logStruct
        logStruct.timeWindowing.status              = 'success';
        logStruct.timeWindowing.window_duration_min = window_duration;
        logStruct.timeWindowing.samples_per_window  = window_size;
        logStruct.timeWindowing.num_input_samples   = num_timepoints;
        logStruct.timeWindowing.num_output_windows  = num_windows;
        logStruct.timeWindowing.trim_start          = trim_start;
        logStruct.timeWindowing.trim_end            = trim_end;
        logStruct.timeWindowing.timestamp           = datetime('now');
        logStruct.timeWindowing.notes               = '';
        logStruct.timeWindowing.windows             = windows;

        if verbose
            disp('Reshaping complete');
            fprintf('Trimmed samples: start = %d, end = %d\n', trim_start, trim_end);
            fprintf('Total windows: %d | Samples per window: %d\n', num_windows, window_size);
        end

    catch ME
        tw_eeg_data = [];
        skipFlag = true;

        logStruct.timeWindowing.status              = ['error: ', ME.message];
        logStruct.timeWindowing.window_duration_min = window_duration;
        logStruct.timeWindowing.samples_per_window  = window_size;
        logStruct.timeWindowing.num_input_samples   = num_timepoints;
        logStruct.timeWindowing.num_output_windows  = 0;
        logStruct.timeWindowing.trim_start          = NaN;
        logStruct.timeWindowing.trim_end            = NaN;
        logStruct.timeWindowing.timestamp           = datetime('now');
        logStruct.timeWindowing.notes               = 'Exception thrown in timeWindowEEG2';
        logStruct.timeWindowing.windows             = struct([]);
    end
end
