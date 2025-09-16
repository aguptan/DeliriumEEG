function saveAllElectrodeSpectrograms5(tw_eeg_data, fs, eeg_channels, selected_labels, parent_dir, folder_name, window_duration)
% SAVEALLELECTRODESPECTROGRAMS5 - Computes and saves spectrograms for all electrodes and time windows.
%
% Inputs:
%   tw_eeg_data     - EEG data (Channels x Timepoints x TimeWindows)
%   fs              - Sampling frequency (Hz)
%   eeg_channels    - Indices of electrodes to process
%   selected_labels - Electrode labels (cell array)
%   parent_dir      - Time-window-specific parent directory (e.g., E:\...\Spectrograms\15min)
%   folder_name     - Dataset folder name (e.g., AM_1)
%   window_duration - Current time window duration (in minutes)

    % Spectrogram parameters
    window_length = 200;
    overlap_length = window_length / 2;
    nfft = window_length;

    [~, num_timepoints, num_tw] = size(tw_eeg_data);

    % Confirm label count matches
    if length(eeg_channels) ~= length(selected_labels)
        error('Mismatch between eeg_channels and selected_labels');
    end

    % Folder where all spectrograms for this dataset will go
    save_path = fullfile(parent_dir, folder_name);
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end
    disp(['Created main directory: ' save_path]);

    % Loop through all electrodes
    for ch_idx = 1:length(eeg_channels)
        electrode_label = selected_labels{ch_idx};

        % Folder for individual electrode
        electrode_dir = fullfile(save_path, electrode_label);
        if ~exist(electrode_dir, 'dir')
            mkdir(electrode_dir);
        end
        disp(['Created electrode folder: ' electrode_dir]);

        % Loop through each time window
        for tw = 1:num_tw
            signal = squeeze(tw_eeg_data(ch_idx, :, tw));
            [s, f, t] = spectrogram(signal, window_length, overlap_length, nfft, fs);
            s_db = 10 * log10(abs(s) + eps);

            % Adjust time axis to reflect offset of this segment
            time_offset = (tw - 1) * (num_timepoints / fs);
            t_adjusted = t + time_offset;

            % Generate and save figure
            fig = figure('Visible', 'off', 'Position', [100 100 224 224]);
            ax = axes(fig, 'Position', [0 0 1 1]);
            imagesc(ax, t_adjusted, f, s_db);

            set(gca, 'YDir', 'normal');
            axis off;
            axis tight;
            colormap('jet');
            caxis([min(s_db(:)), max(s_db(:))]);

            % Format: Spectrogram_AM_1_C4_15min_TW1.jpg
            save_filename = fullfile(electrode_dir, ...
                sprintf('Spectrogram_%s_%s_%dmin_TW%d.jpg', folder_name, electrode_label, window_duration, tw));

            exportgraphics(fig, save_filename, 'Resolution', 300);
            close(fig);
        end

        disp(['Spectrograms saved for ' electrode_label]);
    end

    disp(['All spectrograms saved in ' save_path]);
end


