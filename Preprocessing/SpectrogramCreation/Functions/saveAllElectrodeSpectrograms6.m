function logStruct = saveAllElectrodeSpectrograms6(tw_eeg_data, fs, eeg_channels, selected_labels, ...
    parent_dir, folder_name, window_duration, logStruct, verbose)
    format long g
    if nargin < 9
        verbose = true;
    end

   
    window_length  = 1600;           
    overlap_length = 1200;           
    df             = fs / window_length;  % 0.125 Hz
    f_low          = 0.5;            % Hz
    f_high         = 45;             % Hz
    f_vec          = f_low:df:f_high;     % frequency vector for spectrogram
    % Note: we no longer need 'nfft' when using a frequency vector
    win = hann(window_length, 'periodic');


    [~, num_timepoints, num_tw] = size(tw_eeg_data);

    if length(eeg_channels) ~= length(selected_labels)
        error('Mismatch between eeg_channels and selected_labels');
    end

    save_path = fullfile(parent_dir, folder_name);
    if ~exist(save_path, 'dir'), mkdir(save_path); end
    if verbose, disp(['Created main directory: ' save_path]); end

    % Loop through all electrodes
    for ch_idx = 1:length(eeg_channels)
        electrode_label = selected_labels{ch_idx};
        prefix = matlab.lang.makeValidName(['electrode_' electrode_label]);

        electrode_dir = fullfile(save_path, electrode_label);
        if ~exist(electrode_dir, 'dir'), mkdir(electrode_dir); end
        if verbose, disp(['Created electrode folder: ' electrode_dir]); end

        for tw = 1:num_tw
            signal = squeeze(tw_eeg_data(ch_idx, :, tw));
            [s, f, t] = spectrogram(signal, win, overlap_length, f_vec, fs);
            s_power = abs(s).^2;              % power
            s_db    = 10 * log10(s_power + eps);

            time_offset = (tw - 1) * (num_timepoints / fs);
            t_adjusted = t + time_offset;

            fig = figure('Visible', 'off', 'Position', [100 100 224 224]);
            ax = axes(fig, 'Position', [0 0 1 1]);
            imagesc(ax, t_adjusted, f, s_db);
            set(gca, 'YDir', 'normal'); axis off; axis tight;
            colormap('parula');  % clearer for red-green color vision
            caxis([min(s_db(:)), max(s_db(:))]);  % keep as-is (optional fix below)


            save_filename = fullfile(electrode_dir, ...
                sprintf('Spectrogram_%s_%s_%dmin_TW%d.png', folder_name, electrode_label, window_duration, tw));
            exportgraphics(fig, save_filename, 'Resolution', 300);
            close(fig);

            % Append metadata into existing window entry
            % SECTION D — Metadata block (where you append to logStruct.timeWindowing.windows(tw))
            logStruct.timeWindowing.windows(tw).([prefix '_filename'])          = save_filename;
            logStruct.timeWindowing.windows(tw).([prefix '_image_size'])        = [224 224];
            logStruct.timeWindowing.windows(tw).([prefix '_frequency_range'])   = [f(1), f(end)];
            logStruct.timeWindowing.windows(tw).([prefix '_fs'])                = fs;
            
            % Frequency-vector metadata (replaces legacy nfft meaning)
            logStruct.timeWindowing.windows(tw).([prefix '_freq_low'])          = f_low;
            logStruct.timeWindowing.windows(tw).([prefix '_freq_high'])         = f_high;
            logStruct.timeWindowing.windows(tw).([prefix '_df'])                = df;
            logStruct.timeWindowing.windows(tw).([prefix '_num_freq_bins'])     = numel(f);
            
            % Time-domain windowing details
            logStruct.timeWindowing.windows(tw).([prefix '_window'])            = window_length;
            logStruct.timeWindowing.windows(tw).([prefix '_noverlap'])          = overlap_length;
            
            % New: document spectral choices
            logStruct.timeWindowing.windows(tw).([prefix '_window_type'])       = 'hann';
            logStruct.timeWindowing.windows(tw).([prefix '_is_db_power'])       = true;
            
            % Optional flag to signal frequency-vector usage
            logStruct.timeWindowing.windows(tw).([prefix '_used_freq_vector'])  = true;


        end

        if verbose, disp(['Spectrograms saved for ' electrode_label]); end
    end

    if verbose, disp(['All spectrograms saved in ' save_path]); end
end
