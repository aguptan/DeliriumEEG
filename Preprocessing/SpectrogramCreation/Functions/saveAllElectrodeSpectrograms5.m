function logStruct = saveAllElectrodeSpectrograms5(tw_eeg_data, fs, eeg_channels, selected_labels, ...
    parent_dir, folder_name, window_duration, logStruct, verbose)

    format long g
    if nargin < 9
        verbose = true;
    end

    % Parameters
    window_length  = 1600;           
    overlap_length = 1200;           
    df             = fs / window_length;  % 0.125 Hz
    f_low          = 0.5;                 
    f_high         = 45;                  
    f_vec          = f_low:df:f_high;     
    win            = hann(window_length, 'periodic');

    [~, num_timepoints, num_tw] = size(tw_eeg_data);

    if length(eeg_channels) ~= length(selected_labels)
        error('Mismatch between eeg_channels and selected_labels');
    end

    save_path = fullfile(parent_dir, folder_name);
    if ~exist(save_path, 'dir'), mkdir(save_path); end
    if verbose, disp(['Created main directory: ' save_path]); end

    % ------------------------------------------------------------------
    % Pass 1: compute spectrograms, store results, collect histogram data
    % ------------------------------------------------------------------
    all_vals = []; % vector for histogram values
    spec_store = cell(length(eeg_channels), num_tw); % store s_db results

    for ch_idx = 1:length(eeg_channels)
        electrode_label = selected_labels{ch_idx};

        for tw = 1:num_tw
            signal = squeeze(tw_eeg_data(ch_idx, :, tw));
            [s, f, t] = spectrogram(signal, win, overlap_length, f_vec, fs);
            s_power = abs(s).^2;              
            s_db    = 10 * log10(s_power + eps);

            % Store results for later plotting
            spec_store{ch_idx, tw} = struct( ...
                's_db', s_db, ...
                'f', f, ...
                't', t, ...
                'time_offset', (tw - 1) * (num_timepoints / fs) ...
            );

            % Collect values for percentile calculation
            all_vals = [all_vals; s_db(:)];
        end
    end

    % ------------------------------------------------------------------
    % Pass 2: derive global 5–95% color scaling
    % ------------------------------------------------------------------
    clim = prctile(all_vals, [5 95]);

    if verbose
        fprintf('Global CLim for spectrograms: [%.2f, %.2f] dB\n', clim(1), clim(2));
    end

    % ------------------------------------------------------------------
    % Pass 3: loop again and save images with fixed CLim
    % ------------------------------------------------------------------
    for ch_idx = 1:length(eeg_channels)
        electrode_label = selected_labels{ch_idx};
        prefix = matlab.lang.makeValidName(['electrode_' electrode_label]);

        electrode_dir = fullfile(save_path, electrode_label);
        if ~exist(electrode_dir, 'dir'), mkdir(electrode_dir); end
        if verbose, disp(['Created electrode folder: ' electrode_dir]); end

        for tw = 1:num_tw
            spec = spec_store{ch_idx, tw};
            s_db = spec.s_db;
            f    = spec.f;
            t    = spec.t + spec.time_offset;

            % Create figure and plot
            fig = figure('Visible', 'off', 'Position', [100 100 224 224]);
            ax = axes(fig, 'Position', [0 0 1 1]);
            imagesc(ax, t, f, s_db);
            set(gca, 'YDir', 'normal'); axis off; axis tight;
            colormap('parula');
            caxis(clim);

            save_filename = fullfile(electrode_dir, ...
                sprintf('Spectrogram_%s_%s_%dmin_TW%d.png', folder_name, electrode_label, window_duration, tw));
            exportgraphics(fig, save_filename, 'Resolution', 300);
            close(fig);

            % Append metadata into logStruct
            logStruct.timeWindowing.windows(tw).([prefix '_filename'])          = save_filename;
            logStruct.timeWindowing.windows(tw).([prefix '_image_size'])        = [224 224];
            logStruct.timeWindowing.windows(tw).([prefix '_frequency_range'])   = [f(1), f(end)];
            logStruct.timeWindowing.windows(tw).([prefix '_fs'])                = fs;
            logStruct.timeWindowing.windows(tw).([prefix '_freq_low'])          = f_low;
            logStruct.timeWindowing.windows(tw).([prefix '_freq_high'])         = f_high;
            logStruct.timeWindowing.windows(tw).([prefix '_df'])                = df;
            logStruct.timeWindowing.windows(tw).([prefix '_num_freq_bins'])     = numel(f);
            logStruct.timeWindowing.windows(tw).([prefix '_window'])            = window_length;
            logStruct.timeWindowing.windows(tw).([prefix '_noverlap'])          = overlap_length;
            logStruct.timeWindowing.windows(tw).([prefix '_window_type'])       = 'hann';
            logStruct.timeWindowing.windows(tw).([prefix '_is_db_power'])       = true;
            logStruct.timeWindowing.windows(tw).([prefix '_used_freq_vector'])  = true;
            logStruct.timeWindowing.windows(tw).([prefix '_clim'])              = clim; % store global limits
        end

        if verbose, disp(['Spectrograms saved for ' electrode_label]); end
    end

    if verbose, disp(['All spectrograms saved in ' save_path]); end
end
