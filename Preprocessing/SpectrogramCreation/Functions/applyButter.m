function [filtered_eeg_data, psd_orig, psd_filtered, f, logStruct] = applyButter(eeg_data, fs, hp_cutoff, filter_order, electrode_idx, plt, logStruct, verbose)
    % APPLYBUTTER - Applies high-pass Butterworth filter and logs step.
    %
    % Inputs:
    %   eeg_data      - EEG data matrix (Channels x Timepoints)
    %   fs            - Sampling frequency (Hz)
    %   hp_cutoff     - High-pass cutoff frequency (Hz)
    %   filter_order  - Butterworth filter order
    %   electrode_idx - Electrode index for PSD visualization
    %   plt           - 1 to plot PSD, 0 to skip
    %   logStruct     - Existing log structure to append to
    %   verbose       - true/false for printing/logging
    %
    % Outputs:
    %   filtered_eeg_data - Filtered EEG
    %   psd_orig          - Pre-filter PSD (if plotted)
    %   psd_filtered      - Post-filter PSD (if plotted)
    %   f                 - Frequency vector
    %   logStruct         - Updated log with .highpass entry

    if nargin < 8
        verbose = true;
    end

    [num_electrodes, ~] = size(eeg_data);
    filtered_eeg_data = zeros(size(eeg_data));

    try
        % Design filter
        [b, a] = butter(filter_order, hp_cutoff / (fs / 2), 'high');

        % Apply to each electrode
        for electrode = 1:num_electrodes
            filtered_eeg_data(electrode, :) = filtfilt(b, a, eeg_data(electrode, :));
        end

        if verbose, disp('Butterworth filtering completed.'); end

        % PSD
        if plt == 1
            [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
            [psd_filtered, ~] = pwelch(filtered_eeg_data(electrode_idx, :), [], [], [], fs);

            figure;
            plot(f, 10*log10(psd_orig), 'b', 'LineWidth', 1.5); hold on;
            plot(f, 10*log10(psd_filtered), 'r', 'LineWidth', 1.5);
            title(['Welch Power Spectral Density (PSD) - Electrode ', num2str(electrode_idx)]);
            xlabel('Frequency (Hz)');
            ylabel('Power (dB/Hz)');
            legend('Original EEG', 'Filtered EEG');
            grid on;
            xlim([0, 100]);

            if verbose, disp(['PSD plotted for electrode ', num2str(electrode_idx), '.']); end
        else
            psd_orig = [];
            psd_filtered = [];
            f = [];
        end

        % Append to log
        logStruct.highpass = struct( ...
            'status', 'applied', ...
            'cutoff_Hz', hp_cutoff, ...
            'order', filter_order, ...
            'numChannels', num_electrodes, ...
            'usedPSDplot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', '' ...
        );

    catch ME
        % On failure
        filtered_eeg_data = [];
        psd_orig = [];
        psd_filtered = [];
        f = [];

        logStruct.highpass = struct( ...
            'status', ['error: ', ME.message], ...
            'cutoff_Hz', hp_cutoff, ...
            'order', filter_order, ...
            'numChannels', size(eeg_data, 1), ...
            'usedPSDplot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', 'Filtering failed in applyButter' ...
        );

        if verbose
            fprintf('Error in applyButter: %s\n', ME.message);
        end
    end
end
