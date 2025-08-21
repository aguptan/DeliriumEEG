function [filtered_eeg_data, psd_orig, psd_filtered, f] = applyButter(eeg_data, fs, hp_cutoff, filter_order, electrode_idx, plt)
    % APPLYBUTTERWORTHFILTER - Applies a high-pass Butterworth filter and optionally plots PSD.
    %
    % Inputs:
    %   eeg_data      - EEG data matrix (Channels x Timepoints)
    %   fs            - Sampling frequency (Hz)
    %   hp_cutoff     - High-pass cutoff frequency (Hz)
    %   filter_order  - Order of the Butterworth filter
    %   electrode_idx - Electrode index for PSD visualization
    %   plt           - 1 to plot PSD, 0 to skip
    %
    % Outputs:
    %   filtered_eeg_data - High-pass filtered EEG data
    %   psd_orig          - PSD of original EEG data (optional)
    %   psd_filtered      - PSD of filtered EEG data (optional)
    %   f                 - Frequency vector
    
    [num_electrodes, ~] = size(eeg_data); % Get number of channels
    filtered_eeg_data = zeros(size(eeg_data)); % Initialize filtered data

    % Design high-pass Butterworth filter
    [b, a] = butter(filter_order, hp_cutoff / (fs / 2), 'high');

    % Apply filtering to each channel
    for electrode = 1:num_electrodes  
        signal = eeg_data(electrode, :); % Extract signal
        filtered_eeg_data(electrode, :) = filtfilt(b, a, signal); % Apply high-pass filter
    end

    disp('Butterworth filtering completed.');

    % Compute PSD if plotting is requested
    if plt == 1
        % Compute PSD using Welch’s method
        [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
        [psd_filtered, ~] = pwelch(filtered_eeg_data(electrode_idx, :), [], [], [], fs);
        
        % Plot PSD before and after filtering
        figure;
        plot(f, 10*log10(psd_orig), 'b', 'LineWidth', 1.5); hold on;
        plot(f, 10*log10(psd_filtered), 'r', 'LineWidth', 1.5);
        title(['Welch Power Spectral Density (PSD) - Electrode ', num2str(electrode_idx)]);
        xlabel('Frequency (Hz)');
        ylabel('Power (dB/Hz)');
        legend('Original EEG', 'Filtered EEG');
        grid on;
        xlim([0, 100]); % Extend frequency range to 100 Hz

        disp(['PSD plotted for electrode ', num2str(electrode_idx), '.']);
    else
        psd_orig = [];
        psd_filtered = [];
        f = [];
    end
end
