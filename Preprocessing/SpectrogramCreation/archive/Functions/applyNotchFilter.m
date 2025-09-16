function [nf_eeg_data, psd_orig, psd_filtered, f] = applyNotchFilter(eeg_data, fs, notchFreqs, Q, electrode_idx, plt)
    % APPLYNOTCHFILTERANDPLOTPSD - Applies notch filtering and optionally plots PSD.
    % 
    % Inputs:
    %   eeg_data      - EEG data matrix (Channels x Timepoints)
    %   fs            - Sampling frequency (Hz)
    %   notchFreqs    - Array of notch filter frequencies to remove
    %   Q             - Quality factor for notch filters
    %   electrode_idx - Electrode index for PSD visualization
    %   plt           - 1 to plot PSD, 0 to skip
    %
    % Outputs:
    %   nf_eeg_data   - Notch-filtered EEG data
    %   psd_orig      - PSD of original EEG data (optional)
    %   psd_filtered  - PSD of filtered EEG data (optional)
    %   f             - Frequency vector

    [num_electrodes, ~] = size(eeg_data); % Get number of channels
    nf_eeg_data = zeros(size(eeg_data)); % Initialize filtered data

    % Compute bandwidths
    BW1 = notchFreqs(1) / Q;
    BW2 = notchFreqs(2) / Q;
    BW3 = notchFreqs(3) / Q;

    % Precompute Butterworth notch filters
    [b1, a1] = butter(2, [(notchFreqs(1) - BW1/2), (notchFreqs(1) + BW1/2)] / (fs/2), 'stop');
    [b2, a2] = butter(2, [(notchFreqs(2) - BW2/2), (notchFreqs(2) + BW2/2)] / (fs/2), 'stop');
    [b3, a3] = butter(2, [(notchFreqs(3) - BW3/2), (notchFreqs(3) + BW3/2)] / (fs/2), 'stop');
    
    % Apply notch filters to each channel
    for electrode = 1:num_electrodes
        signal = eeg_data(electrode, :); % Extract signal
        
        % Apply the three precomputed notch filters
        signal = filtfilt(b1, a1, signal);
        signal = filtfilt(b2, a2, signal);
        signal = filtfilt(b3, a3, signal);
        
        % Store the filtered signal
        nf_eeg_data(electrode, :) = signal;
    end
    
    disp('Notch filtering completed.');

    % Compute PSD if plotting is requested
    if plt == 1
        % Compute PSD using Welch’s method
        [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
        [psd_filtered, ~] = pwelch(nf_eeg_data(electrode_idx, :), [], [], [], fs);
        
        % Plot PSD before and after filtering
        figure;
        
        subplot(2,1,1);
        plot(f, 10*log10(psd_orig), 'b');
        title(['Original PSD - Electrode ', num2str(electrode_idx)]);
        xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
        xlim([0, fs/2]); grid on;

        subplot(2,1,2);
        plot(f, 10*log10(psd_filtered), 'r');
        title(['Filtered PSD - Electrode ', num2str(electrode_idx)]);
        xlabel('Frequency (Hz)'); ylabel('Power (dB/Hz)');
        xlim([0, fs/2]); grid on;

        disp('PSD plotted for comparison.');
    else
        psd_orig = [];
        psd_filtered = [];
        f = [];
    end
end
