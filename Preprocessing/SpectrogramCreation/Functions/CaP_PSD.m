function [psd_orig, psd_filtered, f] = CaP_PSD(eeg_data, rreeg_data, fs, electrode_idx, plt)
    % COMPUTEANDPLOTPSD - Computes and optionally plots the Power Spectral Density (PSD)
    % for a given electrode.
    %
    % Inputs:
    %   eeg_data      - Original EEG data matrix (Channels x Timepoints)
    %   rreeg_data    - Processed EEG data after re-referencing (Channels x Timepoints)
    %   fs            - Sampling frequency (Hz)
    %   electrode_idx - Index of the electrode to analyze
    %   plt           - 1 to compute & plot PSD, 0 to skip
    %
    % Outputs:
    %   psd_orig      - PSD of original EEG data (empty if plt = 0)
    %   psd_filtered  - PSD of filtered EEG data (empty if plt = 0)
    %   f             - Frequency vector (empty if plt = 0)

    if plt == 1
        % Compute PSD using Welch’s method
        [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
        [psd_filtered, ~] = pwelch(rreeg_data(electrode_idx, :), [], [], [], fs);
        
        % Plot PSD
        figure;
        
        % Original PSD
        subplot(2,1,1);
        plot(f, 10*log10(psd_orig), 'b');
        title(['Original PSD - Electrode ', num2str(electrode_idx)]);
        xlabel('Frequency (Hz)');
        ylabel('Power (dB/Hz)');
        xlim([0, fs/2]); 
        grid on;

        % Filtered PSD
        subplot(2,1,2);
        plot(f, 10*log10(psd_filtered), 'r');
        title(['Re-Referenced PSD - Electrode ', num2str(electrode_idx)]);
        xlabel('Frequency (Hz)');
        ylabel('Power (dB/Hz)');
        xlim([0, fs/2]); 
        grid on;

        disp('PSD plotted for comparison.');
    else
        psd_orig = [];
        psd_filtered = [];
        f = [];
    end
end
