function [psd_orig, psd_filtered, f, logStruct] = CaP_PSD(eeg_data, rreeg_data, fs, electrode_idx, plt, logStruct, verbose)
    % CaP_PSD - Computes and optionally plots PSD, logging the step.
    %
    % Inputs:
    %   eeg_data      - Original EEG (Channels x Timepoints)
    %   rreeg_data    - Re-referenced EEG (Channels x Timepoints)
    %   fs            - Sampling rate in Hz
    %   electrode_idx - Electrode index to visualize
    %   plt           - 1 to compute and plot, 0 to skip
    %   logStruct     - cumulative preprocessing log
    %   verbose       - true/false to toggle output
    %
    % Outputs:
    %   psd_orig      - Original PSD
    %   psd_filtered  - Re-referenced PSD
    %   f             - Frequency vector
    %   logStruct     - Updated log with .psdPlot field

    if nargin < 7
        verbose = true;
    end

    try
        if plt == 1
            [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
            [psd_filtered, ~] = pwelch(rreeg_data(electrode_idx, :), [], [], [], fs);

            figure;
            subplot(2,1,1);
            plot(f, 10*log10(psd_orig), 'b');
            title(['Original PSD - Electrode ', num2str(electrode_idx)]);
            xlabel('Frequency (Hz)');
            ylabel('Power (dB/Hz)');
            xlim([0, fs/2]); 
            grid on;

            subplot(2,1,2);
            plot(f, 10*log10(psd_filtered), 'r');
            title(['Re-Referenced PSD - Electrode ', num2str(electrode_idx)]);
            xlabel('Frequency (Hz)');
            ylabel('Power (dB/Hz)');
            xlim([0, fs/2]); 
            grid on;

            if verbose, disp('PSD plotted for comparison.'); end
        else
            psd_orig = [];
            psd_filtered = [];
            f = [];
        end

        % Logging
        logStruct.psdPlot = struct( ...
            'status', 'plotted', ...
            'electrode', electrode_idx, ...
            'usedPlot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', '' ...
        );

    catch ME
        psd_orig = [];
        psd_filtered = [];
        f = [];

        logStruct.psdPlot = struct( ...
            'status', ['error: ', ME.message], ...
            'electrode', electrode_idx, ...
            'usedPlot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', 'Failed in CaP_PSD' ...
        );

        if verbose
            fprintf('Error in CaP_PSD: %s\n', ME.message);
        end
    end
end
