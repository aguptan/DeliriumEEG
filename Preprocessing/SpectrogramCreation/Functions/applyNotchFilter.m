function [nf_eeg_data, psd_orig, psd_filtered, f, logStruct] = applyNotchFilter(eeg_data, fs, notchFreqs, Q, electrode_idx, plt, logStruct, verbose)
    % APPLYNOTCHFILTER - Applies notch filtering and logs outcome in logStruct.
    %
    % Inputs:
    %   eeg_data      - EEG data matrix (Channels x Timepoints)
    %   fs            - Sampling frequency (Hz)
    %   notchFreqs    - Array of notch filter frequencies to remove
    %   Q             - Quality factor for notch filters
    %   electrode_idx - Electrode index for PSD visualization
    %   plt           - 1 to plot PSD, 0 to skip
    %   logStruct     - existing cumulative log structure
    %   verbose       - true/false for print/log output
    %
    % Outputs:
    %   nf_eeg_data   - Notch-filtered EEG data
    %   psd_orig      - PSD of original EEG data (if plotted)
    %   psd_filtered  - PSD of filtered EEG data (if plotted)
    %   f             - Frequency vector
    %   logStruct     - Updated log structure with .notch field added

    if nargin < 8
        verbose = true;
    end

    [num_electrodes, ~] = size(eeg_data);
    nf_eeg_data = zeros(size(eeg_data));

    try
        % Precompute filters
        BWs = notchFreqs ./ Q;
        filters = cell(1, length(notchFreqs));
        for i = 1:length(notchFreqs)
            f0 = notchFreqs(i);
            BW = BWs(i);
            [b, a] = butter(2, [(f0 - BW/2), (f0 + BW/2)] / (fs/2), 'stop');
            filters{i} = {b, a};
        end

        % Apply filters
        for electrode = 1:num_electrodes
            signal = eeg_data(electrode, :);
            for i = 1:length(filters)
                b = filters{i}{1}; a = filters{i}{2};
                signal = filtfilt(b, a, signal);
            end
            nf_eeg_data(electrode, :) = signal;
        end

        if verbose, disp('Notch filtering completed.'); end

        % PSD if requested
        if plt == 1
            [psd_orig, f] = pwelch(eeg_data(electrode_idx, :), [], [], [], fs);
            [psd_filtered, ~] = pwelch(nf_eeg_data(electrode_idx, :), [], [], [], fs);

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

            if verbose, disp('PSD plotted for comparison.'); end
        else
            psd_orig = [];
            psd_filtered = [];
            f = [];
        end

        % Append to logStruct
        logStruct.notch = struct( ...
            'status', 'applied', ...
            'notchFreqs', notchFreqs, ...
            'Q', Q, ...
            'numChannels', num_electrodes, ...
            'usedPSDplot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', '' ...
        );

    catch ME
        % Handle failure
        nf_eeg_data = [];
        psd_orig = [];
        psd_filtered = [];
        f = [];

        logStruct.notch = struct( ...
            'status', ['error: ', ME.message], ...
            'notchFreqs', notchFreqs, ...
            'Q', Q, ...
            'numChannels', size(eeg_data, 1), ...
            'usedPSDplot', logical(plt), ...
            'timestamp', datetime('now'), ...
            'notes', 'Filtering failed in applyNotchFilter' ...
        );

        if verbose
            fprintf('Error in applyNotchFilter: %s\n', ME.message);
        end
    end
end
