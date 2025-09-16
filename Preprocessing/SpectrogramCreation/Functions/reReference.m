function [rreeg_data, logStruct] = reReference(filtered_eeg_data, logStruct, verbose)
    % REREFERENCE - Performs common average referencing (CAR) and logs it.
    %
    % Inputs:
    %   filtered_eeg_data - EEG data after filtering (Channels x Timepoints)
    %   logStruct         - cumulative preprocessing log
    %   verbose           - true/false, controls print output
    %
    % Outputs:
    %   rreeg_data        - Re-referenced EEG data
    %   logStruct         - Updated log with .reRef field

    if nargin < 3
        verbose = true;
    end

    try
        % Compute common average
        common_mean_signal = mean(filtered_eeg_data, 1); % mean across channels at each timepoint

        % Subtract from all channels
        rreeg_data = filtered_eeg_data - common_mean_signal;

        if verbose, disp('Re-referencing completed.'); end

        logStruct.reRef = struct( ...
            'status', 'common average', ...
            'numChannels', size(filtered_eeg_data, 1), ...
            'timestamp', datetime('now'), ...
            'notes', '' ...
        );
    catch ME
        rreeg_data = [];
        logStruct.reRef = struct( ...
            'status', ['error: ', ME.message], ...
            'numChannels', size(filtered_eeg_data, 1), ...
            'timestamp', datetime('now'), ...
            'notes', 'Re-referencing failed in reReference' ...
        );

        if verbose
            fprintf('Error in reReference: %s\n', ME.message);
        end
    end
end
