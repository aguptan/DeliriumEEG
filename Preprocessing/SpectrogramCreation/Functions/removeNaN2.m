function [eeg_data, nanFlag, logStruct] = removeNaN2(eeg_data, logStruct, verbose)
    % REMOVE NAN VALUES BY INTERPOLATION ACROSS CHANNELS
    %
    % Inputs:
    %   eeg_data  - EEG data matrix (Channels x Timepoints)
    %   logStruct - existing preprocessing log (struct)
    %   verbose   - optional, default true
    %
    % Outputs:
    %   eeg_data  - cleaned EEG data
    %   nanFlag   - true if too many NaNs remain and should skip dataset
    %   logStruct - updated with .removeNaN field

    if nargin < 3
        verbose = true;
    end

    num_electrodes = size(eeg_data, 1);
    nanFlag = false;
    total_nans_before = sum(isnan(eeg_data), 'all');
    usedInterpolation = false;
    usedFillMissing = false;

    % Interpolation step
    for electrode = 1:num_electrodes
        nan_idx = isnan(eeg_data(electrode, :)); 
        valid_idx = ~nan_idx; 
        if any(valid_idx) && any(nan_idx)
            eeg_data(electrode, nan_idx) = interp1(find(valid_idx), eeg_data(electrode, valid_idx), find(nan_idx), 'linear');
            usedInterpolation = true;
        end
    end

    % Post-processing
    total_nans_after = sum(isnan(eeg_data), 'all');
    if verbose
        fprintf('Total NaN values after interpolation: %d\n', total_nans_after);
    end

    if total_nans_after == 0
        status = 'interpolated';
        if verbose, disp('All NaN values successfully interpolated.'); end
    elseif total_nans_after < 10
        eeg_data = fillmissing(eeg_data, 'previous', 2);
        usedFillMissing = true;
        status = 'interpolated + filledmissing';
        if verbose, disp('Filled remaining small number of NaNs with previous values.'); end
    else
        nanFlag = true;
        status = 'too many NaNs';
        if verbose, disp('Too many NaNs remain — skipping this dataset.'); end
    end

    if verbose, disp('NaN removal completed'); end

    % Append to log
    logStruct.removeNaN = struct( ...
        'status', status, ...
        'totalNaNsBefore', total_nans_before, ...
        'totalNaNsAfter', total_nans_after, ...
        'usedInterpolation', usedInterpolation, ...
        'usedFillMissing', usedFillMissing, ...
        'timestamp', datetime('now'), ...
        'notes', '' ...
    );
end
