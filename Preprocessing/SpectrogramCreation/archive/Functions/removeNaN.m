%% This script is on the Samsung USB

function [eeg_data, nanFlag] = removeNaN2(eeg_data)
    % REMOVE NAN VALUES BY INTERPOLATION ACROSS CHANNELS
    % eeg_data: input matrix of size Channels x Timepoints
    % nanFlag: true if too many NaNs remain and dataset should be skipped
    
    num_electrodes = size(eeg_data, 1); % Number of channels
    nanFlag = false; % Default: keep processing
    
    for electrode = 1:num_electrodes
        nan_idx = isnan(eeg_data(electrode, :)); 
        valid_idx = ~nan_idx; 
        
        if any(valid_idx) % Ensure at least one valid point exists
            eeg_data(electrode, nan_idx) = interp1(find(valid_idx), eeg_data(electrode, valid_idx), find(nan_idx), 'linear');
        end
    end
    
    % Check remaining NaNs
    total_nans_after = sum(isnan(eeg_data), 'all');
    fprintf('Total NaN values after interpolation: %d\n', total_nans_after);

    if total_nans_after == 0
        disp('All NaN values successfully interpolated.');
    elseif total_nans_after < 10
        eeg_data = fillmissing(eeg_data, 'previous', 2);
        disp('Filled remaining small number of NaNs with previous values.');
    else
        disp('Too many NaNs remain — skipping this dataset.');
        nanFlag = true;
        return; % Exit early
    end

    disp('NaN removal completed');  
end
