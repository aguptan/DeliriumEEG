function rreeg_data = reReference(filtered_eeg_data)
    % REREFERENCE - Performs common average referencing (CAR) on EEG data.
    %
    % Inputs:
    %   filtered_eeg_data - EEG data after filtering (Channels x Timepoints)
    %
    % Outputs:
    %   rreeg_data        - Re-referenced EEG data (Common Mean Subtracted)
    
    % Compute the common mean across EEG channels
    common_mean_signal = mean(filtered_eeg_data, 2); % Average across EEG channels
    
    % Subtract the common mean from each EEG channel
    rreeg_data = filtered_eeg_data - common_mean_signal; 
    
    % Display message
    disp('Re-referencing completed.');
    
    % Clear temporary variables (MATLAB won't carry them over outside the function)
    clear common_mean_signal
end
