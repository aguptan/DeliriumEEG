%% --- 1. SETUP ---
clear; clc; close all;
fprintf('--- Starting Sequential Processing ---\n\n');

% --- DEFINE CORE PATHS ---
% 1. INPUT DIRECTORY: Path to the root folder containing patient subfolders
raw_data_root = 'Location';

% 2. OUTPUT DIRECTORY: Path where the processed .mat files will be saved
output_dir = 'Location';

% 3. TOOLBOX LOCATION: Path to the BioSig toolbox needed for reading EEG files
biosig_path = 'Location'; 

% --- VALIDATE PATHS AND SETUP ---
% Create the output directory if it doesn't exist
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(biosig_path, 'dir')
    error('BioSig toolbox path not found: %s', biosig_path);
end
    
% Get a list of all patient folders to be processed
all_folders = dir(raw_data_root);
patient_folders = all_folders([all_folders.isdir] & ~ismember({all_folders.name}, {'.', '..'}));

if isempty(patient_folders)
    error('No patient folders found in the specified raw data directory: %s', raw_data_root);
end

fprintf('Found %d patient folders to process.\n', length(patient_folders));
fprintf('============================================================\n');


%% --- 2. MAIN PROCESSING LOOP ---
% This loop iterates through each patient folder one by one.
tic; % Start a timer to measure total processing time

for i = 1:length(patient_folders)
    
    % --- SETUP FOR THE CURRENT PATIENT ---
    patient_folder_path = fullfile(raw_data_root, patient_folders(i).name);
    [~, patient_id, ~] = fileparts(patient_folder_path);
    
    fprintf('--> [%d/%d] Starting processing for patient: %s\n', i, length(patient_folders), patient_id);
    
    % --- BEGIN PROCESSING LOGIC FOR THIS PATIENT ---
    try
        % Find all .EEG files in the current patient's folder
        eeg_files_struct = dir(fullfile(patient_folder_path, '*.EEG'));
        
        if isempty(eeg_files_struct)
            fprintf('    No .EEG files found for %s. Skipping to next patient.\n', patient_id);
            fprintf('------------------------------------------------------------\n');
            continue; % Skips the rest of the loop for this patient
        end
        
        % Prepare to collect data from all files for this patient
        eeg_filenames = {eeg_files_struct.name};
        num_files = length(eeg_filenames);
        eegStruct = struct('Filename', {}, 'Data', {}, 'Labels', {});
        
        % --- Inner loop: Process each .EEG file for the current patient ---
        for j = 1:num_files
            current_filename = eeg_filenames{j};
            eeg_file_path = fullfile(patient_folder_path, current_filename);
            fprintf('    -> Processing file %d of %d: %s\n', j, num_files, current_filename);
            
            try
                % Add the toolbox path, read the data, then remove the path
                addpath(biosig_path);
                warning('off', 'SOPEN:overflow');
                warning('off', 'BioSig:sopen:overflow');
                
                hdr = sopen(eeg_file_path, 'r');
                data = sread(hdr);
                hdr = sclose(hdr);
                
                rmpath(biosig_path);
                
                % Store the formatted data in the structure
                eegStruct(j).Filename = current_filename;
                eegStruct(j).Data = data'; % Transpose data
                eegStruct(j).Labels = hdr.Label;
                
                clear data; % IMPORTANT: Clear large data variable to save memory
                
            catch ME_file
                fprintf('    ERROR on file %s: %s\n', current_filename, ME_file.message);
                % Ensure biosig path is removed even if an error occurred
                if exist(biosig_path, 'dir'), rmpath(biosig_path); end
            end
        end
        
        % --- Save the combined data for the current patient ---
        if ~isempty(eegStruct)
            output_filename = fullfile(output_dir, [patient_id, '.mat']);
            fprintf('    SAVING combined data for %s to: %s\n', patient_id, output_filename);
            save(output_filename, 'eegStruct', '-v7.3');
        else
             fprintf('    No files were processed successfully for %s.\n', patient_id);
        end
        
    catch ME_patient
        fprintf('  A FATAL ERROR occurred while processing patient %s: %s\n', patient_id, ME_patient.message);
        fprintf('  Attempting to continue with the next patient...\n');
    end
    
    fprintf('--> [%d/%d] Finished processing for patient: %s\n', i, length(patient_folders), patient_id);
    fprintf('------------------------------------------------------------\n');
end

total_time = toc; % Stop the timer

%% --- 3. FINAL SUMMARY ---
fprintf('\n============================================================\n');
fprintf('All patients have been processed.\n');
fprintf('Total elapsed time: %.2f minutes.\n', total_time / 60);