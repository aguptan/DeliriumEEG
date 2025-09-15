%% =========================
%% INTEGRATED EEG PIPELINE
%% 1) Convert raw .EEG -> per-patient .mat (eegStruct)  [SEQUENTIAL]
%% 2) Parallel preprocessing + spectrogram export       [PARALLEL]
%% =========================
clear; clc; close all;
%% -------- TOP-LEVEL DIRECTORIES (EDIT THESE ONLY) --------
RAW_DATA_ROOT   = 'Location_to_raw_EEG_root';      % Data location: root folder with patient subfolders containing .EEG
EEG2MAT_DIR     = 'Location_to_save_EEG2MAT';      % EEG2MAT: where per-patient .mat files (eegStruct) will be saved
SPECTROGRAM_DIR = 'Location_of_Final_Directory';   % Final spectrogram directory
BIOSIG_PATH     = 'BioSig_Toolbox_directory';
FUNC_PATH       = 'FunctionsICreated'; % functions for stage 2
%% --------------------------------------------------------


%% =========================
%% --- SEQUENTIAL: EEG -> MAT (eegStruct) ---
%% =========================

fprintf('--- Starting Sequential Processing (EEG -> MAT) ---\n\n');

% Validate & prepare paths
if ~exist(EEG2MAT_DIR, 'dir'), mkdir(EEG2MAT_DIR); end
if ~exist(BIOSIG_PATH, 'dir')
    error('BioSig toolbox path not found: %s', BIOSIG_PATH);
end
if ~exist(RAW_DATA_ROOT, 'dir')
    error('Raw data root not found: %s', RAW_DATA_ROOT);
end

% Discover patient folders
all_folders     = dir(RAW_DATA_ROOT);
patient_folders = all_folders([all_folders.isdir] & ~ismember({all_folders.name}, {'.', '..'}));
if isempty(patient_folders)
    error('No patient folders found in raw data directory: %s', RAW_DATA_ROOT);
end
fprintf('Found %d patient folders to process.\n', numel(patient_folders));
fprintf('============================================================\n');

tic; % total sequential time
for i = 1:numel(patient_folders)
    % Current patient folder
    patient_folder_path = fullfile(RAW_DATA_ROOT, patient_folders(i).name);
    raw_name = patient_folders(i).name;              % e.g. "1. JH"
    parts = regexp(raw_name, '(\d+)\.\s*(\w+)', 'tokens', 'once');
    if ~isempty(parts)
        patient_num  = parts{1};                     % "1"
        patient_init = parts{2};                     % "JH"
        patient_id   = sprintf('%s_%s', patient_init, patient_num);  % "JH_1"
    else
        patient_id   = raw_name; % fallback, in case pattern doesn’t match
    end

    fprintf('--> [%d/%d] Starting processing for patient: %s\n', i, numel(patient_folders), patient_id);

    try
        eeg_files_struct = dir(fullfile(patient_folder_path, '*.EEG'));
        if isempty(eeg_files_struct)
            fprintf('    No .EEG files found for %s. Skipping.\n', patient_id);
            fprintf('------------------------------------------------------------\n');
            continue;
        end

        eeg_filenames = {eeg_files_struct.name};
        num_files     = numel(eeg_filenames);
        eegStruct     = struct('Filename', {}, 'Data', {}, 'Labels', {});

        % Process each .EEG
        for j = 1:num_files
            current_filename = eeg_filenames{j};
            eeg_file_path    = fullfile(patient_folder_path, current_filename);
            fprintf('    -> Processing file %d of %d: %s\n', j, num_files, current_filename);

            try
                addpath(BIOSIG_PATH);
                warning('off', 'SOPEN:overflow');
                warning('off', 'BioSig:sopen:overflow');

                hdr  = sopen(eeg_file_path, 'r');
                data = sread(hdr);
                hdr  = sclose(hdr);

                rmpath(BIOSIG_PATH);

                eegStruct(j).Filename = current_filename;
                eegStruct(j).Data     = data.';    % transpose to [channels x samples]
                eegStruct(j).Labels   = hdr.Label;

                clear data;
            catch ME_file
                fprintf('    ERROR on file %s: %s\n', current_filename, ME_file.message);
                if exist(BIOSIG_PATH, 'dir'), rmpath(BIOSIG_PATH); end
            end
        end

        % Save per-patient MAT
        if ~isempty(eegStruct)
            out_file = fullfile(EEG2MAT_DIR, [patient_id, '.mat']);
            fprintf('    SAVING combined data for %s to: %s\n', patient_id, out_file);
            save(out_file, 'eegStruct', '-v7.3');
        else
            fprintf('    No files were processed successfully for %s.\n', patient_id);
        end

    catch ME_patient
        fprintf('  FATAL while processing patient %s: %s\n', patient_id, ME_patient.message);
        fprintf('  Continuing to next patient...\n');
    end

    fprintf('--> [%d/%d] Finished processing for patient: %s\n', i, numel(patient_folders), patient_id);
    fprintf('------------------------------------------------------------\n');
end
seq_total_time = toc;
fprintf('\n============================================================\n');
fprintf('EEG -> MAT stage complete. Total elapsed time: %.2f minutes.\n', seq_total_time/60);
fprintf('============================================================\n\n');


%% =========================
%% --- PARALLEL: Preprocess & Export Spectrograms ---
%% =========================
format long g;
% ----------------------------
% CONFIG
% ----------------------------
func_path = FUNC_PATH;                 % from top
addpath(func_path);                    % add on client first

data_dir         = EEG2MAT_DIR;        % <-- use Stage 1 outputs
spectrogram_root = SPECTROGRAM_DIR;    % <-- final spectrogram destination

time_windows = sort([30, 60, 120, 150, 180], 'descend'); % minutes
fs = 200;
verbose = false;
eeg_channels = 1:19;    % electrodes to process
plt = 0;                % disable plotting on workers

% Filters
notchFreqs   = [20, 60, 80];
Q            = 75;
hp_cutoff    = 0.5;
filter_order = 4;

% ----------------------------
% DISCOVER FILES
% ----------------------------
mat_files = dir(fullfile(data_dir, '*.mat'));
if isempty(mat_files)
    error('No .mat files found in: %s', data_dir);
end

if ~exist(spectrogram_root, 'dir'), mkdir(spectrogram_root); end
ts_run = char(datetime('now','Format','yyyyMMdd_HHmmss'));
summary_log_path = fullfile(spectrogram_root, sprintf('processing_summary_%s.txt', ts_run));

% ----------------------------
% BUILD FLATTENED JOB LIST (file, dataset, window)
% ----------------------------
jobs = struct('file_idx',{},'dataset_idx',{},'window',{},'base','');
for f = 1:numel(mat_files)
    file_path = fullfile(mat_files(f).folder, mat_files(f).name);
    [~, base_filename, ~] = fileparts(mat_files(f).name);

    n_datasets = [];
    try
        info = whos('-file', file_path, 'eegStruct');
        if ~isempty(info), n_datasets = max(1, prod(info.size)); end
    catch
    end
    if isempty(n_datasets)
        tmp = load(file_path, 'eegStruct');
        n_datasets = numel(tmp.eegStruct);
        clear tmp;
    end

    for d = 1:n_datasets
        for w = 1:numel(time_windows)
            jobs(end+1).file_idx     = f;           
            jobs(end  ).dataset_idx  = d;
            jobs(end  ).window       = time_windows(w);
            jobs(end  ).base         = base_filename;
        end
    end
end
numJobs = numel(jobs);
fprintf('Discovered %d jobs across %d files.\n', numJobs, numel(mat_files));

% ----------------------------
% DEFINE UNIFORM DATASET ENTRY TEMPLATE (prevents vertcat mismatches)
% ----------------------------
dataset_template = struct( ...
    'id',             '', ...
    'patient_id',     '', ...
    'dataset_index',  NaN, ...
    'source_file',    '', ...
    'window_duration',NaN, ...
    'skipped',        false, ...
    'skip_reason',    '', ...
    'error_message',  '', ...
    'logStruct',      struct() );

% ----------------------------
% OUTPUT COLLECTORS (parfor-sliced)
% ----------------------------
processed_files = cell(numJobs,1);
dataset_entries = repmat({dataset_template}, numJobs, 1);  % prefill every cell with template
status_lines    = cell(numJobs,1);

% ----------------------------
% START/REUSE POOL + BROADCAST PATH TO WORKERS
% ----------------------------
p = gcp('nocreate');
if isempty(p), p = parpool('local'); end 
pctRunOnAll addpath(func_path);   

% ----------------------------
% PARFOR
% ----------------------------
parfor j = 1:numJobs
    job = jobs(j);
    fprintf('Worker %d\n', getCurrentTask().ID)
    % Resolve file path & names (broadcast data)
    mf = mat_files(job.file_idx);
    file_path = fullfile(mf.folder, mf.name);
    base_filename   = job.base;
    dataset_idx     = job.dataset_idx;
    window_duration = job.window;

    % Paths for outputs/logs
    folder_name = sprintf('%s_%d', base_filename, dataset_idx);
    current_parent_dir = fullfile(spectrogram_root, sprintf('%dmin', window_duration));
    log_folder = fullfile(current_parent_dir, folder_name);
    if ~exist(current_parent_dir, 'dir'), mkdir(current_parent_dir); end
    if ~exist(log_folder, 'dir'), mkdir(log_folder); end
    log_file = fullfile(log_folder, sprintf('%s_%dmin_log.txt', folder_name, window_duration));
    fid = fopen(log_file, 'a');
    hasLog = fid >= 0;
    if hasLog
        fprintf(fid, '=== %s | %s | dataset %d | %dmin ===\n', ...
            datestr(now,'yyyy-mm-dd HH:MM:SS'), base_filename, dataset_idx, window_duration);
    end

    % ---- Pre-initialize locals (no 'clear' in parfor) ----
    S = struct(); eegStruct = struct(); dat = []; labels = [];
    eeg_data = []; eeg_data2 = []; nf_eeg_data = []; filtered_eeg_data = []; rreeg_data = []; tw_eeg_data = [];
    selected_labels = []; num_electrodes = 0; electrode_idx = 1; ch = [];
    current_step = 'INIT';

    % Per-task log struct (meta + step placeholders)
    logStruct = struct();
    logStruct.meta = struct('project_name','Delirium EEG Preprocessing', ...
                            'fs_Hz',fs, ...
                            'time_window_min',window_duration, ...
                            'dataset_index',dataset_idx, ...
                            'source_file',mf.name, ...
                            'patient_id',base_filename, ...
                            'timestamp_start',char(datetime('now','Format','yyyyMMdd_HHmmss')));
    logStruct.steps = struct();

    % Prepare a local entry from the uniform template up-front
    entry = dataset_template;
    entry.id              = sprintf('%s_%dmin', folder_name, window_duration);
    entry.patient_id      = base_filename;
    entry.dataset_index   = dataset_idx;
    entry.source_file     = mf.name;
    entry.window_duration = window_duration;

    try
        % LOAD
        current_step = 'LOAD';
        S = load(file_path, 'eegStruct');  % transparent
        eegStruct = S.eegStruct;
        dat = eegStruct(dataset_idx).Data;
        labels = eegStruct(dataset_idx).Labels;

        % SELECT_CHANNELS (no 'clear'; use local alias)
        current_step = 'SELECT_CHANNELS';
        ch = eeg_channels;
        eeg_data = dat(ch, :);
        num_electrodes = size(eeg_data, 1);
        selected_labels = labels(ch);
        electrode_idx = min(max(1, ch(1)), num_electrodes);

        % REMOVE_NAN
        current_step = 'REMOVE_NAN';
        [eeg_data2, nanFlag, logStruct] = removeNaN2(eeg_data, logStruct, verbose);
        if nanFlag
            if hasLog, fprintf(fid, 'Skip: too many NaNs after removeNaN2.\n'); end
            entry.skipped       = true;
            entry.skip_reason   = 'Too many NaNs';
            entry.error_message = '';
            logStruct.meta.timestamp_end = char(datetime('now','Format','yyyyMMdd_HHmmss'));
            entry.logStruct = logStruct;
            dataset_entries{j} = entry;
            status_lines{j} = sprintf('%s dataset %d %dmin - skipped (NaNs)', base_filename, dataset_idx, window_duration);
            if hasLog, fclose(fid); end
            continue
        end

        % NOTCH
        current_step = 'NOTCH';
        [nf_eeg_data, ~, ~, ~, logStruct] = applyNotchFilter(eeg_data2, fs, notchFreqs, Q, electrode_idx, plt, logStruct, verbose);

        % HIGHPASS
        current_step = 'HIGHPASS';
        [filtered_eeg_data, ~, ~, ~, logStruct] = applyButter(nf_eeg_data, fs, hp_cutoff, filter_order, electrode_idx, plt, logStruct, verbose);

        % REREF
        current_step = 'REREF';
        [rreeg_data, logStruct] = reReference(filtered_eeg_data, logStruct, verbose);

        % PSD
        current_step = 'PSD';
        [~, ~, ~, logStruct] = CaP_PSD(eeg_data2, rreeg_data, fs, electrode_idx, plt, logStruct, verbose);

        % TIMEWINDOW
        current_step = 'TIMEWINDOW';
        [tw_eeg_data, skipFlag, logStruct] = timeWindowEEG2(rreeg_data, fs, window_duration, logStruct, verbose);
        if skipFlag
            if hasLog, fprintf(fid, 'Skip: too short for %d-min window.\n', window_duration); end
            entry.skipped       = true;
            entry.skip_reason   = 'Too short for window';
            entry.error_message = '';
            logStruct.meta.timestamp_end = char(datetime('now','Format','yyyyMMdd_HHmmss'));
            entry.logStruct = logStruct;
            dataset_entries{j} = entry;
            status_lines{j} = sprintf('%s dataset %d %dmin - skipped (too short)', base_filename, dataset_idx, window_duration);
            if hasLog, fclose(fid); end
            continue
        end

        % SAVE_SPECTROGRAMS
        current_step = 'SAVE_SPECTROGRAMS';
        logStruct = saveAllElectrodeSpectrograms5( ...
                        tw_eeg_data, fs, eeg_channels, selected_labels, ...
                        current_parent_dir, folder_name, window_duration, logStruct, verbose);
        if hasLog, fprintf(fid, 'Success: spectrograms saved.\n'); end

        % Success record
        processed_files{j} = entry.id;
        status_lines{j}    = sprintf('%s dataset %d %dmin - success', base_filename, dataset_idx, window_duration);

        logStruct.meta.timestamp_end = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        entry.skipped       = false;
        entry.skip_reason   = '';
        entry.error_message = '';
        entry.logStruct     = logStruct;
        dataset_entries{j}  = entry;

        if hasLog, fclose(fid); end

    catch ME
        if hasLog
            fprintf(fid, 'FATAL at step %s: %s\n', current_step, ME.message);
            if ~isempty(ME.stack)
                st = ME.stack(1);
                fprintf(fid, '  at %s (line %d)\n', st.name, st.line);
            end
            fclose(fid);
        end
        status_lines{j} = sprintf('%s dataset %d %dmin - ERROR at %s: %s', ...
                                  base_filename, dataset_idx, window_duration, current_step, ME.message);

        logStruct.meta.timestamp_end = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        entry.skipped       = true;
        entry.skip_reason   = '';
        entry.error_message = sprintf('ERROR at %s: %s', current_step, ME.message);
        entry.logStruct     = logStruct;
        dataset_entries{j}  = entry;
    end
end

% ----------------------------
% ASSEMBLE MASTER_LOG 
% ----------------------------
MASTER_LOG = struct();
MASTER_LOG.meta = struct('project_name','Delirium EEG Preprocessing', ...
                         'date_created',ts_run, ...
                         'fs_Hz',fs, ...
                         'time_windows',time_windows, ...
                         'notes','Parallel run: unit = dataset × window');

% Every cell has a struct with the same fields (by template), so this is safe:
MASTER_LOG.datasets = vertcat(dataset_entries{:});

save(fullfile(spectrogram_root, sprintf('MASTER_LOG_%s.mat', ts_run)), 'MASTER_LOG');

% Write summary file
try
    fidS = fopen(summary_log_path, 'w');
    fprintf(fidS, 'Run: %s\n', ts_run);
    fprintf(fidS, 'Jobs: %d\n', numJobs);
    fprintf(fidS, '--- Status ---\n');
    for k = 1:numJobs
        if ~isempty(status_lines{k}), fprintf(fidS, '%s\n', status_lines{k}); end
    end
    fclose(fidS);
catch
    warning('Could not write summary log: %s', summary_log_path);
end

% Console summary
fprintf('\nAll processed dataset×window tasks (successes only):\n');
succ = processed_files(~cellfun(@isempty, processed_files));
if ~isempty(succ), fprintf('%s\n', succ{:}); else, fprintf('(none)\n'); end

toc;
