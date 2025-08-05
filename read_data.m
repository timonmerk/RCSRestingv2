function matPaths = findEphysMatFiles(rootPath)
    % Initialize result cell array
    out_path = '/Volumes/timonmerk/restingstate_data';
    matPaths = {};

    % Recursively search for directories named 'resting-state1' or 'resting-state2'
    dirInfo = dir(fullfile(rootPath, '**', 'resting-state*'));

    % Filter only directories
    restingDirs = dirInfo([dirInfo.isdir]);

    % Loop through each matching resting-state directory
    % for i = 1:length(restingDirs)
    for i = length(restingDirs):-1:1
        currentDir = fullfile(restingDirs(i).folder, restingDirs(i).name);
        % Get .mat files containing 'ephys' in the name
        matFiles = dir(fullfile(currentDir, '*lfp*.mat'));
        
        for j = 1:length(matFiles)
            name_ = matFiles(j).name(1:end-4);
            dataPath = fullfile(matFiles(j).folder, matFiles(j).name);

            % Anticipate both hemisphere outputs (left, right, etc.)
            hemispheres = {'left', 'right', 'NA'};  % Optional — adapt based on naming convention
            skip = false;
            
            for h = 1:length(hemispheres)
                suffix = ['_' hemispheres{h}];
                table_file = fullfile(out_path, [name_ suffix '.csv']);
                % stim_file  = fullfile(out_path, [name_ '_stim' suffix '.csv']);
            
                % If either file does not exist → we must process
                if exist(table_file, 'file')
                    skip = true;
                    break;
                end
            end
            
            if skip
                fprintf('Skipping %s — all output files already exist.\n', name_);
                continue;
            end
            % Get file size in bytes
            fileInfo = dir(dataPath);
            fileSizeMB = fileInfo.bytes / (1024 * 1024);  % Convert to MB
            
            fprintf('Loading %s (%.2f MB)\n', dataPath, fileSizeMB);
            loadedVars = load(dataPath);
            
            % Normalize variable name
            if isfield(loadedVars, 'lfpData') && ~isfield(loadedVars, 'lfp_data')
                loadedVars.lfp_data = loadedVars.lfpData;
            end
            
            if ~isfield(loadedVars, 'lfp_data')
                continue;  % Skip if no lfp_data
            end
            
            lfp_data = loadedVars.lfp_data;
            n = length(lfp_data);
            
            for hem_idx = 1:n
                hemisphere = lfp_data(hem_idx).hemisphere;
                suffix = ['_' hemisphere];
            
                % --- Combined Data Table ---
                if isfield(lfp_data(hem_idx), 'combinedDataTable')
                    combined_file = fullfile(out_path, [name_ suffix '.csv']);
                    if ~exist(combined_file, 'file')
                        writetable(lfp_data(hem_idx).combinedDataTable, combined_file);
                    else
                        fprintf('Skipping existing file: %s\n', combined_file);
                    end
                end
            
                % --- Simple Stim Log ---
                if isfield(lfp_data(hem_idx), 'simple_stim_log')
                    stim_file = fullfile(out_path, [name_ '_stim' suffix '.csv']);
                    if ~exist(stim_file, 'file')
                        writetable(lfp_data(hem_idx).simple_stim_log, stim_file);
                    else
                        fprintf('Skipping existing file: %s\n', stim_file);
                    end
                end
            end
        end
    end
end

patients = {'aDBS003', 'aDBS004', 'aDBS005', 'aDBS007', ...
            'aDBS008', 'aDBS009', 'aDBS010', 'aDBS011', 'aDBS012'};

for i = 1:length(patients)
    patient = patients{i};
    root = fullfile('/Volumes/datalake/aDBS-49155/preprocessed-new', patient);
    matEphysFiles = findEphysMatFiles(root);
    
end

% load('/Volumes/datalake/aDBS-49155/preprocessed-new/aDBS010/2022-04-06/resting-state2/aDBS010_resting-state2_20220406160434444445_synced_behav_v3.mat');
% table_1 = lfpData(1).combinedDataTable;
% table_2 = lfpData(2).combinedDataTable;
% 
% if strcmp(lfpData(1).hemisphere, 'left')
%     writetable(table_1, 'aDBS010_2022-04-06_left.csv');
%     writetable(table_2, 'aDBS010_2022-04-06_right.csv');
%     writetable(lfpData(1).simple_stim_log, 'aDBS010_2022-04-06_stim_left.csv');
%     writetable(lfpData(2).simple_stim_log, 'aDBS010_2022-04-06_stim_right.csv');
% else
%     writetable(table_1, 'aDBS010_2022-04-06_right.csv');
%     writetable(table_2, 'aDBS010_2022-04-06_left.csv');
% end