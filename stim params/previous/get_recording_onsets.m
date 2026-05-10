rootDir = "/Volumes/datalake/aDBS-49155/preprocessed-new";

files = dir(fullfile(rootDir, '**', '*resting-state*'));
% Keep only files (exclude folders)
filesOnly = allEntries(~[allEntries.isdir]);

% (Optional) Get full file paths
filePaths = fullfile({filesOnly.folder}, {filesOnly.name});

% Initialize output struct
allData = struct('filename', {}, 'timestamp_unix', {});

% Loop through each file
for i = 1:length(filePaths)
    filePath = filePaths(i);
    % fprintf('Processing %d/%d: %s\n', i, length(filePaths), filePath);

    try
        % Load file
        S = load(string(filePath));

        % Extract timestamp safely
        if isfield(S, 'data') && isfield(S.data, 'behavior') && ...
           isfield(S.data.behavior, 'behav_start_timestamp_unix')
            ts = S.data.behavior.behav_start_timestamp_unix;
        else
            ts = NaN; % Missing value
        end
    catch ME
        warning('Error loading %s: %s', files(i).name, ME.message);
        ts = NaN;
    end

    % Add to struct
    allData(i).filename = string(filePath);
    allData(i).timestamp_unix = ts;
end

save(fullfile('all_behavior_timestamps.mat'), 'allData');
