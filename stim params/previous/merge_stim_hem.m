load("StimParams.mat");

patients = fieldnames(StimParams);

% Initialize empty table for all patients
T_all = table();

for p = 1:numel(patients)
    patID = patients{p};  % e.g., 'x004'
    patientStruct = StimParams.(patID);

    % Iterate through each date/session for this patient
    dateFields = fieldnames(patientStruct);
    for i = 1:numel(dateFields)
        thisDate = dateFields{i};
        sessionStruct = patientStruct.(thisDate);

        % Iterate through hemispheres
        hemiFields = fieldnames(sessionStruct);
        for j = 1:numel(hemiFields)
            hemiName = hemiFields{j};
            T = sessionStruct.(hemiName);

            % Convert to string-safe table
            T_string = varfun(@string, T);

            % Add metadata
            T_string.PatientID   = repmat(string(patID), height(T_string), 1);
            T_string.SessionDate = repmat(string(thisslDate), height(T_string), 1);
            T_string.Hemisphere  = repmat(string(hemiName), height(T_string), 1);

            % Append
            T_all = [T_all; T_string];
        end
    end
end

% Reorder columns for clarity
T_all = movevars(T_all, {'PatientID','SessionDate','Hemisphere'}, 'Before', 1);

% Save as CSV
filename = ['StimParams_AllPatients_' datestr(now, 'yyyymmdd_HHMM') '.csv'];
writetable(T_all, filename);

fprintf('✅ Combined CSV saved as: %s\n', filename);