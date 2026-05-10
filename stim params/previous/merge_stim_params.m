load("Stimulation_Timon.mat");

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
            % Select from table only HostUnixTime, activeGroup,
            % therapyStatus, stimParams_prog1, stimParams_prog2,
            Tsel = T(:, {'HostUnixTime', 'activeGroup', 'therapyStatus', ...
              'stimParams_prog1', 'stimParams_prog2'});

            T_string = varfun(@string, Tsel);

            % Add metadata
            T_string.PatientID   = repmat(string(patID), height(T_string), 1);
            T_string.SessionDate = repmat(string(thisDate), height(T_string), 1);
            

            if ismember(patID, ["x004", "x005", "x007"])
               hemiName = "left";
               Tadd = T_string;
               Tadd.string_stimParams = Tadd.string_stimParams_prog1;
               Tadd.Hemisphere  = repmat(string(hemiName), height(Tadd), 1);
               Tadd = removevars(Tadd, {'string_stimParams_prog1'});
               Tadd = removevars(Tadd, {'string_stimParams_prog2'});
               T_all = [T_all; Tadd];

               hemiName = "right";
               Tadd = T_string;
               Tadd.Hemisphere  = repmat(string(hemiName), height(Tadd), 1);
               Tadd.string_stimParams = Tadd.string_stimParams_prog2;
               Tadd = removevars(Tadd, {'string_stimParams_prog1'});
               Tadd = removevars(Tadd, {'string_stimParams_prog2'});
               T_all = [T_all; Tadd];
            else
                T_string.Properties.VariableNames{'string_stimParams_prog1'} = 'string_stimParams';
                T_string = removevars(T_string, {'string_stimParams_prog2'});
                T_string.Hemisphere  = repmat(string(hemiName), height(T_string), 1);
                T_all = [T_all; T_string];
            end
        end
    end
end

% Reorder columns for clarity
T_all = movevars(T_all, {'PatientID','SessionDate','Hemisphere'}, 'Before', 1);

% Save as CSV
filename = ['StimParams_AllPatients_' datestr(now, 'yyyymmdd_HHMM') '.csv'];
writetable(T_all, filename);

fprintf('✅ Combined CSV saved as: %s\n', filename);