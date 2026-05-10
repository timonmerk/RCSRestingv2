addpath('D:\Libraries\rcsanalysis\matlab')
addpath('D:\Libraries\Analysis-rcs-data-master-May2021\Analysis-rcs-data-master\code')

pts={'aDBS009'};
devices={'DeviceNPC700500H'};

for p=1
    d=dir(strcat('D:\OCD-Patient-Data\',pts{p}));
    d=d(3:end);
    names=string({d.name});
    d=d(startsWith(names,'2'));
    for date=1:length(d)
       disp(d(date).name)
       preprocDir=strcat('D:\OCD-Preprocessed-Data\',pts{p},'\',d(date).name);
       iPath=strcat(preprocDir,'\impedances.mat');
        if ~exist(iPath,'file')
            lfpBasePath=strcat(d(date).folder,'\',d(date).name,'\LFP');
            if exist(lfpBasePath,'dir')
                lfpFolders=dir(lfpBasePath);
                lfpFolders=lfpFolders(3:end);
                lfpNames=string({lfpFolders.name});
                lfpFolders=lfpFolders(startsWith(lfpNames,'Session'));
                eventTables=cell(length(lfpFolders),1);
                foundSettings=false;
                for l=1:length(lfpFolders)
                    lfpPath=strcat(lfpBasePath,'\',lfpFolders(l).name,'\',devices{p});
                    if exist(strcat(lfpPath,'\DeviceSettings.json'),'file') %contains(lfpPath, '500H')
                        if ~foundSettings && exist(strcat(lfpPath,'\DeviceSettings.json'),'file')
                            [tds,ps,fs,met]=createDeviceSettingsTable(lfpPath);
                            [stimSettingsOut, stimMetaData] = createStimSettingsFromDeviceSettings(lfpPath);
                            impedances.tds=tds;
                            impedances.ps=ps;
                            impedances.fs=fs;
                            impedances.met=met;
                            impedances.stimSettingsOut=stimSettingsOut;
                            impedances.stimMetaData=stimMetaData;
                            foundSettings=true;
                        end
                        eventFolder=strcat(lfpPath,'\EventLog.json');
                        eventTables{l}=loadEventLog(eventFolder);
                        disp(lfpFolders(l).name)
                        [lead_integrity_events,lead_integrity_HostUnixTime] = get_lead_integrity_events(eventTables);
                        impedances.values=lead_integrity_events;
                        impedances.times=lead_integrity_HostUnixTime;
                        if ~exist(preprocDir,'dir')
                            mkdir(preprocDir)
                        end
                        save(iPath,'impedances')
                    else
                    end
                end
            end
        end
        disp("Complete")
    end
end