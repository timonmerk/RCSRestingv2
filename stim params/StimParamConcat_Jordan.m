

for i = 1:8
    if i ==1
        load('files004.mat');
        pat = "x004";
    end
    if i == 2
        load('files005.mat');
        pat = "x005";
    end
    if i == 3
        load('files007.mat');
        pat = "x007";
    end
    if i==4
        load('files008.mat');
        pat = "x008";
    end
    if i==5
        load('files009.mat');
        pat = "x009";
    end
    if i==6
        load('files010.mat');
        pat = "x010";
    end
    if i==7
        load('files011.mat');
        pat = "x011";
    end
   
    if i==8
        load('files012.mat');
        pat = "x012";
    end


for j = 1:numel(fileArr)
dateStr = regexp(fileArr{j}, '\d{4}-\d{2}-\d{2}', 'match');
dateStr = "d" + dateStr{1};
dateStr = replace(dateStr,'-','_');
clear lfpData
clear data
load(fileArr{j});
hemiCheck = {lfpData.hemisphere};
type = extractBetween(fileArr{j},'stat','/aDBS0');
type = string(type{1});
type = strcat("x",type);
if contains(type,'-')
    type = replace(type,'-','');
end
if i > 3
if contains(hemiCheck{1},'left')

    leftCheck = 1;
    RightCheck = 2;

else
    leftCheck = 2;
    RightCheck = 1;
end


StimParams.(pat).(dateStr).(type).left = lfpData(leftCheck).stimLogSettings;
StimParams.(pat).(dateStr).(type).right = lfpData(RightCheck).stimLogSettings;
StimParams.(pat).(dateStr).(type).BehavStartUnix = data.behavior.behav_start_timestamp_unix;

else

StimParams.(pat).(dateStr).(type).BothHemi = lfpData.stimLogSettings;
StimParams.(pat).(dateStr).(type).BehavStartUnix = data.behavior.behav_start_timestamp_unix;

end
end


end