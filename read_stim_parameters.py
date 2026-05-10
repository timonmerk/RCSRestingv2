import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#df_params_merged = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/stim params/StimParams_AllPatients_20251124_0831.csv")
df_params_merged = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/stim params/StimParams_AllPatients_20260106_1736.csv")

stim_split = df_params_merged['string_stimParams'].str.split(',', expand=True)

# Assign column names
stim_split.columns = ['Contact', 'Amplitude_mA', 'PulseWidth_us', 'Frequency_Hz']

def apply_contact_split(contact):
    anode = None
    cathode = None

    if type(contact) != float and contact != '' and contact != "Disabled":
        anode = contact[0]
        cathode = contact[2]

    return anode, cathode

# Apply the function to split Contact column
stim_split[['Anode+', 'Cathode-']] = stim_split['Contact'].apply(apply_contact_split).apply(pd.Series)
stim_split = stim_split.apply(lambda x: x.str.strip())
stim_split['Amplitude_mA'] = stim_split['Amplitude_mA'].str.replace('mA', '', regex=False).astype(float)
stim_split['PulseWidth_us'] = stim_split['PulseWidth_us'].str.replace('us', '', regex=False).astype(float)
stim_split['Frequency_Hz'] = stim_split['Frequency_Hz'].str.replace('Hz', '', regex=False).astype(float)
stim_split['Contact'] 

# Merge back with original DataFrame
df_params_merged = pd.concat([df_params_merged, stim_split], axis=1)

# I expect only unique BheaviorStartUnix values
behav_starts = df_params_merged['BehavStartUnix'].unique().tolist()
# for each behav_starts, there are multiple rows with stim_parameters; select the earliest one
df_params_merged['SessDate'] = pd.to_datetime(df_params_merged['SessionDate'].str.replace('d', '').str.replace('_', '-'))
df_params_merged["PatientID"] = df_params_merged["PatientID"].apply(lambda x: int(x[1:]))
df_params_merged["exact_time"] = pd.to_datetime(df_params_merged["string_HostUnixTime"], unit='ms')

stim_params_list = []
missed_list = []
for ts in behav_starts:
    for hem in ['left', 'right']:
        ts_dt = pd.to_datetime(ts, unit='ms')
        df_subset = df_params_merged[df_params_merged['BehavStartUnix'] == ts]
        df_subset_hem = df_subset[df_subset['Hemisphere'] == hem]

        df_earlier = df_subset_hem[df_subset_hem['exact_time'] <= ts_dt]
        if df_earlier.empty:
            missed_list.append(ts)
            continue

        df_stim_params = df_earlier.sort_values(by='exact_time').tail(1)
        time_diff = ts_dt - df_stim_params['exact_time'].values[0]
        df_stim_params["time_diff"] = time_diff
        df_stim_params["timestamp_behavior"] = ts
        if time_diff < pd.Timedelta(minutes=120):
            stim_params_list.append(df_stim_params)
        else:
            missed_list.append(ts)

# this can be adapted, for each bheavior timestamp I practically just
# need the most recent stim settings

df_stim_params_final = pd.concat(stim_params_list).reset_index(drop=True)
# if string_therapyStatus is 0, then set Amplitude_mA, PulseWidth_us, Frequency_Hz to 0
df_stim_params_final.loc[df_stim_params_final['string_therapyStatus'] == 0, ['Amplitude_mA', 'PulseWidth_us', 'Frequency_Hz']] = 0
df_stim_params_final.to_csv("stim params/StimParams_Processed_all_times_2026.csv", index=False)

# so, now plot the params for left and right hemisphere over time for each patient
df_use = df_stim_params_final.copy()
df_use["TEED"] = df_use["Amplitude_mA"] * df_use["PulseWidth_us"] * df_use["Frequency_Hz"] / 1000  # in uW
df_use["Days_Since_First_Session"] = (df_use["SessDate"] - df_use.groupby("PatientID")["SessDate"].transform('min')).dt.days
# remove ind with 0 Amplitude (DBS off)
df_use = df_use[df_use["Amplitude_mA"] > 0]

# 8 * 4 = 32
plt.figure(figsize=(13, 9))
for i, sub in enumerate(df_use['PatientID'].unique()):
    
    for j, col in enumerate(["Amplitude_mA", "PulseWidth_us", "Frequency_Hz", "TEED"]):
        plt.subplot(8, 4, i * 4 + j + 1)
        for hem in ['left', 'right']:
            df_sub_hem = df_use[(df_use['PatientID'] == sub) & (df_use['Hemisphere'] == hem)]
            plt.plot(df_sub_hem['Days_Since_First_Session'], df_sub_hem[col], marker='o', label=f'{hem}', markersize=3)
        # remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if j == 0:
            plt.ylabel(f'{sub}')
            
        else:
            plt.ylabel('')
            # remove x ticks and 
            #plt.gca().set_xticklabels([])
        if i == 0:
            plt.title(f"{col}")
        plt.xlabel('Days Since First Session')
        if i != len(df_use['PatientID'].unique()) - 1:
            plt.gca().set_xticklabels([])
            plt.xlabel('')
        if i == 0 and j == 0:
            plt.legend(title='Hemisphere')

#plt.tight_layout()
plt.savefig("figures/StimParams_OverTime_all_patients.pdf")
    

        


pdf_path = '/Users/Timon/Documents/Houston/resting_state_OCD/stim params/StimParams_OverTime_per_patient.pdf'
patients = df_use['PatientID'].unique()
with PdfPages(pdf_path) as pdf:
    for metric in ["Amplitude_mA", "PulseWidth_us", "Frequency_Hz"]:
        plt.figure(figsize=(18, 6))
        for hem in ["left", "right"]:
            df_hem = df_use[df_use['Hemisphere'] == hem]
            plt.subplot(1, 2, 1 if hem == "left" else 2)
            for patient in patients:
                patient_data = df_hem[df_hem['PatientID'] == patient]
                plt.plot(patient_data['SessDate'], patient_data[metric], marker='o', label=f'Patient {patient}')
            plt.xlabel('Session Date')
            plt.ylabel(metric)
            plt.title(f'{metric} {hem}')
            plt.legend()
        pdf.savefig()
        plt.close()


# ok, now check for each patient and each date if there are multiple entries
# set the earlier entry to rs1 and the later to rs2
df_stim_params_final = df_stim_params_final.sort_values(by=['PatientID', 'SessDate', 'exact_time']).reset_index(drop=True)
df_stim_params_final['rs_id'] = ''
hits = []
for patient in df_stim_params_final['PatientID'].unique():
    df_patient = df_stim_params_final[df_stim_params_final['PatientID'] == patient]
    df_patient = df_patient.query("Hemisphere == 'left'")  # only need to check one hemisphere
    for sess_date in df_patient['SessDate'].unique():
        df_sess = df_patient[df_patient['SessDate'] == sess_date]
        if len(df_sess) == 1:
            print(f"Only one entry for patient {patient} on date {sess_date}")
        elif len(df_sess) == 2:
            df_earlier = df_sess.sort_values(by='exact_time').head(1)
            df_later = df_sess.sort_values(by='exact_time').tail(1)
            if (df_earlier["sessName"].iloc[0] == "xe1" or df_earlier["sessName"].iloc[0] == "xe")  and df_later["sessName"].iloc[0] == "xe2":
                hits.append((patient, sess_date))
            else:
                print(f"Unexpected sessName for patient {patient} on date {sess_date}")
        else:
            print(f"More than 2 entries for patient {patient} on date {sess_date}")

# need to check now the filenames

# df_stim_params_final[["PatientID", "SessionDate", "rs_id", "sessName"]]

# ok, 'xe' ist eigentlich manchmal eigentlich 'xe1'
# es kann aber auch sein, dass eins von denen DBS off ist!
# obwohl ich das rauskriege, die sind auch immer null im string_therapyStatus
# ok, das macht bisher egtl alles Sinn...
# frage ist jetzt nur, wie kann ich die mappen.. eigentlich nur über den 
# namen der Session und den string Namen im file
# damit habe ich dann pre- and post DBS setting changes
# plus left and right
# plus DBS off.. das ist super wichtig... 

# pdf_path = '/Users/Timon/Documents/Houston/resting_state_OCD/stim params/TEED_over_time_per_patient.pdf'
# patients = df_use['PatientID'].unique()
# with PdfPages(pdf_path) as pdf:
#     for metric in ["TEED_1kOhmImpedance", "Amplitude_mA", "PulseWidth_us", "Frequency_Hz"]:
#         plt.figure(figsize=(18, 6))
#         for hem in ["left", "right"]:
#             df_hem = df_use[df_use['Hemisphere'] == hem]
#             plt.subplot(1, 2, 1 if hem == "left" else 2)
#             for patient in patients:
#                 patient_data = df_hem[df_hem['PatientID'] == patient]
#                 plt.plot(patient_data['SessDate'], patient_data[metric], marker='o', label=f'Patient {patient}')
#             plt.xlabel('Session Date')
#             plt.ylabel(metric)
#             plt.title(f'{metric} Over Time for Each Patient')
#             plt.legend()
#         pdf.savefig()
#         plt.close()



#############

# df_behavior = pd.read_csv("/Users/Timon/Documents/Houston/resting_state_OCD/stim params/previous/all_behavior_timestamps.csv")
# df_behavior["patient_id"] = df_behavior["filename"].apply(lambda x: x.split("/")[5])  # remove leading 'P'
# df_behavior["timestamp_date"] = pd.to_datetime(df_behavior["timestamp_unix"], unit='ms')
# # get unique year month day combinations
# df_behavior["date_only"] = df_behavior["timestamp_date"].dt.date

# # get unique timestamp_unix values per date for patient aDBS009
# df_behavior_aDBS009 = df_behavior[df_behavior["patient_id"] == "aDBS009"]
# unique_timestamps_aDBS009 = df_behavior_aDBS009.groupby("date_only")["timestamp_unix"].apply(list).reset_index()

