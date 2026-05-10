PATH_ = "/Users/Timon/Documents/Houston/resting_state_OCD/example_table_events.csv"

import json
import pandas as pd
import numpy as np

df = pd.read_csv(PATH_)
df["Unnified Derived Time"] = pd.to_datetime(df["Unified Derived Time"], unit='ms')

#json_path = '/Volumes/datalake/aDBS-49155/aDBS005 Recordings/2020-11-05/task-msit/pid_aDBS005_1604606758464.json'
json_path = '/Volumes/datalake/aDBS-49155/aDBS007 Recordings/2021-02-10/task-msit/pid_aDBS007_1612990638114.json'
with open(json_path, 'r') as f:
    js_data = json.load(f)

df_js = pd.DataFrame(js_data)

df_js["timestamp"] = df_js["time_elapsed"] + df_js["timestamp"].iloc[0]
df_js["timestamp"] = pd.to_datetime(df_js["timestamp"], unit='ms')


from matplotlib import pyplot as plt

first_time_json = df_js["timestamp"].iloc[4]
# get closest time in df
first_time_csv = df["Unnified Derived Time"].iloc[(df["Unnified Derived Time"] - first_time_json).abs().argsort()[:1]].iloc[0]
df_limit = df[df["Unnified Derived Time"] >= first_time_csv]


plt.figure()
plt.subplot(2,1,1)
plt.plot(np.diff(df_js["timestamp"].iloc[4:].values)/1000000, label='pid msit json file')
plt.plot(np.diff(df_limit["Unnified Derived Time"].values)/1000000, label='neural harmonized csv')
plt.xlabel("Sample")
plt.ylabel("Time Difference (ms)")
plt.title("Offset diff between neural harmonized Trial onset and json task onset")
plt.legend()
plt.subplot(2,1,2)
plt.plot(df_js["timestamp"].iloc[4:].values, label='pid msit json file')
plt.plot(df_limit["Unnified Derived Time"].values, label='neural harmonized csv')
plt.xlabel("Sample")
plt.ylabel("Time (ms)")

plt.title("Time neural harmonized Trial onset and json task onset")
plt.legend()
plt.tight_layout()
plt.show()