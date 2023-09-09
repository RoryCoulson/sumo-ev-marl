import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# update folder path to the experiment you want to plot, e.g.:
OUTPUTS_FOLDER_PATH = '../experiments/outputs/2_station_strip'

mean_total_times = {}
station_stats = {}

plots = [("step", "system_total_waiting_time"),
         ("step", "system_mean_waiting_time"),
         ("step", "cumulative_penalties"),
         ("step", "cumulative_total_wait_time"),
         ]
x_label = "Episodes"
labels = [(x_label, "Total wait time"),
          (x_label, "Mean wait time"),
          (x_label, "Cumulative\n penalties"),
          (x_label, "Cumulative\n wait time"),
          ]

fig, axs = plt.subplots(len(plots))
fig.suptitle('Training plots')

files = []
folder_names = [folder[0] for folder in os.walk(OUTPUTS_FOLDER_PATH)][1:]
for folder_name in folder_names:
    algorithm = folder_name.split('/')[-1]
    file = f"{folder_name}/{algorithm}_total_metrics_0.csv"
    files.append(file)

colours = ['red', 'blue', 'green', 'yellow']
# plot wait time results
for plot_id, (plot_x, plot_y) in enumerate(plots):

    for i, file in enumerate(files):
        df = pd.read_csv(file, sep=",")
        file_label = str(file.split('/')[-2])
        axs[plot_id].plot(df[plot_x], df[plot_y], colours[i],
                          label=file_label, alpha=0.5)
        axs[plot_id].set(xlabel=labels[plot_id][0],
                         ylabel=labels[plot_id][1])
        axs[plot_id].legend(loc="upper right")
        axs[plot_id].grid(True)
plt.show()
plt.close()

# plot mean wait time results
for file, folder_name in zip(files, folder_names):
    name = folder_name.split('/')[-1]
    df = pd.read_csv(file, sep=",")
    num_stations = len([col for col in df.columns if col.endswith('_cumulative_diff_rewards')])
    df_last_ep = df.tail(5000)  # for experiments with ep len 5,000

    mean_total_wait_time = np.mean(df_last_ep['system_total_waiting_time'])
    print(f"Mean total wait time ({name}): {mean_total_wait_time}")
    mean_total_times[name] = mean_total_wait_time

    station_wait_times = []
    for cs_index in range(1, num_stations+1):
        cs_wait_time = np.sum(
            df_last_ep[f'{cs_index}_accumulated_waiting_time'])
        station_wait_times.append(cs_wait_time)

    station_stats[name] = station_wait_times

df_mean_total_wait = pd.DataFrame(mean_total_times, index=[''])
df_mean_total_wait.plot.bar(rot=0, title="Mean wait time", color=colours)

print('Station stats:', station_stats)
df_station_stats = pd.DataFrame(station_stats)
df_station_stats.plot.bar(
    rot=0, title="Mean wait time per station", color=colours)

plt.show(block=True)
