import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# update
num_stations = 2

prs = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument("-f", nargs="+", required=True, help="Files\n")
args = prs.parse_args()

mean_total_times = {}
station_stats = {}

for file in args.f:
    for f in sorted(glob.glob(file + "*"), reverse=True):
        name = f.split('/')[1].split('.')[0]
        df = pd.read_csv(f, sep=",")
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

colors = ['red', 'blue', 'green', 'yellow']

df_mean_total_wait = pd.DataFrame(mean_total_times, index=[''])
df_mean_total_wait.plot.bar(rot=0, title="Mean wait time", color=colors)

print('station_stats:', station_stats)
df_station_stats = pd.DataFrame(station_stats)
df_station_stats.plot.bar(
    rot=0, title="Mean wait time per station", color=colors)

plt.show(block=True)
