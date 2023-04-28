import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_stations = 4

greedy_df = pd.read_csv('data/Greedy.csv')
dqn_df = pd.read_csv('data/DQN.csv')
ppo_df = pd.read_csv('data/PPO.csv')
greedy_df_final_episode = greedy_df.loc[greedy_df['step'] >= 124000]
dqn_df_final_episode = dqn_df.loc[dqn_df['step'] >= 124000]
ppo_df_final_episode = ppo_df.loc[ppo_df['step'] >= 124000]

# mean total wait time
mean_greedy_total_wait_time = np.mean(
    greedy_df_final_episode['system_total_waiting_time'])
mean_dqn_total_wait_time = np.mean(
    dqn_df_final_episode['system_total_waiting_time'])
mean_ppo_total_wait_time = np.mean(
    ppo_df_final_episode['system_total_waiting_time'])

print(
    f"mean_greedy_total_wait_time: {mean_greedy_total_wait_time}, mean_dqn_total_wait_time: {mean_dqn_total_wait_time}, mean_ppo_total_wait_time: {mean_ppo_total_wait_time}")

station_stats = {'greedy': [], 'ppo': [], 'dqn': []}
station_stats_labels = []
for cs_index in range(1, num_stations+1):
    station_stats_labels.append(f'Station {cs_index}')

    station_stats['greedy'].append(
        np.sum(greedy_df_final_episode[f'{cs_index}_accumulated_waiting_time']))
    station_stats['dqn'].append(
        np.sum(dqn_df_final_episode[f'{cs_index}_accumulated_waiting_time']))
    station_stats['ppo'].append(
        np.sum(ppo_df_final_episode[f'{cs_index}_accumulated_waiting_time']))


greedy = [mean_greedy_total_wait_time]
mappo = [mean_ppo_total_wait_time]
madqn = [mean_dqn_total_wait_time]

index = ['']
df = pd.DataFrame({'greedy': greedy, 'mappo': mappo,
                   'madqn': madqn}, index=index)


df.plot.bar(rot=0, title="Mean wait time")


index = station_stats_labels
df2 = pd.DataFrame({'Greedy': station_stats['greedy'], 'PPO': station_stats['ppo'],
                   'DQN': station_stats['dqn']}, index=index)


df2.plot.bar(rot=0, title="Total wait time per station")
plt.show(block=True)
