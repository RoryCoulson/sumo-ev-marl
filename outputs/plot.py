import argparse
import glob
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# python plot.py -f data/

if __name__ == "__main__":

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")

    args = prs.parse_args()

    plots = [("step", "system_total_waiting_time"),
             ("step", "system_mean_waiting_time"),
             ("step", "cumulative_penalties"),
             ("step", "cumulative_total_wait_time"),
             #  ("step", "mean_reward"),
             #  ("step", "mean_cumulative_reward"),
             #  ("step", "cumulative_rewards"),
             ]

    x_label = "Episodes"
    labels = [(x_label, "Total wait time"),
              (x_label, "Mean wait time"),
              (x_label, "Cumulative\n penalties"),
              (x_label, "Cumulative\n wait time"),
              #   (x_label, "Mean reward"),
              #   (x_label, "Cumulative\n mean reward"),
              #   (x_label, "Cumulative\n rewards"),
              ]

    fig, axs = plt.subplots(len(plots))
    fig.suptitle('Training plots')

    for plot_id, (plot_x, plot_y) in enumerate(plots):

        colours = ['r-', 'b-', 'g-', 'y-']  # add more if more

        # File reading and grouping
        i = 0
        for file in args.f:
            for f in sorted(glob.glob(file + "*"), reverse=True):
                df = pd.read_csv(f, sep=",")

                file_label = str(f.split('/')[1].split('.')[0])
                axs[plot_id].plot(df[plot_x], df[plot_y], colours[i],
                                  label=file_label, alpha=0.5)

                # axs[plot_id].set_title(plot_y)
                axs[plot_id].set(xlabel=labels[plot_id][0],
                                 ylabel=labels[plot_id][1])
                axs[plot_id].legend(loc="upper right")
                axs[plot_id].grid(True)
                i += 1

    plt.show()
