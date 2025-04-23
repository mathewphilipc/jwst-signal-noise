import pandas as pd
import numpy as np
from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
import matplotlib.pyplot as plt
import math
import statistics
import argparse
import json
import scipy

true_freq_list = [1, 3, 10, 30, 100, 300, 1000, 10000]
read_noise_list = [1, 3, 10, 30, 100, 300, 1000, 10000]
grid = np.zeros((len(true_freq_list), len(read_noise_list)))
for i in range(len(true_freq_list)):
    freq = true_freq_list[i]
    for j in range(len(read_noise_list)):
        read_noise = read_noise_list[j]
        input_file = "data/poisson_simulations/freq_" + str(freq) + "_read_noise_" + str(read_noise) + "/summary.json"
        print(f"input_file = {input_file}")
        with open(input_file) as json_data:
            d = json.load(json_data)
            print(type(d))
            ratio = d["ols_brandt_spread_ratio"]
            print(f"Ratio = {ratio}")
            grid[i][j] = ratio

# Plot grid as heat map
fig, ax = plt.subplots()
im = ax.imshow(grid, aspect='auto') 

# 5) Label the axes
ax.set_xticks(np.arange(len(read_noise_list)))
ax.set_xticklabels(read_noise_list)
ax.set_yticks(np.arange(len(true_freq_list)))
ax.set_yticklabels(true_freq_list)
ax.set_xlabel("read noise")
ax.set_ylabel("true freq")
ax.set_title("OLS vs Brandt stdev ratio at various true frequencies and read noises\n(ramp = 101 measurements in 1.0s)")

fig.colorbar(im, ax=ax, label="your output metric")

plt.tight_layout()

fig.savefig(
    "data/plots/spread_ratio_heatmap.png",        # filename; extension sets the format
    dpi=300,               # resolution in dots-per-inch
    bbox_inches="tight"    # trim extra whitespace
)

plt.show()

