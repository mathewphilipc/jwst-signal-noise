# Plot heatmap of ratios of expected formal errors (specifically, rms
# deviations of estimated slope from true slope) to empirical spread versus
# read noise and true frequency.

import numpy as np
import matplotlib.pyplot as plt
import json

true_freq_list = [1, 3, 10, 30, 100, 300, 1000]
read_noise_list = [1, 3, 10, 30, 100, 300, 1000]
grid = np.zeros((len(true_freq_list), len(read_noise_list)))
for i in range(len(true_freq_list)):
    freq = true_freq_list[i]
    for j in range(len(read_noise_list)):
        read_noise = read_noise_list[j]
        input_file = "data/poisson_simulations/freq_" + str(freq) + "_read_noise_" + str(read_noise) + "/summary.json"
        # print(f"input_file = {input_file}")
        with open(input_file) as json_data:
            d = json.load(json_data)
            ols_chisq = d["ols_chisq_mean"]
            brandt_chisq = d["brandt_chisq_mean"]
            grid[i][j] = brandt_chisq - ols_chisq

# Plot grid as heat map
fig, ax = plt.subplots()
im = ax.imshow(grid, aspect='auto') 

# 5) Label the axes
ax.set_xticks(np.arange(len(read_noise_list)))
ax.set_xticklabels(read_noise_list)
ax.set_yticks(np.arange(len(true_freq_list)))
ax.set_yticklabels(true_freq_list)
ax.set_xlabel("read noise")
ax.set_ylabel("true rate")
ax.set_title("Difference between Brandt and OLS mean chisq vs true rates and read noises\n(ramp = 101 measurements in 1.0s)")

fig.colorbar(im, ax=ax, label="Mean chisq ratio")

plt.tight_layout()

fig.savefig(
    "data/plots/chisq_difference_heatmap.png",        # filename; extension sets the format
    dpi=300,               # resolution in dots-per-inch
    bbox_inches="tight"    # trim extra whitespace
)

plt.show()

