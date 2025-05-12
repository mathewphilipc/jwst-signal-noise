from src.simulate_lib import simulate_multiaccum
import argparse
import pandas as pd
import numpy as np
from fitramp.fitramp import Covar, fit_ramps
import matplotlib.pyplot as plt
import argparse
from src.simulate_lib import poisson_chisq

parser = argparse.ArgumentParser()

parser.add_argument("--truefreq", 
                    type=int, 
                    required=True, 
                    help="True (simulated) frequency"
)

parser.add_argument("--readnoise",
                    type=int, 
                    required=True, 
                    help="Standard deviation of simulated read noise"
)

parser.add_argument("--num_measurements",
                    type=int, 
                    required=True, 
                    help="Number of measurements up the ramp. Includes" \
                    "measurements at exactly t=0.0 and t=1.0"
)

def fit_brandt_slope_and_chisq(data):
    n = len(data)
    my_covar = Covar([s for s in range(n)])
    diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
    for t in range(1,len(diffs)+1):
        diffs[t-1] = data[t] - data[t-1]
    #sig = 20.1 for JWST images
    ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=float(read_noise), rescale=True)
    slope = ramp_result.countrate[0]
    #intercept = ramp_result.pedestal[0]
    chisq = poisson_chisq(history=data, intercept=0, slope=slope, read_noise=read_noise)
    return [slope, chisq]

args = parser.parse_args()

freq = args.truefreq
read_noise = args.readnoise
num_measurements = args.num_measurements
slope_list = []
chisq_list = []
num_trials = 100
neg_slope_count = 0
#for i in range(num_trials):
#    history = simulate_multiaccum(freq=freq, num_measurements=num_measurements, read_noise=read_noise)
#    slope, chisq = fit_brandt_slope_and_chisq(history)
#    print(slope)
#    slope_list.append(slope)
#    chisq_list.append(chisq)
#    if slope < 0:
#        neg_slope_count += 1
#print(f"Negative slope fraction = {neg_slope_count / num_trials}.")
#plt.hist(slope_list, bins=10)
#plt.show()
#plt.hist(chisq_list, bins=10)
#plt.show()

print("How does chisq vary with num_measurements?")
min_measurements = 5
max_measurements = 100
for variable_num_measurements in range(min_measurements, max_measurements):
    history = simulate_multiaccum(freq=freq, num_measurements=variable_num_measurements, read_noise=read_noise)
    slope, chisq = fit_brandt_slope_and_chisq(history)
    print(chisq)
    chisq_list.append(chisq)

plt.plot(np.arange(min_measurements, max_measurements), chisq_list)
plt.title("chisq versus num_measurements at fixed rate and read noise") 
plt.xlabel("num_measurements") 
plt.ylabel("chisq") 
plt.show()

fig.savefig(
    "data/plots/chisq_vs_num_measurements.png",        # filename; extension sets the format
    dpi=300,               # resolution in dots-per-inch
    bbox_inches="tight"    # trim extra whitespace
)