import pandas as pd
import numpy as np
from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
import matplotlib.pyplot as plt
import math
import statistics
import argparse
import json
import scipy
from src.simulate_lib import true_param_covariance, poisson_chisq


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
parser.add_argument("--plot_distributions",
                    action="store_true",
                    help="Optionally plot fitted slope distributions"
)

args = parser.parse_args()

true_freq = args.truefreq
read_noise = args.readnoise
plot_distributions = args.plot_distributions

# Estimate slope naively
def fit_naive_slope(data):
    num_measurements = len(data)
    diff = data[num_measurements - 1] - data[0]
    return diff / (num_measurements - 1)

# Estimate slope by OLS
def fit_ols_slope_and_chisq(data):
    n = len(curr_data)
    X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T
    ols_fit_params = np.linalg.lstsq(X, curr_data, rcond=None)[0] # (2,) nparr containing [slope, intercept]
    slope = float(ols_fit_params[0])
    intercept = float(ols_fit_params[1])
    chisq = poisson_chisq(history=data, intercept=intercept, slope=slope, read_noise=read_noise)
    return [slope, chisq]

# Estimate slope naively
def fit_brandt_slope_and_chisq(data):
    n = len(curr_data)
    my_covar = Covar([s for s in range(n)])
    diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
    for t in range(1,len(diffs)+1):
        diffs[t-1] = curr_data[t] - curr_data[t-1]
    #sig = 20.1 for JWST images
    ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=float(read_noise), rescale=True)
    slope = ramp_result.countrate[0]
    #intercept = ramp_result.pedestal[0]
    chisq = poisson_chisq(history=data, intercept=0, slope=slope, read_noise=read_noise)
    return [slope, chisq]

input_dir = "data/poisson_simulations/freq_" + str(true_freq) + "_read_noise_" + str(read_noise)
input_file = input_dir + "/simulations.csv"
output_file = input_dir + "/summary.json"

df = pd.read_csv(input_file, index_col=0)
num_exps = len(df.columns)
ols_slopes = []
ols_chisqs = []
ols_intercepts = []
brandt_slopes = []
brandt_chisqs = []
naive_slopes = []

for i in range(num_exps):
    curr_data = df.iloc[:, i].tolist()
    if (i%100==0):
        print("Studying sample", i)
    [ols_slope, ols_chisq] = fit_ols_slope_and_chisq(curr_data)
    ols_slopes.append(ols_slope)
    ols_chisqs.append(ols_chisq)
    [brandt_slope, brandt_chisq] = fit_brandt_slope_and_chisq(curr_data)
    brandt_slopes.append(brandt_slope)
    brandt_chisqs.append(brandt_chisq)
    naive_slopes.append(fit_naive_slope(curr_data))


num_bins = 40
if (plot_distributions):
    plt.hist(ols_slopes, bins=num_bins)
    plt.xlabel('OLS-estimated slope')
    plt.ylabel('Frequency')
    plt.title('Distribution of OLS-estimated slopes sans read noise')
    plt.show()

    plt.hist(brandt_slopes, bins=num_bins)
    plt.xlabel('Brandt-estimated slope')
    plt.ylabel('Frequency')
    plt.title('Distribution of Brandt-estimated slopes sans read noise')
    plt.show()

    plt.hist(naive_slopes, bins=num_bins)
    plt.xlabel('Naive-estimated slope')
    plt.ylabel('Frequency')
    plt.title('Distribution of naive-estimated slopes sans read noise')
    plt.show()

# Summarize results, store as json

output_data = {}

print("\nOLS summary:")
slope_mean = statistics.mean(ols_slopes)
slope_stdev = statistics.stdev(ols_slopes)
slope_stderr = slope_stdev / math.sqrt(len(ols_slopes) - 1)
# One-sided significance of truefreq given measured mean + stderr
z = (slope_mean - true_freq/100.0) / slope_stderr
p = scipy.stats.norm.sf(abs(z))
output_data["ols_slope_mean"] = slope_mean
output_data["ols_slope_stdev"] = slope_stdev
output_data["ols_slope_stderr"] = slope_stderr
output_data["ols_slope_z"] = z
output_data["ols_slope_p"] = p
output_data["ols_chisq_mean"] = statistics.mean(ols_chisqs)
output_data["ols_chisq_stdev"] = statistics.stdev(ols_chisqs)
#output_data["ols_chisq"] = poisson_chisq(history=)
print(f"mean = {slope_mean}")
print(f"stdev = {slope_stdev}")
print(f"stderr = {slope_stderr}")
print(f"z = {z}")
print(f"p = {p}")
print(f"2-sigma confidence interval = [{slope_mean - 2*slope_stderr}, {slope_mean + 2*slope_stderr}]")

print("\nBrandt summary:")
slope_mean = statistics.mean(brandt_slopes)
slope_stdev = statistics.stdev(brandt_slopes)
slope_stderr = slope_stdev / math.sqrt(len(brandt_slopes) - 1)
# One-sided significance of truefreq given measured mean + stderr
z = (slope_mean - true_freq/100.0) / slope_stderr
p = scipy.stats.norm.sf(abs(z))
output_data["brandt_slope_mean"] = slope_mean
output_data["brandt_slope_stdev"] = slope_stdev
output_data["brandt_slope_stderr"] = slope_stderr
output_data["brandt_slope_z"] = z
output_data["brandt_slope_p"] = p
output_data["brandt_chisq_mean"] = statistics.mean(brandt_chisqs)
output_data["brandt_chisq_stdev"] = statistics.stdev(brandt_chisqs)
print(f"mean = {slope_mean}")
print(f"stdev = {slope_stdev}")
print(f"stderr = {slope_stderr}")
print(f"z = {z}")
print(f"p = {p}")
print(f"2-sigma confidence interval = [{slope_mean - 2*slope_stderr}, {slope_mean + 2*slope_stderr}]")

print("\nNaive summary:")
slope_mean = statistics.mean(naive_slopes)
slope_stdev = statistics.stdev(naive_slopes)
slope_stderr = slope_stdev / math.sqrt(len(naive_slopes) - 1)
# One-sided significance of truefreq given measured mean + stderr
z = (slope_mean - true_freq/100.0) / slope_stderr
p = scipy.stats.norm.sf(abs(z))
output_data["naive_slope_mean"] = slope_mean
output_data["naive_slope_stdev"] = slope_stdev
output_data["naive_slope_stderr"] = slope_stderr
output_data["naive_slope_z"] = z
output_data["naive_slope_p"] = p
print(f"mean = {slope_mean}")
print(f"stdev = {slope_stdev}")
print(f"stderr = {slope_stderr}")
print(f"z = {z}")
print(f"p = {p}")
print(f"2-sigma confidence interval = [{slope_mean - 2*slope_stderr}, {slope_mean + 2*slope_stderr}]")

stdev_ratio = statistics.stdev(ols_slopes) / statistics.stdev(brandt_slopes)
print(f"Ratio of OLS to Brandt spreads = {stdev_ratio}")
output_data["ols_brandt_spread_ratio"] = stdev_ratio

# Compute rms expected deviation between Brandt slope and true slope
true_cov = true_param_covariance(true_freq=true_freq, read_noise=read_noise,
                                 num_measurements=101)
# Factor of 100 because Brandt estimates per time step, not per unit time.
rms_formal_error = math.sqrt(true_cov[1][1]) / 100
output_data["rms_slope_error"] = rms_formal_error


json_object = json.dumps(output_data, indent=4)

with open(output_file, "w") as outfile:
    outfile.write(json_object)
