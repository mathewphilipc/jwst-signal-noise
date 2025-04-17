import pandas as pd
import numpy as np
from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
import matplotlib.pyplot as plt
import math
import statistics

# Estimate slope naively
def fit_naive_slope(data):
    num_measurements = len(data)
    diff = data[num_measurements - 1] - data[0]
    return diff / (num_measurements - 1)

# Estimate slope by OLS
def fit_ols_slope(data):
    n = len(curr_data)
    X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T
    ols_fit_params = np.linalg.lstsq(X, curr_data, rcond=None)[0] # (2,) nparr containing [slope, intercept]
    return float(ols_fit_params[0])

# Estimate slope naively
def fit_brandt_slope(data):
    n = len(curr_data)
    my_covar = Covar([s for s in range(n)])
    diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
    for t in range(1,len(diffs)+1):
        diffs[t-1] = curr_data[t] - curr_data[t-1]
    ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=20.1, rescale=False)
    return ramp_result.countrate[0]

df = pd.read_csv("data/poisson_simulations/freq_100_read_noise_0/simulations.csv", index_col=0)
num_exps = len(df.columns)
ols_slopes = []
ols_intercepts = []
brandt_slopes = []
naive_slopes = []

for i in range(num_exps):
    curr_data = df.iloc[:, i].tolist()
    if (i%100==0):
        print("Studying sample", i)
    ols_slopes.append(fit_ols_slope(curr_data))
    brandt_slopes.append(fit_brandt_slope(curr_data))
    naive_slopes.append(fit_naive_slope(curr_data))


num_bins = 40

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


print("\nOLS summary:")
mean = statistics.mean(ols_slopes)
stdev = statistics.stdev(ols_slopes)
stderr = stdev / math.sqrt(len(ols_slopes) - 1)
print(f"mean = {mean}")
print(f"stdev = {stdev}")
print(f"stderr = {stderr}")
print(f"2-sigma confidence interval = [{mean - 2*stderr}, {mean + 2*stderr}]")

print("\nBrandt summary:")
mean = statistics.mean(brandt_slopes)
stdev = statistics.stdev(brandt_slopes)
stderr = stdev / math.sqrt(len(brandt_slopes) - 1)
print(f"mean = {mean}")
print(f"stdev = {stdev}")
print(f"stderr = {stderr}")
print(f"2-sigma confidence interval = [{mean - 2*stderr}, {mean + 2*stderr}]")

print("\nNaive summary:")
mean = statistics.mean(naive_slopes)
stdev = statistics.stdev(naive_slopes)
stderr = stdev / math.sqrt(len(naive_slopes) - 1)
print(f"mean = {mean}")
print(f"stdev = {stdev}")
print(f"stderr = {stderr}")
print(f"2-sigma confidence interval = [{mean - 2*stderr}, {mean + 2*stderr}]")
