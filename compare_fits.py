import pandas as pd
import numpy as np
from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv("data/poisson_simulations/freq_100_read_noise_0/simulations.csv", index_col=0)
print(df)
print("num columns:")
print(len(df.columns))
num_exps = len(df.columns)
ols_slopes = []
ols_intercepts = []
brandt_slopes = []

for i in range(num_exps):
    curr_data = df.iloc[:, i].tolist()
    #print("\nCurrent data:")
    #print(curr_data)
    if (i%100==0):
        print("Studying sample", i)

    #print("Estimating from OLS...")
    n = len(curr_data)
    X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T
    ols_fit_params = np.linalg.lstsq(X, curr_data, rcond=None)[0] # (2,) nparr containing [slope, intercept]
    #print(f"OLS-estimated countrate slope = {ols_fit_params[0]}")
    ols_slopes.append(float(ols_fit_params[0]))
    #print(f"OLS-estimated countrate intercept = {ols_fit_params[1]}")
    ols_intercepts.append(float(ols_fit_params[1]))

    #print("Estimating Brandt-style...")
    my_covar = Covar([s for s in range(n)])
    diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
    for t in range(1,len(diffs)+1):
        diffs[t-1] = curr_data[t] - curr_data[t-1]
    ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=20.1, rescale=False)
    brandt_fit_slope = ramp_result.countrate[0]
    #print(f"Brandt estimated count rate = {brandt_fit_slope}")
    brandt_slopes.append(brandt_fit_slope)

#print("\n\n\n\nOLS slopes:")
#print(ols_slopes)
#print("OLS intercepts:")
#print(ols_intercepts)

num_bins = 20

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

print("OLS summary:")
print("mean:", statistics.mean(ols_slopes))
print("stdev:", statistics.stdev(ols_slopes))

print("Brandt summary:")
print("mean:", statistics.mean(brandt_slopes))
print("stdev:", statistics.stdev(brandt_slopes))

