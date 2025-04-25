from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math
import random

read_noise = 12.3

#fits_image_filename = '../blanton-project/images/apR-a-28580010.fits'
fits_image_filename = '../blanton-project/images/apR-a-28580052.fits'
print("Opening fits file...")
hdul = fits.open(fits_image_filename)

my_image = hdul[1].data

mid_over_time = []
final_image = []

n = 47

# An interesting subimage to plot
[y_min, y_max]=[20, 80]
[x_min, x_max] = [1920,1980]

y_len = y_max - y_min
x_len = x_max - x_min

image_history = []

print("Studying images...")
final_image = []
for i in range(1,n+1):
    curr_image = hdul[i].data
    print(f"Checking image {i}")
    image_history.append(curr_image)

image_history = np.asarray(image_history).astype(np.int64)
print(f"Original history shape = {image_history.shape}")
print(f"New history shape = {image_history.shape}")

naive_reconstruction = np.zeros((y_len, x_len))
brandt_reconstruction = np.zeros((y_len, x_len))

ols_ssre_list = []
brandt_ssre_list = []
ols_slope_list = []

for i in range(x_len):
    for j in range(y_len):
        X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T

        pixel_history = image_history[:, y_min + j, x_min + i]
        ols_fit_params = np.linalg.lstsq(X, pixel_history, rcond=None)[0] # (2,) nparr containing [slope, intercept]
        naive_reconstruction[j][i] = ols_fit_params[0]
        print(f"\nOLS estimated count rate slope = {ols_fit_params[0]}")
        ols_slope_list.append(ols_fit_params[0])

        # Computed sum of squared relative errors for OLS
        predicted_values = X @ ols_fit_params
        residuals = pixel_history - predicted_values
        ssre_statistic = math.sqrt(np.sum((residuals / pixel_history)**2))
        print(f"OLS ssre = {ssre_statistic}")
        ols_ssre_list.append(ssre_statistic)

        fit_intercept = True
        my_covar = Covar(read_times=[s + 1 for s in range(n)], pedestal=fit_intercept)
        diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
        for t in range(1,len(diffs)+1):
            diffs[t-1] = pixel_history[t] - pixel_history[t-1]
        if (fit_intercept):
            diffs = np.insert(diffs, 0, pixel_history[0], axis=0)
        ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=read_noise, rescale=True)
        brandt_fit_slope = ramp_result.countrate[0]
        print(f"Brandt estimated count rate = {brandt_fit_slope}")
        brandt_fit_intercept = ramp_result.pedestal[0]
        brandt_reconstruction[j][i] = brandt_fit_slope

        # Computed sum of squared relative errors for Brandt
        predicted_values = X @ np.array([brandt_fit_slope, brandt_fit_intercept])
        residuals = pixel_history - predicted_values
        ssre_statistic = math.sqrt(np.sum((residuals / pixel_history)**2))
        print(f"Brandt ssre = {ssre_statistic}")
        brandt_ssre_list.append(ssre_statistic)

print(f"\n\n\nMean OLS ssre = {np.median(ols_ssre_list)}")
print(f"\n\n\nMean brandt bootstrap ssre = {np.median(brandt_ssre_list)}")
ssre_ratios = [(brandt_ssre_list[i] / ols_ssre_list[i]) for i in range(len(ols_ssre_list))]
print(f"\n\n\nList of ssre ratios:")
print(f"\n\n\nmedian ssre_ratio = {np.median(ssre_ratios)}")
print(f"\n\n\nmean ssre_ratio = {np.mean(ssre_ratios)}")
print(f"\n\n\nmin sqrt_ratio = {np.min(ssre_ratios)}")
print(f"\n\n\nSorted list of ratios: {sorted(ssre_ratios)}")


plt.hist(sqre_ratios, bins=100)
plt.xlabel('Ratio of brandt vs OLS ssre ratios')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()


x = ols_slope_list
y = ssre_ratios
plt.scatter(x,y)
plt.xlabel('OLS-fitted slope')
plt.ylabel('ratio of ssres')
plt.title('OLS-fitted slope vs ratio of (Brandt vs OLS) ssres')
plt.show()





final_image = hdul[n].data.astype(np.int64)
initial_image = hdul[1].data.astype(np.int64)

difference_image = final_image - initial_image
difference_image = difference_image[20:80, 1920:1980]

hdul.close()



print("Plotting original subimage...")
plt.imshow(difference_image, cmap='hot', interpolation='nearest')
plt.show()

print("Plotting OLS reconstructed subimage...")
plt.imshow(naive_reconstruction, cmap='hot', interpolation='nearest')
plt.show()

print("Plotting Brandt reconstructed subimage...")
plt.imshow(brandt_reconstruction, cmap='hot', interpolation='nearest')
plt.show()
