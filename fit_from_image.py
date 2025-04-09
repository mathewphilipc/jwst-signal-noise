from fitramp.fitramp import Covar, Ramp_Result, fit_ramps
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import random

sigma_read = 20.9

#fits_image_filename = 'images/apR-a-28580010.fits'
fits_image_filename = 'images/apR-a-28580052.fits'
print("Opening fits file...")
hdul = fits.open(fits_image_filename)

my_image = hdul[1].data

mid_over_time = []
final_image = []

n = 47

# An interesting subimage to plot
[y_min, y_max]=[20, 80]
[x_min, x_max] = [1920,1980]
#[y_min, y_max] = [0, 500]
#[x_min, x_max] = [1750, 2250]

y_len = y_max - y_min
x_len = x_max - x_min

#image_history = np.zeros((n,y_max - y_min + 1, x_min - x_min + 1)).astype(np.int64)
image_history = []

print("Studying images...")
final_image = []
for i in range(1,n+1):
    curr_image = hdul[i].data
    print(f"Checking image {i}")
    image_history.append(curr_image)

image_history = np.asarray(image_history).astype(np.int64)
print(f"Original history shape = {image_history.shape}")
#image_history = image_history[:, y_min:y_max, x_min:x_max]
print(f"New history shape = {image_history.shape}")

naive_reconstruction = np.zeros((y_len, x_len))
brandt_reconstruction = np.zeros((y_len, x_len))

ols_sqre_history = []
brandt_sqre_history = []
brandt_native_chisq_history = []

for i in range(x_len):
    for j in range(y_len):
        X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T

        pixel_history = image_history[:, y_min + j, x_min + i]
        ols_fit_params = np.linalg.lstsq(X, pixel_history, rcond=None)[0] # (2,) nparr containing [slope, intercept]
        naive_reconstruction[j][i] = ols_fit_params[0]
        print(f"\nOLS estimated count rate slope = {ols_fit_params[0]}")
        ols_fit_intercept = ols_fit_params[1]

        # Computed chi-squared
        predicted_values = X @ ols_fit_params
        residuals = pixel_history - predicted_values
        sqre_squared_statistic = np.sum((residuals**2) / predicted_values)
        print(f"OLS sqre = {sqre_squared_statistic}")
        ols_sqre_history.append(sqre_squared_statistic)

        my_covar = Covar([s for s in range(n)])
        diffs = np.ndarray(shape=(n-1,1), dtype=np.int64)
        for t in range(1,len(diffs)+1):
            diffs[t-1] = pixel_history[t] - pixel_history[t-1]
        ramp_result = fit_ramps(diffs = diffs, Cov = my_covar, sig=20.1, rescale=False)
        brandt_fit_slope = ramp_result.countrate[0]
        print(f"Brandt estimated count rate = {brandt_fit_slope}")
        brandt_reconstruction[j][i] = brandt_fit_slope

        # Computed chi-squared
        predicted_values = X @ np.array([brandt_fit_slope, ols_fit_intercept])
        print(f"predicted values has type {type(predicted_values)}")
        residuals = pixel_history - predicted_values
        sqre_statistic = np.sum((residuals**2) / predicted_values)
        print(f"Brandt sqre = {sqre_statistic}")
        print(f"Brandt chi-squared = {ramp_result.chisq}")
        brandt_sqre_history.append(sqre_squared_statistic)
        brandt_native_chisq_history.append(ramp_result.chisq)

print(f"\n\n\nMean OLS sqre = {np.median(ols_sqre_history)}")
print(f"\n\n\nMean brandt bootstrap sqre = {np.median(brandt_sqre_history)}")
print(f"\n\n\nMean brandt native chi squared = {np.median(brandt_native_chisq_history)}\n\n\n")


#final_image = hdul[n].data[y_min:y_max, x_min:x_max].astype(np.int64)
#initial_image = hdul[1].data[y_min:y_max, x_min:x_max].astype(np.int64)
final_image = hdul[n].data.astype(np.int64)
initial_image = hdul[1].data.astype(np.int64)

difference_image = final_image - initial_image
difference_image = difference_image[20:80, 1920:1980]

hdul.close()


#print(final_image.shape)

print("Plotting original subimage...")
plt.imshow(difference_image, cmap='hot', interpolation='nearest')
plt.show()

print("Plotting OLS reconstructed subimage...")
plt.imshow(naive_reconstruction, cmap='hot', interpolation='nearest')
plt.show()

print("Plotting Brandt reconstructed subimage...")
plt.imshow(brandt_reconstruction, cmap='hot', interpolation='nearest')
plt.show()



t = np.linspace(1,n-1,n-1)
fig, ax = plt.subplots()
ax.plot(t, diffs)
ax.set(xlabel='time', ylabel='voltage',
       title='Voltage over time for top-left pixel')
ax.grid()
fig.savefig("test.png")
plt.show()

