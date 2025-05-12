print("lets extract read noise")

from astropy.io import fits
import math
import numpy as np

fits_image_filename = '../blanton-project/images/apR-a-28580052.fits'
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

final_image = []
for i in range(1,n+1):
    curr_image = hdul[i].data
    print(f"Checking image {i}")
    image_history.append(curr_image)
image_history = np.asarray(image_history).astype(np.int64)

sample_read_noises = []
for i in range(x_len):
    for j in range(y_len):
        X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T

        pixel_history = image_history[:, y_min + j, x_min + i]
        ols_fit_params = np.linalg.lstsq(X, pixel_history, rcond=None)[0] # (2,) nparr containing [slope, intercept]
        ols_slope = ols_fit_params[0]
        if (-0.1 < ols_slope < 0.1):
            print(f"\nOLS estimated count rate slope = {ols_slope}")
            print(f"Pixel history mean = {np.mean(pixel_history)}")
            print(f"Sample read noise = {np.std(pixel_history)}")
            sample_read_noises.append(np.std(pixel_history))

print(f"Mean empirical read noise = {np.mean(sample_read_noises)}")
print(f"Sample standard deviation = {np.std(sample_read_noises)}")
print(f"Sample standard error = {np.std(sample_read_noises)/math.sqrt(len(sample_read_noises) - 1)}")
final_image = hdul[n].data.astype(np.int64)
initial_image = hdul[1].data.astype(np.int64)

difference_image = final_image - initial_image
difference_image = difference_image[20:80, 1920:1980]

hdul.close()






