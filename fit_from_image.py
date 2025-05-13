from fitramp.fitramp import Covar, fit_ramps
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math
from src.simulate_lib import poisson_chisq

# Estimate slope by OLS
def fit_ols_slope_and_chisq(data):
    n = len(data)
    X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T
    ols_fit_params = np.linalg.lstsq(X, data, rcond=None)[0] # (2,) nparr containing [slope, intercept]
    slope = float(ols_fit_params[0])
    intercept = float(ols_fit_params[1])
    chisq = poisson_chisq(history=data, intercept=intercept, slope=slope, read_noise=read_noise)
    return [slope, chisq]

# Estimate slope naively
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

ols_slope_list = []
brandt_chisq_list = []
ols_chisq_list = []

for i in range(x_len):
    for j in range(y_len):
        X = np.vstack([np.linspace(1, n, n), np.ones(n)]).T

        pixel_history = image_history[:, y_min + j, x_min + i]
        [ols_slope, ols_chisq] = fit_ols_slope_and_chisq(pixel_history)
        ols_slope_list.append(ols_slope)
        ols_chisq_list.append(ols_chisq)
        naive_reconstruction[j][i] = ols_slope

        [brandt_slope, brandt_chisq] = fit_ols_slope_and_chisq(pixel_history)
        brandt_chisq_list.append(brandt_chisq)
        brandt_reconstruction[j][i] = brandt_slope

print(f"\n\n\nMean OLS chisq = {np.median(ols_chisq_list)}")
print(f"\n\n\nMean brandt bootstrap chisq = {np.median(brandt_chisq_list)}")
chisq_ratios = [(brandt_chisq_list[i] / ols_chisq_list[i]) for i in range(len(ols_chisq_list))]
print(f"\n\n\nList of chisq ratios:")
print(f"\n\n\nmedian chisq_ratio = {np.median(chisq_ratios)}")
print(f"\n\n\nmean chisq_ratio = {np.mean(chisq_ratios)}")
print(f"\n\n\nmin sqrt_ratio = {np.min(chisq_ratios)}")
print(f"\n\n\nSorted list of ratios: {sorted(chisq_ratios)}")


plt.hist(schisq_ratios, bins=100)
plt.title('Frequency histogram of Brandt:OLS chisq ratio')
plt.ylabel('Frequency')
plt.xlabel('chisq ratio')
plt.savefig(
    "data/plots/chisq_ratio_frequency.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


x = ols_slope_list
y = chisq_ratios
plt.scatter(x,y)
plt.xlabel('OLS-estimated rate')
plt.ylabel('Ratio of chisqs')
plt.title('Brandt:OLS chisq ratio vs OLS-estimated rate')
plt.savefig(
    "data/plots/chisq_ratio_versus_rate.png",
    dpi=300,
    bbox_inches="tight"
)
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
