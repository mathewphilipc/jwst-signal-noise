import argparse
import random
import numpy as np

def simulate_multiaccum(freq, num_measurements, read_noise):
    """
    Simulates the data output of a MULTIACCUM process. We freely pick time units so the
    experiment has unit duration. We also have one frame per group (no frame averaging)
    and treat a single pixel. In the notation of arxiv.org:0706.2344 we have:
        (arXiv notation) = (local notation)
        f = freq
        sigma_read = read_noise
        m = 1 (no frame averaging)
        n = num_measurements
        tf = 1/(num_measurements - 1) (dt below)
        tg = tf (no frame average)

    Parameters:
    num_measurements (int): Number of experimental measurements taken.
    freq (float): Expected frequency of events per experiment.
    read_noise (float): Standard deviation of read noise (IID Gaussian).

    Returns:
    list: A list containing the cumulative event count at each measurement.
    Initial entry is t = 0, final is t = 1.0.
    """

    dt = 1.0 / (num_measurements - 1)
    exp_output = [0]
    for i in range(num_measurements - 1):
        exp_output.append(exp_output[i] + np.random.poisson(freq*dt))
    exp_output = [round(x + np.random.normal(0, read_noise)) for x in exp_output]
    return exp_output

def empirical_ols_multiaccum_statistics(read_noise, freq, num_measurements, num_trials):
    """
    Simulates many multiaccum processes, calculates fitted freq for each, and return sample mean + stddev.
    As above we treat a single pixel sans frame averaging and measurements start at t=0.0, end at 1.0.
    With enough trials this should agree with theoretical_ols_multiaccum_variance at
    frames_per_group=1, groups=num_measurements, frame_time=group_time=1.0/(num_measurements - 1).

    Parameters:
    num_measurements (int): Number of experimental measurements taken.
    freq (float): Expected frequency of events per experiment.
    read_noise (float): Standard deviation of read noise (IID Gaussian).

    Returns:
    array[float]: [mean, stddev] of all fitted slopes.

    """
    dt = 1.0 / (num_measurements - 1)
    fitted_slopes = []
    for sample in range(num_samples):
        exp_output = simulate_multiaccum(freq, num_measurements, read_noise)
        time_array = np.array([i*dt for i in range(num_measurements)])
        fitted_coeffs = np.polyfit(time_array, np.array(exp_output), 1)
        fitted_slopes.append(fitted_coeffs[0])

    return [np.mean(fitted_slopes), np.std(fitted_slope)]



def theoretical_ols_multiaccum_stddev(read_noise, freq, groups, frames_per_group, frame_time, group_time):
    """
    Predicts the sample stddev associated with running a MULTIACCUM process with
    exactly known parameters, then fitting by OLS to estimate the true freq value.
    cf arxiv.org:0706.2344 eq (1).

    Parameters:
    freq (float): Expected frequency of events per experiment (i.e., Poisson paramete)
    read_noise (float): Standard deviation of read noise (IID Gaussian).
    groups (int): Number of groups of frames (each measurement is an avg over a group).
    frames_per_group (int): number of frames in each group.
    frame_time (int): Time between adjacent frames.
    group_time (float): Time between the first frame in adjacent groups.

    Returns:
    float: The sample stddev described above.
    """

    n = groups
    m = frames_per_group
    tf = frame_time
    tg = group_time

    first_term = 12*(n - 1)*(read_noise**2)/(m*n*(n + 1))
    second_term = 6*(n**2 + 1)*(n - 1)*tg*freq/(5*n*(n + 1))
    third_term = -2*(2*m - 1)*(n - 1)*(m - 1)*tf*freq / (m*n*(n+1))

    return np.sqrt(first_term + second_term + third_term)

