import argparse
import random
import numpy as np

def simulate_multiaccum(freq, num_measurements, read_noise=0):
    """
    Simulates the data output of a MULTIACCUM process. We freely pick time units so the
    experiment has unit duration. In the notation of arxiv.org:0706.2344 we have:
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
    Initial entry is the start of Poisson process, but still has read noise.
    """

    dt = 1.0 / (num_measurements - 1)
    exp_output = [np.random.poisson(freq*dt)]
    for i in range(num_measurements - 1):
        exp_output.append(exp_output[i] + np.random.poisson(freq*dt))
    exp_output = [round(x + np.random.normal(0, read_noise)) for x in exp_output]
    return exp_output

def multiaccum_variance(read_noise, freq, groups, frames_per_group, frame_time, group_time):
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

    return first_term + second_term + third_term

