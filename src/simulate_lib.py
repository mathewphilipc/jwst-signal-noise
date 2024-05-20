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
    list: A list containing the cumulative event count at each measurement. Initial entry is the start of Poisson process, but still has read noise.
    """

    dt = 1.0 / (num_measurements - 1)
    exp_output = [np.random.poisson(freq*dt)]
    for i in range(num_measurements - 1):
        exp_output.append(exp_output[i] + np.random.poisson(freq*dt))
    exp_output = [round(x + np.random.normal(0, read_noise)) for x in exp_output]
    return exp_output

