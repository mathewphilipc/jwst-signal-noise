import argparse
import random
import numpy as np

def simulate_poisson_process(freq, num_measurements):
    """
    Simulates the data output of a Poisson process. We freely pick time units so the
    experiment has unit duration.

    Parameters:
    num_measurements (int): Number of experimental measurements taken.
    freq (float): Expected frequency of events per experiment.

    Returns:
    list: A list containing the cumulative event count at each measurement (including trivial initial 0).
    """

    dt = 1.0 / num_measurements
    exp_output = [0]
    for i in range(num_measurements):
        exp_output.append(exp_output[i] + np.random.poisson(freq*dt))

    return exp_output

