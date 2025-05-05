import argparse
import random
import numpy as np
from numpy.linalg import inv

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
    # read_noise = scale = stdev of Gaussian noise
    exp_output = [round(x + np.random.normal(loc=0, scale=read_noise)) for x in exp_output]
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

def true_param_covariance(true_freq, read_noise, num_measurements):
    """
    Given a true frequency, a true read_noise, and a number of measurements
    (taken to be uniform over a unit time interval, including at the boundaries
    t=0.0 and t=1.0), computes (A^T C^{-1} A)^{-1} where A is the design matrix
    and C is the true correlation matrix bewteen measurements.
    """
    # Time step between measurements 
    dt = 1/(num_measurements - 1)

    # Design matrix
    A = np.zeros(shape=(num_measurements, 2))
    for i in range(num_measurements):
        A[i][0] = 1
        A[i][1] = i*dt

    # Pure Poisson contribute to covariance
    C_poisson = np.zeros(shape=(num_measurements, num_measurements))
    for i in range(num_measurements):
        for j in range(num_measurements):
            C_poisson[i][j] = true_freq*dt*min(i,j)

    # Gaussian read noise contribution to covariance
    C_noise = np.zeros(shape=(num_measurements, num_measurements))
    for i in range(num_measurements):
        C_noise[i][i] = read_noise

    return inv(np.transpose(A) @ inv(C_noise + C_poisson) @ A)

def poisson_chisq(history, intercept, slope, read_noise):
    """
    Given a history of measurements up the ramp of a multiaccum process, a
    fitted slope and intercept, and a read noise, computes chisq (that is, -1/2
    of the log(likelihood)) for the fit. Note that several contributions to
    chisq are directly functions only of the covariance matrix, not of the
    intercept and slope, and thus are constants sometimes thrown away when
    extremizing w.r.t. intercept and slope. We retain these explicitly.

    Parameters:
    history (list): List of floats corresponding to measurements up the ramp.
    intercept (float): Fitted intercepted.
    slope (float): Fitted slope.
    read_noise (float): Read noise.

    Returns:
    float: The fit's chi squared value.
    """
    N = len(history)
    hist_arr = (np.array(history, dtype=float))[:, np.newaxis]

    # Negative slopes are nonsense. Hacky solution is to replace them with 0.
    slope = max(slope,0)

    # Design matrix (assuming time units where one time step is unity).
    A = np.zeros(shape=(N, 2))
    for i in range(N):
        A[i][0] = 1
        A[i][1] = i
    x = np.array([[intercept], [slope]])

    # Covariance matrix
    # Pure Poisson contribute to covariance
    C_poisson = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            C_poisson[i][j] = slope*min(i,j)

    # Gaussian read noise contribution to covariance
    C_noise = np.zeros(shape=(N, N))
    for i in range(N):
        C_noise[i][i] = read_noise

    C =  C_poisson + C_noise

    #currdet = np.linalg.det(C)
    #print(f"Covariance det = {currdet}")
    # if (currdet < 0):
    #    eigenvalues, eigenvectors = np.linalg.eig(C)
    #    print(eigenvalues)
    #print(f"\n\n\n\nA has shape {A.shape}")
    #print(f"x has shape {x.shape}")
    #print(f"Ax has shape {(A@x).shape}")
    #print(f"data has shape {hist_arr.shape}")
    #print(f"\n\n\n\n(Ax - hist)^T = {(np.transpose(hist_arr - A@x)).shape}")
    #print(f"Second shape = {C.shape}")
    #print(f"Third shape = {(hist_arr - A@x).shape}")
    #print(f"Overall shape = {(np.transpose(hist_arr - A@x)@C@(hist_arr - A@x)).shape}")


    result = (np.transpose(hist_arr - A@x)@C@(hist_arr - A@x))[0][0] + np.log(np.linalg.det(C)) + N*np.log(np.pi)
    return result