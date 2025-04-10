from src.simulate_lib import simulate_multiaccum
import pandas as pd
print("hello")

freq = 100.0
num_measurements = 20
read_noise = 0.0

#monte_carlo_data = simulate_multiaccum(freq=100.0, num_measurements=100, read_noise = 0.0)
#print(monte_carlo_data)

data =[]
for i in range(10):
    curr_data = simulate_multiaccum(freq=freq, num_measurements=num_measurements, read_noise=read_noise)
    data.append(curr_data)

df = pd.DataFrame(data).T
print(df)


df.to_csv('data/poisson_simulations/freq_100_read_noise_0/simulations.csv')
