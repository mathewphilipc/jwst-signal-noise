from src.simulate_lib import simulate_multiaccum
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--truefreq", 
                    type=int, 
                    required=True, 
                    help="True (simulated) frequency"
)

parser.add_argument("--readnoise",
                    type=int, 
                    required=True, 
                    help="Standard deviation of simulated read noise"
)

args = parser.parse_args()

freq = args.truefreq
read_noise = args.readnoise

num_measurements = 101 # Includes measurements at exactly t=0.0 and t=1.0.
num_samples = 10000 # Generally 10,000

data =[]
for i in range(num_samples):
    curr_data = simulate_multiaccum(freq=freq, num_measurements=num_measurements, read_noise=read_noise)
    data.append(curr_data)

df = pd.DataFrame(data).T
print(df)

output_dir = "data/poisson_simulations/freq_" + str(freq) + "_read_noise_" + str(read_noise)
print(f"Output directory = {output_dir}")
output_file = output_dir + "/simulations.csv"
if not os.path.exists(output_dir):
    print("Creating directory...")
    os.makedirs(output_dir)
df.to_csv(output_file)
